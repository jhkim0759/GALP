"""
Prompt-guided COCO indoor scene filtering with Qwen3.5.

This script keeps indoor room scenes that:
1. are not close-up crops,
2. show room-level context,
3. contain many visible objects.

Compared to the older CLIP-only filter, this pipeline combines:
- mask/object-count prefiltering,
- prompt-based Qwen3.5-VL judgement,
- resumable JSONL cache for expensive vision inference.

Example:
    python data_toolkit/filter_coco_indoor.py \
        --index /mnt/storage/jhkim/coco/train2017/index_filtered.json \
        --output /mnt/storage/jhkim/coco/train2017/index_qwen_filtered.json \
        --qwen-model-path ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/<snapshot> \
        --prefilter-min-parts 12 \
        --prefilter-max-largest-mask-ratio 0.35 \
        --prefilter-max-total-mask-ratio 0.70
"""

import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


DEFAULT_QWEN_PATH = (
    "/home/kimj0010/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-27B/snapshots/"
    "b7ca741b86de18df552fd2cc952861e04621a4bd"
)


def compute_mask_areas(mask_paths: list[str]) -> list[float]:
    """Return per-mask area ratio in [0, 1]."""
    ratios = []
    for mask_path in mask_paths:
        try:
            image = np.array(Image.open(mask_path))
            total = image.shape[0] * image.shape[1]
            area = (image > 127).sum()
            ratios.append(area / total if total > 0 else 0.0)
        except Exception:
            ratios.append(0.0)
    return ratios


def get_mask_areas(scene: dict[str, Any]) -> list[float]:
    cached = scene.get("mask_area_ratios")
    if isinstance(cached, list) and len(cached) == len(scene["mask_paths"]):
        return [float(v) for v in cached]
    return compute_mask_areas(scene["mask_paths"])


def default_output_path(index_path: str) -> str:
    path = Path(index_path)
    return str(path.with_name(f"{path.stem}_qwen_filtered.json"))


def resolve_dtype(dtype_name: str, cuda_available: bool) -> Any:
    if dtype_name == "auto":
        return torch.bfloat16 if cuda_available else torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def sanitize_qwen35_config(model_path: str):
    from transformers import Qwen3VLConfig

    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    cfg["model_type"] = "qwen3_vl"
    cfg["architectures"] = ["Qwen3VLForConditionalGeneration"]

    if isinstance(cfg.get("vision_config"), dict):
        cfg["vision_config"]["model_type"] = "qwen3_vl"

    text_cfg = cfg.get("text_config", {})
    if isinstance(text_cfg, dict):
        text_cfg["model_type"] = "qwen3_vl_text"
        rope_params = text_cfg.pop("rope_parameters", None)
        if isinstance(rope_params, dict):
            text_cfg["rope_theta"] = rope_params.get(
                "rope_theta",
                text_cfg.get("rope_theta", 10_000_000),
            )
            text_cfg["rope_scaling"] = {
                "rope_type": rope_params.get("rope_type", "default"),
                "mrope_section": rope_params.get("mrope_section", [24, 20, 20]),
            }
        elif text_cfg.get("rope_scaling") is None:
            text_cfg["rope_scaling"] = {
                "rope_type": "default",
                "mrope_section": [24, 20, 20],
            }

    return Qwen3VLConfig.from_dict(cfg)


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        value = int(value)
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


def extract_json_like_fields(text: str) -> dict[str, Any]:
    patterns = {
        "indoor_room_score": r'"indoor_room_score"\s*:\s*([0-9.]+)',
        "wide_view_score": r'"wide_view_score"\s*:\s*([0-9.]+)',
        "object_density_score": r'"object_density_score"\s*:\s*([0-9.]+)',
        "confidence": r'"confidence"\s*:\s*([0-9.]+)',
        "keep": r'"keep"\s*:\s*(true|false)',
        "reason": r'"reason"\s*:\s*"([^"]*)',
    }

    extracted = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1)
        if key == "keep":
            extracted[key] = value.lower() == "true"
        else:
            extracted[key] = value
    return extracted


def canonicalize_keys(data: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in data.items():
        normalized[str(key).strip()] = value
    return normalized


def coerce_score(value: Any, default: int) -> int:
    try:
        numeric = float(value)
    except Exception:
        return default
    if 0.0 < numeric <= 1.0:
        numeric *= 5.0
    return max(0, min(5, int(round(numeric))))


def coerce_confidence(value: Any, default: int) -> int:
    try:
        numeric = float(value)
    except Exception:
        return default
    if 0.0 < numeric <= 1.0:
        numeric *= 5.0
    return max(1, min(5, int(round(numeric))))


def normalize_qwen_result(raw_text: str, parsed: dict[str, Any] | None) -> dict[str, Any]:
    parsed_ok = parsed is not None
    parsed = canonicalize_keys(parsed or {})
    fallback = extract_json_like_fields(raw_text)
    merged = {**fallback, **parsed}

    indoor_room_score = coerce_score(merged.get("indoor_room_score"), 0)
    wide_view_score = coerce_score(merged.get("wide_view_score"), 0)
    object_density_score = coerce_score(merged.get("object_density_score"), 0)
    confidence = coerce_confidence(merged.get("confidence"), 1)

    keep_value = merged.get("keep")
    if isinstance(keep_value, bool):
        keep = keep_value
    elif isinstance(keep_value, str):
        keep = keep_value.strip().lower() == "true"
    else:
        keep = False

    reason = merged.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    reason = reason.strip()

    return {
        "indoor_room_score": indoor_room_score,
        "wide_view_score": wide_view_score,
        "object_density_score": object_density_score,
        "confidence": confidence,
        "model_keep": keep,
        "reason": reason[:240],
        "raw_response": raw_text.strip()[:8000],
        "parse_ok": parsed_ok or bool(fallback),
    }


def build_scene_prompt(scene: dict[str, Any]) -> str:
    return f"""
You are curating a dataset of indoor room scenes.

Judge this image conservatively for these requirements:
1. It should be an indoor room scene, not outdoor, not a product shot, and not a tight crop of one object.
2. The camera/view should not be too close. Prefer a medium-wide or wide room view with visible spatial context.
3. The image should contain many visible objects, furniture, or room items.

Annotation hints after mask filtering:
- kept_object_count: {scene["num_parts"]}
- largest_object_mask_ratio: {scene["largest_mask_ratio"]:.4f}
- total_mask_ratio: {scene["total_mask_ratio"]:.4f}

Use the image as primary evidence. Use the annotation hints only as supporting signals.

Score rubric:
- indoor_room_score: 0 to 5
- wide_view_score: 0 to 5
- object_density_score: 0 to 5
- confidence: 1 to 5

Return STRICT JSON ONLY with exactly these keys:
{{
  "indoor_room_score": 0,
  "wide_view_score": 0,
  "object_density_score": 0,
  "keep": false,
  "confidence": 1,
  "reason": ""
}}
""".strip()


class QwenSceneEvaluator:
    def __init__(
        self,
        model_path: str,
        dtype_name: str,
        max_new_tokens: int,
        offload_folder: str | None = None,
    ):
        self.model_path = model_path
        self.dtype_name = dtype_name
        self.max_new_tokens = max_new_tokens
        self.offload_folder = offload_folder
        self.processor = None
        self.model = None
        self.model_device = torch.device("cpu")
        self.model_dtype = torch.float32
        self.use_accelerate_hooks = False

    def load(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        cuda_available = torch.cuda.is_available()
        runtime_dtype = resolve_dtype(self.dtype_name, cuda_available)

        load_kwargs = {
            "local_files_only": True,
            "low_cpu_mem_usage": True,
        }
        if self.offload_folder:
            os.makedirs(self.offload_folder, exist_ok=True)
            load_kwargs["offload_folder"] = self.offload_folder

        model_kwargs = {
            "device_map": "auto",
            "attn_implementation": "sdpa",
        }

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            if hasattr(self.processor, "tokenizer"):
                self.processor.tokenizer.padding_side = "left"
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_path,
                    dtype=runtime_dtype,
                    **model_kwargs,
                    **load_kwargs,
                )
            except TypeError:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_path,
                    torch_dtype=runtime_dtype,
                    **model_kwargs,
                    **load_kwargs,
                )
        except Exception:
            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

            config = sanitize_qwen35_config(self.model_path)
            self.processor = Qwen3VLProcessor.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            self.processor.tokenizer.padding_side = "left"
            try:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    config=config,
                    dtype=runtime_dtype,
                    **model_kwargs,
                    **load_kwargs,
                )
            except TypeError:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    config=config,
                    torch_dtype=runtime_dtype,
                    **model_kwargs,
                    **load_kwargs,
                )
        self.model.eval()

        hf_device_map = getattr(self.model, "hf_device_map", None)
        self.use_accelerate_hooks = hf_device_map is not None
        if not self.use_accelerate_hooks:
            for param in self.model.parameters():
                if param.device.type != "meta":
                    self.model_device = param.device
                    self.model_dtype = param.dtype
                    break
        else:
            for param in self.model.parameters():
                if param.device.type != "meta":
                    self.model_dtype = param.dtype
                    break

    def evaluate_batch(self, scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assert self.processor is not None and self.model is not None
        if not scenes:
            return []

        conversations = []
        image_paths = []
        for scene in scenes:
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": scene["image_path"]},
                            {"type": "text", "text": build_scene_prompt(scene)},
                        ],
                    }
                ]
            )
            image_paths.append(scene["image_path"])

        texts = [
            self.processor.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for conversation in conversations
        ]

        inputs = self.processor(
            images=image_paths,
            text=texts,
            return_tensors="pt",
            padding=True,
        )

        for key in ("pixel_values", "pixel_values_videos"):
            if key in inputs and torch.is_tensor(inputs[key]) and torch.is_floating_point(inputs[key]):
                inputs[key] = inputs[key].to(self.model_dtype)

        if not self.use_accelerate_hooks:
            inputs = {
                key: value.to(self.model_device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        trimmed_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        raw_outputs = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        results = []
        for raw_text in raw_outputs:
            parsed = extract_first_json_object(raw_text)
            results.append(normalize_qwen_result(raw_text, parsed))
        return results


def load_qwen_cache(cache_path: str | None) -> dict[int, dict[str, Any]]:
    if not cache_path or not os.path.exists(cache_path):
        return {}

    cache = {}
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = row.get("image_id")
            if image_id is None:
                continue
            cache[int(image_id)] = row
    return cache


def append_qwen_cache(cache_path: str | None, image_id: int, result: dict[str, Any]):
    if not cache_path:
        return
    row = {"image_id": int(image_id), **result}
    with open(cache_path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_scene(scene: dict[str, Any], args) -> tuple[dict[str, Any] | None, dict[str, int]]:
    stats_delta = {
        "objects_removed_small_mask": 0,
        "scenes_removed_few_parts": 0,
    }

    areas = get_mask_areas(scene)
    keep_indices = [i for i, area in enumerate(areas) if area >= args.min_mask_ratio]
    stats_delta["objects_removed_small_mask"] = len(areas) - len(keep_indices)

    if len(keep_indices) < args.min_num_parts:
        stats_delta["scenes_removed_few_parts"] = 1
        return None, stats_delta

    kept_areas = [float(areas[i]) for i in keep_indices]
    prepared = {
        "image_id": scene["image_id"],
        "image_path": scene["image_path"],
        "pointmap_path": scene.get("pointmap_path"),
        "mesh_paths": [scene["mesh_paths"][i] for i in keep_indices],
        "pose_paths": [scene["pose_paths"][i] for i in keep_indices],
        "mask_paths": [scene["mask_paths"][i] for i in keep_indices],
        "ann_ids": [scene["ann_ids"][i] for i in keep_indices],
        "num_parts": len(keep_indices),
        "mask_area_ratios": [round(areas[i], 6) for i in keep_indices],
        "largest_mask_ratio": max(kept_areas) if kept_areas else 0.0,
        "total_mask_ratio": float(sum(kept_areas)),
        "avg_mask_ratio": float(np.mean(kept_areas)) if kept_areas else 0.0,
    }
    return prepared, stats_delta


def passes_prefilter(scene: dict[str, Any], args) -> tuple[bool, str | None]:
    if scene["num_parts"] < args.prefilter_min_parts:
        return False, "prefilter_few_objects"
    if scene["largest_mask_ratio"] > args.prefilter_max_largest_mask_ratio:
        return False, "prefilter_closeup"
    if scene["total_mask_ratio"] > args.prefilter_max_total_mask_ratio:
        return False, "prefilter_overfilled"
    if scene["total_mask_ratio"] < args.prefilter_min_total_mask_ratio:
        return False, "prefilter_too_sparse"
    return True, None


def final_keep_from_qwen(result: dict[str, Any], args) -> bool:
    return (
        result["indoor_room_score"] >= args.min_indoor_room_score
        and result["wide_view_score"] >= args.min_wide_view_score
        and result["object_density_score"] >= args.min_object_density_score
        and result["confidence"] >= args.min_qwen_confidence
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Qwen3.5-based COCO indoor room filtering")
    parser.add_argument(
        "--index",
        type=str,
        default="/mnt/storage/jhkim/coco/train2017/index_filtered.json",
        help="Input index.json path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: <index>_qwen_filtered.json)",
    )
    parser.add_argument(
        "--qwen-model-path",
        type=str,
        default=DEFAULT_QWEN_PATH,
        help="Local Qwen3.5 model snapshot path",
    )
    parser.add_argument(
        "--qwen-cache",
        type=str,
        default=None,
        help="JSONL cache path for resumable Qwen results",
    )
    parser.add_argument(
        "--offload-folder",
        type=str,
        default=None,
        help="Optional offload folder for model loading",
    )
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--qwen-batch-size", type=int, default=1)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=96)
    parser.add_argument("--max-scenes", type=int, default=None, help="Debug limit after loading index")
    parser.add_argument(
        "--prefilter-only",
        action="store_true",
        help="Stop after metadata prefilter and save Qwen candidates without running the model",
    )

    parser.add_argument("--min-mask-ratio", type=float, default=0.005)
    parser.add_argument("--min-num-parts", type=int, default=6)

    parser.add_argument("--prefilter-min-parts", type=int, default=12)
    parser.add_argument("--prefilter-max-largest-mask-ratio", type=float, default=0.35)
    parser.add_argument("--prefilter-max-total-mask-ratio", type=float, default=0.70)
    parser.add_argument("--prefilter-min-total-mask-ratio", type=float, default=0.08)

    parser.add_argument("--min-indoor-room-score", type=int, default=4)
    parser.add_argument("--min-wide-view-score", type=int, default=3)
    parser.add_argument("--min-object-density-score", type=int, default=3)
    parser.add_argument("--min-qwen-confidence", type=int, default=2)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.output is None:
        args.output = default_output_path(args.index)

    if args.qwen_cache is None:
        args.qwen_cache = args.output.replace(".json", "_qwen_cache.jsonl")

    output_parent = os.path.dirname(args.output)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    cache_parent = os.path.dirname(args.qwen_cache)
    if cache_parent:
        os.makedirs(cache_parent, exist_ok=True)

    print(f"Loading index: {args.index}")
    with open(args.index) as f:
        scenes = json.load(f)
    if isinstance(scenes, dict):
        raise TypeError(f"Expected a list of scenes, got {type(scenes).__name__}")
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    print(f"  Total scenes loaded: {len(scenes)}")

    stats = {
        "total_scenes_before": len(scenes),
        "total_objects_before": int(sum(scene["num_parts"] for scene in scenes)),
        "objects_removed_small_mask": 0,
        "scenes_removed_few_parts": 0,
        "scenes_removed_prefilter_few_objects": 0,
        "scenes_removed_prefilter_closeup": 0,
        "scenes_removed_prefilter_overfilled": 0,
        "scenes_removed_prefilter_too_sparse": 0,
        "scenes_sent_to_qwen": 0,
        "scenes_loaded_from_qwen_cache": 0,
        "scenes_removed_qwen": 0,
        "qwen_parse_failures": 0,
    }

    print("\n=== Stage 1: Mask Filtering + Metadata Preparation ===")
    prepared_scenes = []
    for scene in tqdm(scenes, desc="Preparing scenes"):
        prepared, delta = prepare_scene(scene, args)
        for key, value in delta.items():
            stats[key] += value
        if prepared is not None:
            prepared_scenes.append(prepared)

    print(f"  Survivors after mask filtering: {len(prepared_scenes)}")

    print("\n=== Stage 2: Metadata Prefilter ===")
    qwen_candidates = []
    filtered_prefilter = []
    for scene in prepared_scenes:
        keep, reason = passes_prefilter(scene, args)
        if keep:
            qwen_candidates.append(scene)
        else:
            filtered_prefilter.append((scene, reason))
            stats[f"scenes_removed_{reason}"] += 1

    print(f"  Sent to Qwen candidates: {len(qwen_candidates)}")
    print(f"  Removed by prefilter: {len(filtered_prefilter)}")

    qwen_cache = load_qwen_cache(args.qwen_cache)
    evaluator = None
    final_scenes = []

    if args.prefilter_only:
        print("\n=== Stage 3: Prefilter-Only Mode ===")
        output_scenes = []
        for scene in qwen_candidates:
            output_scenes.append(
                {
                    "image_id": scene["image_id"],
                    "image_path": scene["image_path"],
                    "pointmap_path": scene.get("pointmap_path"),
                    "mesh_paths": scene["mesh_paths"],
                    "pose_paths": scene["pose_paths"],
                    "mask_paths": scene["mask_paths"],
                    "ann_ids": scene["ann_ids"],
                    "num_parts": scene["num_parts"],
                    "mask_area_ratios": scene["mask_area_ratios"],
                    "largest_mask_ratio": round(scene["largest_mask_ratio"], 6),
                    "total_mask_ratio": round(scene["total_mask_ratio"], 6),
                    "avg_mask_ratio": round(scene["avg_mask_ratio"], 6),
                }
            )

        stats["total_scenes_after"] = len(output_scenes)
        stats["total_objects_after"] = int(sum(scene["num_parts"] for scene in output_scenes))
        print(f"  Prefilter-only candidates: {len(output_scenes)}")
        print(f"\nSaving filtered index to: {args.output}")
        with open(args.output, "w") as f:
            json.dump(output_scenes, f, indent=2)

        stats_path = args.output.replace(".json", "_stats.json")
        stats["config"] = {
            "index": args.index,
            "prefilter_only": True,
            "min_mask_ratio": args.min_mask_ratio,
            "min_num_parts": args.min_num_parts,
            "prefilter_min_parts": args.prefilter_min_parts,
            "prefilter_max_largest_mask_ratio": args.prefilter_max_largest_mask_ratio,
            "prefilter_max_total_mask_ratio": args.prefilter_max_total_mask_ratio,
            "prefilter_min_total_mask_ratio": args.prefilter_min_total_mask_ratio,
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Stats saved to: {stats_path}")
        return

    print("\n=== Stage 3: Qwen Scene Evaluation ===")
    uncached_candidates = []
    for scene in qwen_candidates:
        cached = qwen_cache.get(int(scene["image_id"]))
        if cached is not None:
            stats["scenes_loaded_from_qwen_cache"] += 1
            scene["qwen_eval"] = cached
            keep = final_keep_from_qwen(cached, args)
            if cached.get("parse_ok") is False:
                stats["qwen_parse_failures"] += 1
            if keep:
                final_scenes.append(scene)
            else:
                stats["scenes_removed_qwen"] += 1
        else:
            uncached_candidates.append(scene)

    if uncached_candidates:
        evaluator = QwenSceneEvaluator(
            model_path=args.qwen_model_path,
            dtype_name=args.dtype,
            max_new_tokens=args.qwen_max_new_tokens,
            offload_folder=args.offload_folder,
        )
        print(f"  Loading Qwen model from: {args.qwen_model_path}")
        evaluator.load()

        for start in tqdm(range(0, len(uncached_candidates), args.qwen_batch_size), desc="Qwen batches"):
            batch = uncached_candidates[start : start + args.qwen_batch_size]
            results = evaluator.evaluate_batch(batch)
            stats["scenes_sent_to_qwen"] += len(batch)

            for scene, result in zip(batch, results):
                scene["qwen_eval"] = result
                if not result["parse_ok"]:
                    stats["qwen_parse_failures"] += 1
                append_qwen_cache(args.qwen_cache, scene["image_id"], result)
                keep = final_keep_from_qwen(result, args)
                if keep:
                    final_scenes.append(scene)
                else:
                    stats["scenes_removed_qwen"] += 1

        del evaluator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Stage 4: Build Output ===")
    output_scenes = []
    for scene in final_scenes:
        qwen_eval = scene["qwen_eval"]
        output_scenes.append(
            {
                "image_id": scene["image_id"],
                "image_path": scene["image_path"],
                "pointmap_path": scene.get("pointmap_path"),
                "mesh_paths": scene["mesh_paths"],
                "pose_paths": scene["pose_paths"],
                "mask_paths": scene["mask_paths"],
                "ann_ids": scene["ann_ids"],
                "num_parts": scene["num_parts"],
                "mask_area_ratios": scene["mask_area_ratios"],
                "largest_mask_ratio": round(scene["largest_mask_ratio"], 6),
                "total_mask_ratio": round(scene["total_mask_ratio"], 6),
                "avg_mask_ratio": round(scene["avg_mask_ratio"], 6),
                "qwen_indoor_room_score": qwen_eval["indoor_room_score"],
                "qwen_wide_view_score": qwen_eval["wide_view_score"],
                "qwen_object_density_score": qwen_eval["object_density_score"],
                "qwen_confidence": qwen_eval["confidence"],
                "qwen_model_keep": qwen_eval["model_keep"],
                "qwen_reason": qwen_eval["reason"],
                "qwen_parse_ok": qwen_eval["parse_ok"],
            }
        )

    stats["total_scenes_after"] = len(output_scenes)
    stats["total_objects_after"] = int(sum(scene["num_parts"] for scene in output_scenes))

    print("\n=== Results ===")
    print(f"  Scenes:  {stats['total_scenes_before']} -> {stats['total_scenes_after']}")
    print(f"  Objects: {stats['total_objects_before']} -> {stats['total_objects_after']}")
    print(f"  Removed by few parts: {stats['scenes_removed_few_parts']}")
    print(f"  Removed by prefilter close-up: {stats['scenes_removed_prefilter_closeup']}")
    print(f"  Removed by prefilter few objects: {stats['scenes_removed_prefilter_few_objects']}")
    print(f"  Removed by prefilter overfilled: {stats['scenes_removed_prefilter_overfilled']}")
    print(f"  Removed by prefilter too sparse: {stats['scenes_removed_prefilter_too_sparse']}")
    print(f"  Removed by Qwen: {stats['scenes_removed_qwen']}")
    print(f"  Qwen parse failures: {stats['qwen_parse_failures']}")

    print(f"\nSaving filtered index to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(output_scenes, f, indent=2)

    stats_path = args.output.replace(".json", "_stats.json")
    stats["config"] = {
        "index": args.index,
        "min_mask_ratio": args.min_mask_ratio,
        "min_num_parts": args.min_num_parts,
        "prefilter_min_parts": args.prefilter_min_parts,
        "prefilter_max_largest_mask_ratio": args.prefilter_max_largest_mask_ratio,
        "prefilter_max_total_mask_ratio": args.prefilter_max_total_mask_ratio,
        "prefilter_min_total_mask_ratio": args.prefilter_min_total_mask_ratio,
        "min_indoor_room_score": args.min_indoor_room_score,
        "min_wide_view_score": args.min_wide_view_score,
        "min_object_density_score": args.min_object_density_score,
        "min_qwen_confidence": args.min_qwen_confidence,
        "qwen_model_path": args.qwen_model_path,
        "qwen_cache": args.qwen_cache,
        "dtype": args.dtype,
        "qwen_batch_size": args.qwen_batch_size,
        "qwen_max_new_tokens": args.qwen_max_new_tokens,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Filtered file size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")
    print(f"  Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
