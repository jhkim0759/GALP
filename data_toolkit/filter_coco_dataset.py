"""
COCO Dataset Filtering Script
==============================
1. Mask area 기반 소형 object 필터링
2. CLIP 기반 indoor room 분류
3. 필터링된 index.json 저장

Usage:
    conda run -n trellis python data_toolkit/filter_coco_dataset.py \
        --index /mnt/storage/jhkim/coco/train2017/index.json \
        --min-mask-ratio 0.005 \
        --min-num-parts 3 \
        --indoor-threshold 0.5 \
        --batch-size 64 \
        --device cuda:0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ─── CLIP Indoor Room Classifier ────────────────────────────────────────

def build_clip_classifier(device="cuda:0"):
    """CLIP ViT-B-32 기반 indoor room 분류기를 초기화합니다."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai",
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    # Indoor room 관련 positive/negative prompts
    positive_prompts = [
        "a photo of an indoor room",
        "a photo of a living room with furniture",
        "a photo of a bedroom interior",
        "a photo of an office room",
        "a photo of a dining room",
        "a photo of an indoor space with walls and floor",
    ]
    negative_prompts = [
        "a photo of an outdoor scene",
        "a photo of a street or city",
        "a photo of nature or landscape",
        "a close-up photo of food or objects",
        "a photo of animals",
        "a photo of people doing sports",
    ]

    with torch.no_grad():
        pos_tokens = tokenizer(positive_prompts).to(device)
        neg_tokens = tokenizer(negative_prompts).to(device)
        pos_feats = model.encode_text(pos_tokens)
        neg_feats = model.encode_text(neg_tokens)
        pos_feats /= pos_feats.norm(dim=-1, keepdim=True)
        neg_feats /= neg_feats.norm(dim=-1, keepdim=True)

        # Average positive/negative embeddings
        pos_mean = pos_feats.mean(dim=0, keepdim=True)
        neg_mean = neg_feats.mean(dim=0, keepdim=True)
        pos_mean /= pos_mean.norm(dim=-1, keepdim=True)
        neg_mean /= neg_mean.norm(dim=-1, keepdim=True)

    return model, preprocess, pos_mean, neg_mean


@torch.no_grad()
def classify_indoor_batch(model, preprocess, pos_mean, neg_mean,
                          image_paths, device="cuda:0", batch_size=64):
    """
    이미지 배치를 indoor room 여부로 분류합니다.

    Returns:
        scores: list of float — indoor score (0~1), 높을수록 indoor room
        labels: list of bool — indoor room 여부
    """
    scores = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        valid_mask = []
        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                imgs.append(img)
                valid_mask.append(True)
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
                valid_mask.append(False)

        batch_tensor = torch.stack(imgs).to(device)
        img_feats = model.encode_image(batch_tensor)
        img_feats /= img_feats.norm(dim=-1, keepdim=True)

        # Cosine similarity with indoor/outdoor centroids
        sim_pos = (img_feats @ pos_mean.T).squeeze(-1)  # (B,)
        sim_neg = (img_feats @ neg_mean.T).squeeze(-1)  # (B,)

        # Softmax → indoor probability
        logits = torch.stack([sim_neg, sim_pos], dim=-1) * 100.0  # temperature
        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(indoor)

        for j, valid in enumerate(valid_mask):
            if valid:
                scores.append(probs[j].item())
            else:
                scores.append(0.0)

    return scores


# ─── Mask Area Filtering ────────────────────────────────────────────────

def compute_mask_areas(mask_paths):
    """각 mask의 면적 비율을 계산합니다. Returns list of float (0~1)."""
    ratios = []
    for mp in mask_paths:
        try:
            img = np.array(Image.open(mp))
            total = img.shape[0] * img.shape[1]
            area = (img > 127).sum()
            ratios.append(area / total if total > 0 else 0.0)
        except Exception:
            ratios.append(0.0)
    return ratios


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="COCO Dataset Filtering")
    parser.add_argument("--index", type=str,
                        default="/mnt/storage/jhkim/coco/train2017/index.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: same dir as index, index_filtered.json)")
    parser.add_argument("--min-mask-ratio", type=float, default=0.005,
                        help="Minimum mask area ratio (0~1). Default: 0.005 (0.5%%)")
    parser.add_argument("--min-num-parts", type=int, default=3,
                        help="Minimum objects per scene after mask filtering")
    parser.add_argument("--indoor-threshold", type=float, default=0.5,
                        help="CLIP indoor score threshold (0~1). Default: 0.5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip-indoor", action="store_true",
                        help="Skip indoor classification (mask filtering only)")
    args = parser.parse_args()

    # Output path
    if args.output is None:
        index_dir = os.path.dirname(args.index)
        args.output = os.path.join(index_dir, "index_filtered.json")

    # ── 1. Load index ──
    print(f"Loading index: {args.index}")
    with open(args.index) as f:
        scenes = json.load(f)
    print(f"  Total scenes: {len(scenes)}")

    # ── 2. CLIP indoor classification ──
    if not args.skip_indoor:
        print("\n=== CLIP Indoor Room Classification ===")
        model, preprocess, pos_mean, neg_mean = build_clip_classifier(args.device)

        image_paths = []
        for scene in scenes:
            image_paths.append(scene["image_path"])

        print(f"  Classifying {len(image_paths)} images...")
        indoor_scores = classify_indoor_batch(
            model, preprocess, pos_mean, neg_mean,
            image_paths, device=args.device, batch_size=args.batch_size,
        )

        # Attach scores
        for scene, score in zip(scenes, indoor_scores):
            scene["indoor_score"] = round(score, 4)
            scene["is_indoor"] = score >= args.indoor_threshold

        n_indoor = sum(1 for s in scenes if s["is_indoor"])
        print(f"  Indoor: {n_indoor}/{len(scenes)} ({n_indoor/len(scenes)*100:.1f}%)")
        print(f"  Outdoor/Other: {len(scenes)-n_indoor}/{len(scenes)}")

        # Score distribution
        scores_arr = np.array(indoor_scores)
        print(f"  Score stats: min={scores_arr.min():.3f}, "
              f"median={np.median(scores_arr):.3f}, "
              f"mean={scores_arr.mean():.3f}, max={scores_arr.max():.3f}")

        # Free GPU
        del model, preprocess, pos_mean, neg_mean
        torch.cuda.empty_cache()
    else:
        print("Indoor classification skipped")
        for scene in scenes:
            scene["indoor_score"] = -1.0
            scene["is_indoor"] = True  # keep all

    # ── 3. Mask area filtering ──
    print(f"\n=== Mask Area Filtering (min ratio: {args.min_mask_ratio}) ===")

    stats = {
        "total_scenes_before": len(scenes),
        "total_objects_before": sum(s["num_parts"] for s in scenes),
        "objects_removed_small_mask": 0,
        "scenes_removed_few_parts": 0,
        "scenes_removed_outdoor": 0,
    }

    filtered_scenes = []
    for scene in tqdm(scenes, desc="Filtering masks"):
        # Skip outdoor scenes
        if not scene.get("is_indoor", True):
            stats["scenes_removed_outdoor"] += 1
            continue

        # Filter small masks
        mask_paths = scene["mask_paths"]
        areas = compute_mask_areas(mask_paths)

        keep_indices = [i for i, a in enumerate(areas) if a >= args.min_mask_ratio]
        n_removed = len(mask_paths) - len(keep_indices)
        stats["objects_removed_small_mask"] += n_removed

        if len(keep_indices) < args.min_num_parts:
            stats["scenes_removed_few_parts"] += 1
            continue

        # Build filtered scene entry
        filtered_scene = {
            "image_id": scene["image_id"],
            "image_path": scene["image_path"],
            "pointmap_path": scene.get("pointmap_path"),
            "mesh_paths": [scene["mesh_paths"][i] for i in keep_indices],
            "pose_paths": [scene["pose_paths"][i] for i in keep_indices],
            "mask_paths": [scene["mask_paths"][i] for i in keep_indices],
            "ann_ids": [scene["ann_ids"][i] for i in keep_indices],
            "num_parts": len(keep_indices),
            "indoor_score": scene.get("indoor_score", -1.0),
            "is_indoor": scene.get("is_indoor", True),
            "mask_area_ratios": [round(areas[i], 6) for i in keep_indices],
        }
        filtered_scenes.append(filtered_scene)

    stats["total_scenes_after"] = len(filtered_scenes)
    stats["total_objects_after"] = sum(s["num_parts"] for s in filtered_scenes)

    # ── 4. Save ──
    print(f"\n=== Results ===")
    print(f"  Scenes:  {stats['total_scenes_before']} → {stats['total_scenes_after']} "
          f"(-{stats['total_scenes_before']-stats['total_scenes_after']})")
    print(f"  Objects: {stats['total_objects_before']} → {stats['total_objects_after']} "
          f"(-{stats['total_objects_before']-stats['total_objects_after']})")
    print(f"  Removed — outdoor/non-room: {stats['scenes_removed_outdoor']}")
    print(f"  Removed — too few parts after mask filter: {stats['scenes_removed_few_parts']}")
    print(f"  Objects removed (small mask): {stats['objects_removed_small_mask']}")

    print(f"\nSaving to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(filtered_scenes, f, indent=2)
    print(f"  File size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")

    # Also save stats
    stats_path = args.output.replace(".json", "_stats.json")
    stats["config"] = {
        "min_mask_ratio": args.min_mask_ratio,
        "min_num_parts": args.min_num_parts,
        "indoor_threshold": args.indoor_threshold,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
