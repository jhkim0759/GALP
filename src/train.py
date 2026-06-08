import sys
from pathlib import Path

FILE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path.cwd()
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings
import json
warnings.filterwarnings("ignore")  # ignore all warnings
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()  # ignore diffusers warnings

from src.utils.typing_utils import *

import os
import argparse
import logging
import time
import math
from functools import partial
import gc
import importlib.util
from packaging import version
import random
import trimesh
from PIL import Image
import numpy as np
try:
    import wandb
except ImportError:  # make smoke-test path runnable without wandb installed
    wandb = None
from tqdm import tqdm

import torch
import torch.nn.functional as tF
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerate_logger
from accelerate import DataLoaderConfiguration, DeepSpeedPlugin
from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin, InitProcessGroupKwargs
from datetime import timedelta

from transformers import (
    BitImageProcessor,
    Dinov2Model,
)

from hydra.utils import instantiate
from omegaconf import OmegaConf

from pytorch3d.transforms import rotation_6d_to_matrix
from src.utils.train_utils import transform_points
os.environ.setdefault("LIDRA_SKIP_INIT", "1")
# Prefer flash attention; fall back to torch's flash SDPA when the package is missing.
ATTN_BACKEND_CHOICE = os.environ.get("ATTN_BACKEND")
ATTN_BACKEND_REASON = "preset via environment"
if ATTN_BACKEND_CHOICE is None:
    try:
        if importlib.util.find_spec("flash_attn") is not None:
            os.environ["ATTN_BACKEND"] = "flash_attn"
            os.environ['SPARSE_ATTN_BACKEND'] = "flash_attn"
            ATTN_BACKEND_REASON = "flash_attn package detected"
        elif importlib.util.find_spec("torch.nn.attention") is not None:
            os.environ["ATTN_BACKEND"] = "torch_flash_attn"
            os.environ['SPARSE_ATTN_BACKEND'] = "torch_flash_attn"
            ATTN_BACKEND_REASON = "torch flash attention available"
        else:
            os.environ["ATTN_BACKEND"] = "sdpa"
            os.environ['SPARSE_ATTN_BACKEND'] = "sdpa"
            ATTN_BACKEND_REASON = "flash attention not installed; using SDPA"
    except Exception as exc:  # noqa: BLE001
        os.environ["ATTN_BACKEND"] = "sdpa"
        os.environ['SPARSE_ATTN_BACKEND'] = "sdpa"
        ATTN_BACKEND_REASON = f"backend detection failed ({exc}); using SDPA"
ATTN_BACKEND_CHOICE = os.environ["ATTN_BACKEND"]

from src.sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow import (  # noqa: E402
    SparseStructureFlowTdfyWrapper,
)
from src.sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae import (  # noqa: E402
    SparseStructureEncoderTdfyWrapper,
    SparseStructureDecoderTdfyWrapper,
)
from src.sam3d_objects.model.io import (  # noqa: E402
    filter_and_remove_prefix_state_dict_fn,
    load_model_from_checkpoint,
)


from src.utils.train_utils import (
    MyEMAModel, 
    get_configs,
    get_optimizer,
    get_lr_scheduler,
    save_experiment_params,
    save_model_architecture,
)

from safetensors.torch import load_file


class ConditionEmbedding:
    """
    Lightweight helper to run a condition embedder with a configurable input mapping.
    """

    def __init__(self, condition_embedder=None, condition_input_mapping=None):
        self.condition_embedder = condition_embedder
        self.condition_input_mapping = list(condition_input_mapping or [])

    def map_input_keys(self, inputs):
        return [inputs[k] for k in self.condition_input_mapping]

    def __call__(self, inputs):
        condition_args = self.map_input_keys(inputs)
        condition_kwargs = {
            k: v for k, v in inputs.items() if k not in self.condition_input_mapping
        }
        if self.condition_embedder is None:
            return condition_args, condition_kwargs

        tokens = self.condition_embedder(*condition_args, **condition_kwargs)
        return (tokens,), {}


class ConditionEmbeddingSplit:
    """
    V2 variant of ConditionEmbedding.

    Calls EmbedderFuser with ``return_dict=True`` so each modality's tokens
    are kept separate, then groups them into:
      - ``cond_img`` : image + mask tokens  (DINOv2 outputs)
      - ``cond_pm``  : pointmap tokens      (PointPatchEmbed outputs)

    The split is driven by ``pointmap_keys`` — any kwarg_name that appears in
    this set is routed to ``cond_pm``; everything else goes to ``cond_img``.
    """

    # Default EmbedderFuser kwarg names that carry pointmap information
    DEFAULT_POINTMAP_KEYS = frozenset({"pointmap", "rgb_pointmap"})

    def __init__(
        self,
        condition_embedder=None,
        condition_input_mapping=None,
        pointmap_keys=None,
    ):
        self.condition_embedder = condition_embedder
        self.condition_input_mapping = list(condition_input_mapping or [])
        self.pointmap_keys = (
            frozenset(pointmap_keys)
            if pointmap_keys is not None
            else self.DEFAULT_POINTMAP_KEYS
        )

    def __call__(self, inputs):
        if self.condition_embedder is None:
            return None, None

        # Pass all batch keys as kwargs (same logic as ConditionEmbedding)
        condition_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in self.condition_input_mapping
        }

        # Ask EmbedderFuser to return per-key token tensors
        token_dict = self.condition_embedder(
            **condition_kwargs, return_dict=True
        )

        img_tokens = []
        pm_tokens = []
        for key, tok in token_dict.items():
            if key in self.pointmap_keys:
                pm_tokens.append(tok)
            else:
                img_tokens.append(tok)

        cond_img = torch.cat(img_tokens, dim=1) if img_tokens else None
        cond_pm = torch.cat(pm_tokens, dim=1) if pm_tokens else None
        return cond_img, cond_pm


def l2_abs_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Absolute-difference loss averaged over all non-batch dimensions.
    """
    diff = torch.abs(pred - target)
    reduce_dims = list(range(1, diff.dim()))
    return diff.mean(dim=reduce_dims)


def reconcile_ss_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Map modality-specific attention weights from older checkpoints to the shared
    attention parameters expected by the current codebase.
    """
    remapped: Dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        # Drop keys for modalities that don't exist in the current architecture
        if "latent_mapping.translation_scale" in key:
            continue

        mapped = False
        for attn_part in ("to_qkv", "to_out", "q_rms_norm", "k_rms_norm"):
            needle = f".self_attn.{attn_part}."
            if needle not in key:
                continue

            prefix, rest = key.split(needle, 1)
            if "." not in rest:
                break
            _, param = rest.split(".", 1)
            target_key = f"{prefix}{needle}{param}"
            if target_key not in remapped:
                remapped[target_key] = val
            mapped = True
            break

        if not mapped:
            remapped[key] = val
    return remapped


def ensure_l1_loss_fn(model: torch.nn.Module) -> None:
    """
    Force a model that exposes `loss_fn` to use mean L1 loss.
    """
    if hasattr(model, "loss_fn"):
        model.loss_fn = partial(torch.nn.functional.l1_loss, reduction="mean")


def pad_or_truncate(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    Pad or truncate the last dimension of `tensor` to `target_dim`.
    """
    if tensor.shape[-1] == target_dim:
        return tensor
    if tensor.shape[-1] > target_dim:
        return tensor[..., :target_dim]
    pad_size = target_dim - tensor.shape[-1]
    pad_shape = list(tensor.shape[:-1]) + [pad_size]
    pad_tensor = torch.zeros(
        *pad_shape, device=tensor.device, dtype=tensor.dtype
    )
    return torch.cat([tensor, pad_tensor], dim=-1)



def build_shape_latents(
    voxels: torch.Tensor, token_len: int, in_channels: int
) -> torch.Tensor:
    """
    Convert a voxel grid or precomputed latent tokens into the latent tensor shape
    expected by the flow backbone.
    """
    # If the encoder already returned tokenized latents, only pad/truncate
    # the sequence and channel dimensions.
    if voxels.ndim == 3:
        tokens = pad_or_truncate(voxels, in_channels)
        cur_tokens = tokens.shape[1]
        if cur_tokens < token_len:
            pad_tokens = token_len - cur_tokens
            pad = torch.zeros(
                tokens.shape[0],
                pad_tokens,
                in_channels,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            return torch.cat([tokens, pad], dim=1)
        if cur_tokens > token_len:
            return tokens[:, :token_len]
        return tokens

    if voxels.ndim == 4:
        voxels = voxels.unsqueeze(1)
    target_res = round(token_len ** (1 / 3))
    grid = voxels
    if grid.shape[-1] != target_res:
        grid = tF.interpolate(
            grid,
            size=(target_res, target_res, target_res),
            mode="trilinear",
            align_corners=False,
        )
    if grid.shape[1] < in_channels:
        repeat = math.ceil(in_channels / grid.shape[1])
        grid = grid.repeat(1, repeat, 1, 1, 1)[:, :in_channels]
    elif grid.shape[1] > in_channels:
        grid = grid[:, :in_channels]
    flat = grid.view(grid.shape[0], grid.shape[1], -1).permute(0, 2, 1).contiguous()
    if flat.shape[1] < token_len:
        pad_tokens = token_len - flat.shape[1]
        pad = torch.zeros(
            flat.shape[0], pad_tokens, in_channels, device=flat.device, dtype=flat.dtype
        )
        flat = torch.cat([flat, pad], dim=1)
    elif flat.shape[1] > token_len:
        flat = flat[:, :token_len]
    return flat


def prepare_generator_latents(
    flow_backbone: SparseStructureFlowTdfyWrapper,
    voxels: torch.Tensor,
    translation: torch.Tensor,
    rot6d: torch.Tensor,
    scales: torch.Tensor,
    shape_latent: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Build a latents dict that matches the flow backbone's latent_mapping from raw batch tensors.
    If an encoded `shape_latent` is provided (e.g., output of `ss_encoder`), it is used
    for the shape tokens; otherwise the raw voxels are projected.
    """
    latents: Dict[str, torch.Tensor] = {}
    for name, latent_mod in flow_backbone.latent_mapping.items():
        token_len = latent_mod.pos_emb.shape[0]
        in_ch = latent_mod.input_layer.in_features

        if name == "shape":
            source = shape_latent if shape_latent is not None else voxels
            latents[name] = build_shape_latents(source, token_len, in_ch)
        elif name == "translation":
            latents[name] = pad_or_truncate(translation, in_ch).unsqueeze(1)
        elif name == "6drotation_normalized":
            latents[name] = pad_or_truncate(rot6d, in_ch).unsqueeze(1)
        elif name == "scale":
            scale_vals = scales
            if scale_vals.shape[-1] == 1 and in_ch > 1:
                scale_vals = scale_vals.expand(-1, in_ch)
            latents[name] = pad_or_truncate(scale_vals, in_ch).unsqueeze(1)
        else:
            latents[name] = torch.zeros(
                voxels.shape[0],
                token_len,
                in_ch,
                device=voxels.device,
                dtype=voxels.dtype,
            )
    return latents


def instantiate_and_load_from_pretrained(
    config,
    ckpt_path,
    state_dict_fn=None,
    state_dict_key="state_dict",
    device="cuda",
    strict=False,
):
    model = instantiate(config)

    if ckpt_path is None:
        return model.to(device)

    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path, device="cuda")
        if state_dict_fn is not None:
            state_dict = state_dict_fn(state_dict)
        state_dict = reconcile_ss_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()
    else:
        model = load_model_from_checkpoint(
            model,
            ckpt_path,
            strict=strict,
            device="cpu",
            freeze=True,
            eval=True,
            state_dict_key=state_dict_key,
            state_dict_fn=state_dict_fn,
        )
    return model.to(device)


def strip_module_prefix(sd: dict) -> dict:
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }



def init_ss_generator_scratch(
    ss_generator_config_path, ckpt_path, workspace_dir="", device="cuda", pretrained=False, resolution=32
):
    logger = logging.getLogger(__name__)
    cfg = OmegaConf.load(ss_generator_config_path)
    flow_cfg = cfg["module"]["generator"]["backbone"]["reverse_fn"]["backbone"]
    flow_cfg['latent_mapping']['shape']['pos_embedder']['resolution'] = resolution

    model: SparseStructureFlowTdfyWrapper = instantiate(flow_cfg)
    missing, unexpected = None, None
    return model.to(device), missing, unexpected

def init_ss_generator(
    ss_generator_config_path, ckpt_path, workspace_dir="", device="cuda", pretrained=False, resolution=32
):
    logger = logging.getLogger(__name__)
    cfg = OmegaConf.load(ss_generator_config_path)
    flow_cfg = cfg["module"]["generator"]["backbone"]["reverse_fn"]["backbone"]
    flow_cfg['latent_mapping']['shape']['pos_embedder']['resolution'] = resolution

    model: SparseStructureFlowTdfyWrapper = instantiate(flow_cfg)
    missing, unexpected = None, None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"]
    state_dict = filter_and_remove_prefix_state_dict_fn(
        "_base_models.generator.reverse_fn.backbone."
    )(state_dict)
    state_dict.pop("latent_mapping.shape.pos_emb", None)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        logger.warning("Missing keys while loading flow: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys while loading flow: %s", unexpected)
    else:
        logger.info("Loaded flow weights from %s", ckpt_path)

    return model.to(device), missing, unexpected


def init_ss_generator_v1_4(
    ss_generator_config_path,
    device="cuda",
    resolution=32,
):
    from src.sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow_v1_4 import (
        SparseStructureFlowTdfyWrapperV1_4,
    )
    logger = logging.getLogger(__name__)
    cfg = OmegaConf.load(ss_generator_config_path)
    flow_cfg = cfg["module"]["generator"]["backbone"]["reverse_fn"]["backbone"]
    flow_cfg['latent_mapping']['shape']['pos_embedder']['resolution'] = resolution

    base_model: SparseStructureFlowTdfyWrapper = instantiate(flow_cfg)
    v1_4_model = SparseStructureFlowTdfyWrapperV1_4(base_model=base_model).to(device)

    logger.info("Loaded V1_4 (V3 base + floor rotation token)")
    return v1_4_model, None, None

def init_ss_condition_embedder(
    ss_generator_config_path, ss_generator_ckpt_path, workspace_dir="", device="cuda"
):
    conf = OmegaConf.load(os.path.join(workspace_dir, ss_generator_config_path))
    if "condition_embedder" in conf["module"]:
        return instantiate_and_load_from_pretrained(
            conf["module"]["condition_embedder"]["backbone"],
            os.path.join(workspace_dir, ss_generator_ckpt_path),
            state_dict_fn=filter_and_remove_prefix_state_dict_fn(
                "_base_models.condition_embedder."
            ),
            device=device,
        )
    return None

def load_trellis_ss_wrapper(
    base_path: Union[str, Path],
    kind: Literal["encoder", "decoder"],
    device: torch.device,
):
    """
    Load TRELLIS sparse-structure VAE from a local JSON/safetensors pair using our wrappers.
    """
    base = Path(base_path)
    json_path = base.with_suffix(".json")
    ckpt_path = base.with_suffix(".safetensors")
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    with open(json_path) as f:
        cfg = json.load(f)
    args = cfg.get("args", {})

    if kind == "encoder":
        model = SparseStructureEncoderTdfyWrapper(
            **args,
            sample_posterior=False,
            return_raw=False,
            pretrained_ckpt_path=str(ckpt_path),
        )
    else:
        model = SparseStructureDecoderTdfyWrapper(
            **args,
            pretrained_ckpt_path=str(ckpt_path),
        )

    model = model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model




def main():
    PROJECT_NAME = "MeshLayout"

    parser = argparse.ArgumentParser(
        description="Train a diffusion model for 3D object generation",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--resume_from_iter",
        type=int,
        default=None,
        help="The iteration to load the checkpoint from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--offline_wandb",
        action="store_true",
        help="Use offline WandB for experiment tracking"
    )
    parser.add_argument(
        "--single_iter_test",
        action="store_true",
        help="Run a single synthetic forward/backward step to verify the model wiring."
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=2,
        help="Batch size used for --single_iter_test."
    )
    parser.add_argument(
        "--test_cond_tokens",
        type=int,
        default=4,
        help="Number of condition tokens for the smoke test."
    )
    parser.add_argument(
        "--test_lr",
        type=float,
        default=1e-4,
        help="Learning rate for the smoke test optimizer."
    )
    parser.add_argument(
        "--flow_cfg",
        type=str,
        default=None,
        help="Path to the structure generator config for the smoke test."
    )
    parser.add_argument(
        "--flow_ckpt",
        type=str,
        default=None,
        help="Path to the structure generator checkpoint for the smoke test."
    )
    parser.add_argument(
        "--trim_flow_blocks",
        type=int,
        default=None,
        help="Trim the flow backbone to the first N blocks during the smoke test."
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="The max iteration step for training"
    )
    parser.add_argument(
        "--max_val_steps",
        type=int,
        default=2,
        help="The max iteration step for validation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for the data loader"
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA model for training"
    )
    parser.add_argument(
        "--ema_device",
        type=str,
        default="same",
        choices=["same", "cpu"],
        help="Device for EMA shadow weights. Use 'cpu' to avoid duplicating EMA on each GPU rank."
    )
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        help="Enable DDP unused-parameter detection for batch-dependent branches.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale lr with total batch size (base batch size: 256)"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.,
        help="Max gradient norm for gradient clipping"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Type of mixed precision training"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs"
    )

    parser.add_argument(
        "--val_guidance_scales",
        type=list,
        nargs="+",
        default=[7.0],
        help="CFG scale used for validation"
    )

    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for training"
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=1,
        choices=[1, 2, 3],  # https://huggingface.co/docs/accelerate/usage_guides/deepspeed
        help="ZeRO stage type for DeepSpeed"
    )

    parser.add_argument(
        "--load_ckpt",
        type=str,  
        default=None,
        help="Path to a checkpoint to resume training from (overrides --resume_from_iter)"
    )

    parser.add_argument(
        "--use_latent",
        action="store_true",
        default=False,
        help="Use Shape Latent"
    )

    parser.add_argument(
        "--dataset_mix",
        type=str,
        default="future",
        help="Training dataset mix (e.g. future, future+scannet, future+scannet+coco)"
    )

    parser.add_argument(
        "--use_logscale",
        action="store_true",
        default=False,
        help="Debug mode with more logging"
    )

    parser.add_argument(
        "--scratch",
        action="store_true",
        default=False,
        help="Debug mode with more logging"
    )

    # ---------- V2 model & new losses ----------
    parser.add_argument(
        "--init_weight",
        type=str,
        default=None,
        help="[v1_5] Optional v1_4 ckpt used to initialise v1_5 weights via key remapping (default: floor_rot 010000)"
    )
    parser.add_argument(
        "--pm_ctx_channels",
        type=int,
        default=1024,
        help="[V2] pointmap token dim; must match EmbedderFuser.embed_dims"
    )
    parser.add_argument(
        "--use_pm_surface_loss",
        action="store_true",
        default=False,
        help="Pointmap visible-surface alignment loss: align the instance-mask-region pointmap with the predicted mesh pose"
    )
    parser.add_argument(
        "--pm_surface_loss_weight",
        type=float,
        default=1.0,
        help="Weight for pointmap surface alignment loss"
    )
    parser.add_argument(
        "--load_v2_ckpt",
        type=str,
        default=None,
        help="[V2] Path to a V1 checkpoint to load into the V2 model (for weight reuse and faster convergence)"
    )

    parser.add_argument(
        "--no_pointmap",
        action="store_true",
        default=False,
        help=(
            "Disable pointmap as a model condition. "
            "The model will use only image/mask tokens (V1 without pointmap). "
            "Pointmap files are still loaded for scene coordinate normalization "
            "unless normalize_scene is also disabled."
        ),
    )

    # ---------- Pointmap quality ----------
    parser.add_argument(
        "--use_high_pointmap",
        action="store_true",
        default=False,
        help="Replace the ScanNet pointmap with DA3-LARGE-based pointmaps_high (default: DA3-SMALL pointmaps)",
    )

    # ---------- Pointmap / Mesh Augmentation ----------
    parser.add_argument(
        "--use_pointmap_aug",
        action="store_true",
        default=False,
        help="Nonlinear depth distortion + noise + dropout augmentation on pointmap",
    )
    parser.add_argument(
        "--mesh_aug_prob",
        type=float,
        default=0.0,
        help="Probability of applying yaw-rotation augmentation (0.0 = disabled, 1.0 = always)",
    )

    # ---------- Curriculum Augmentation ----------
    parser.add_argument(
        "--use_curriculum_aug",
        action="store_true",
        default=False,
        help=(
            "Curriculum augmentation: no aug early in training, increasing aug strength later. "
            "Use together with --use_pointmap_aug / --mesh_aug_prob."
        ),
    )
    parser.add_argument(
        "--curriculum_warmup_ratio",
        type=float,
        default=0.1,
        help="Fraction of total steps to warm up without augmentation (default: 0.1 = 10%%)",
    )


    parser.add_argument(
        "--post_training",
        action="store_true",
        default=False,
    )

    # Parse the arguments
    args, extras = parser.parse_known_args()
    # Parse the config file
    configs = get_configs(args.config, extras)  # change yaml configs by `extras`

    from torch.utils.data import ConcatDataset
    from src.datasets import (
        MergedDataset as MergedDataset,
        BatchedMergedDataset as BatchedMergedDataset,
        MultiEpochsDataLoader,
        yield_forever,
    )

    args.val_guidance_scales = [float(x[0]) if isinstance(x, list) else float(x) for x in args.val_guidance_scales]
    if args.max_val_steps > 0: 
        # If enable validation, the max_val_steps must be a multiple of nrow
        # Always keep validation batchsize 1
        divider = configs["val"]["nrow"]
        args.max_val_steps = max(args.max_val_steps, divider)
        if args.max_val_steps % divider != 0:
            args.max_val_steps = (args.max_val_steps // divider + 1) * divider

    # Create an experiment directory using the `tag`
    if args.tag is None:
        args.tag = time.strftime("%Y%m%d_%H_%M_%S")
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    eval_dir = os.path.join(exp_dir, "evaluations")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = get_accelerate_logger(__name__, log_level="INFO")
    file_handler = logging.FileHandler(os.path.join(exp_dir, "log.txt"))  # output to file
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    ))
    logger.logger.addHandler(file_handler)
    logger.logger.propagate = True  # propagate to the root logger (console)

    # Set DeepSpeed config
    if args.use_deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=args.max_grad_norm,
            zero_stage=int(args.zero_stage),
            offload_optimizer_device="cpu",  # hard-coded here, TODO: make it configurable
        )
    else:
        deepspeed_plugin = None

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=args.ddp_find_unused_parameters
    )

    grad_accum_plugin = None
    if args.ddp_find_unused_parameters and args.gradient_accumulation_steps > 1:
        # Branch-dependent unused parameters and DDP no_sync do not mix well:
        # different micro-steps can exercise different heads, which can trip
        # PyTorch reducer bugs during accumulated backward passes.
        grad_accum_plugin = GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps,
            sync_each_batch=True,
        )

    accelerator_kwargs = dict(
        project_dir=exp_dir,
        mixed_precision=args.mixed_precision,
        split_batches=False,  # batch size per GPU
        dataloader_config=DataLoaderConfiguration(non_blocking=args.pin_memory),
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=[ddp_kwargs, InitProcessGroupKwargs(timeout=timedelta(minutes=60))],
    )
    if grad_accum_plugin is not None:
        accelerator_kwargs["gradient_accumulation_plugin"] = grad_accum_plugin
    else:
        accelerator_kwargs["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    # Initialize the accelerator
    accelerator = Accelerator(**accelerator_kwargs)
    logger.info(
        "Attention backend: %s (%s)",
        ATTN_BACKEND_CHOICE,
        ATTN_BACKEND_REASON,
    )
    logger.info(
        "DDP find_unused_parameters: %s",
        args.ddp_find_unused_parameters,
    )
    logger.info(
        "Gradient accumulation sync_each_batch: %s",
        bool(grad_accum_plugin is not None and grad_accum_plugin.sync_each_batch),
    )
    logger.info(f"Accelerator state:\n{accelerator.state}\n")
    
    
    # Set the random seed
    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)
        logger.info(f"You have chosen to seed([{args.seed}]) the experiment [{args.tag}]\n")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Create unified merged dataset
    _mix_parts = args.dataset_mix.split("+")
    use_future3d = "future" in _mix_parts
    use_scannet = "scannet" in _mix_parts
    use_coco = "coco" in _mix_parts

    logger.info(f"Creating merged dataset with: 3D-FUTURE={use_future3d}, ScanNet={use_scannet}, COCO={use_coco}")
    if args.no_pointmap:
        logger.info("Pointmap conditioning DISABLED (--no_pointmap): model will not receive pointmap tokens")
    if args.use_pointmap_aug:
        logger.info("Pointmap augmentation ENABLED (gamma distortion + noise + dropout)")
    if args.mesh_aug_prob > 0:
        logger.info(f"Mesh rotation augmentation ENABLED (prob={args.mesh_aug_prob})")
    if args.use_curriculum_aug:
        logger.info(f"Curriculum augmentation ENABLED (warmup_ratio={args.curriculum_warmup_ratio})")

    # When --no_pointmap is set, explicitly pass use_pointmap=False so the
    # dataset omits pointmap keys from each item (scene normalization still
    # uses pointmap internally for coordinate scaling).
    _use_pointmap = not args.no_pointmap

    # Shared progress variable for curriculum augmentation (multiprocessing.Value)
    from multiprocessing import Value
    import ctypes as _ctypes
    _aug_progress = Value(_ctypes.c_float, 0.0)  # 0.0 → 1.0 during training

    train_dataset = BatchedMergedDataset(
        configs=configs,
        batch_size=configs["train"]["batch_size_per_gpu"],
        is_main_process=accelerator.is_main_process,
        shuffle=True,
        training=True,
        use_latent=args.use_latent,
        use_future3d=use_future3d,
        use_scannet=use_scannet,
        use_coco=use_coco,
        use_pointmap=_use_pointmap,
        use_high_pointmap=args.use_high_pointmap,
        use_pointmap_aug=args.use_pointmap_aug,
        mesh_aug_prob=args.mesh_aug_prob,
        aug_progress=_aug_progress if args.use_curriculum_aug else None,
        curriculum_warmup_ratio=args.curriculum_warmup_ratio,
    )

    _val_use_coco = "coco" in args.dataset_mix if hasattr(args, "dataset_mix") else False
    val_dataset = MergedDataset(
        configs=configs,
        training=False,
        use_latent=args.use_latent,
        use_future3d=True,
        use_scannet=False,
        use_coco=_val_use_coco,
        use_pointmap=_use_pointmap,
        use_high_pointmap=args.use_high_pointmap,
    )
    
    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=configs["train"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        collate_fn=train_dataset.collate_fn,
        shuffle=False,  # Already shuffled in dataset
    )
    val_loader = MultiEpochsDataLoader(
        val_dataset,
        batch_size=configs["val"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )
    random_val_loader = MultiEpochsDataLoader(
        val_dataset,
        batch_size=configs["val"]["batch_size_per_gpu"],
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )

    logger.info(f"Loaded [{len(train_dataset)}] training samples and [{len(val_dataset)}] validation samples\n")

    # Compute the effective batch size and scale learning rate
    total_batch_size = configs["train"]["batch_size_per_gpu"] * \
        accelerator.num_processes * args.gradient_accumulation_steps
    configs["train"]["total_batch_size"] = total_batch_size
    if args.scale_lr:
        configs["optimizer"]["lr"] *= (total_batch_size / 256)
        configs["lr_scheduler"]["max_lr"] = configs["optimizer"]["lr"]
    
    workspace_dir = str(REPO_ROOT)
    ss_generator_config_path = os.path.join("checkpoints", "hf", "ss_generator.yaml")
    ss_generator_ckpt_path = os.path.join("checkpoints", "hf", "ss_generator.ckpt")
    trellis_encoder_ckpt = os.path.join("checkpoints", "ckpt/ckpts/ss_enc_conv3d_16l8_fp16")

    logger.info("Initializing the model...")
    ss_encoder = load_trellis_ss_wrapper(
        Path(workspace_dir) / trellis_encoder_ckpt,
        kind="encoder",
        device=accelerator.device,
    )
    
    print("Init ss_generator (floor rotation prediction)", configs["dataset"]["voxel_resolution"]//4)
    transformer, missing, _ = init_ss_generator_v1_4(
        ss_generator_config_path,
        ss_generator_ckpt_path,
        workspace_dir=workspace_dir,
        device=accelerator.device,
        resolution=configs["dataset"]["voxel_resolution"]//4,
    )

    # v3_input / v3_attn_fix / v3_attn_fix_coco: protect shape so only poses attend globally
    # v1: no protection (legacy behavior, shape+pose attend together)
    if args.load_ckpt is not None and not args.scratch:
        ckpt = torch.load(args.load_ckpt, map_location=accelerator.device, weights_only=False)

        ckpt_sd = strip_module_prefix(ckpt)
        has_base_keys = any(k.startswith("base_model.") for k in ckpt_sd)
        if has_base_keys:
            missing, unexpected = transformer.load_state_dict(ckpt_sd, strict=False)
        else:
            mapped_sd = {f"base_model.{k}": v for k, v in ckpt_sd.items()}
            missing, unexpected = transformer.load_state_dict(mapped_sd, strict=False)
        logger.info(f"ckpt loaded: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            logger.info(f"missing keys: {missing}")

    ss_condition_embedder = init_ss_condition_embedder(
        ss_generator_config_path,
        ss_generator_ckpt_path,
        workspace_dir=workspace_dir,
        device=accelerator.device,
    )
    ss_condition_embedding = ConditionEmbedding(
        ss_condition_embedder, configs.get("ss_condition_input_mapping", [])
    )
    
    transformer.requires_grad_(True)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # Create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:

                # `models` arrives as a list; use the provided state dicts in `weights`
                # to avoid attribute errors when the container is not a single Module.
                for idx, state_dict in enumerate(weights):
                    fname = "transformer.pt" if len(weights) == 1 else f"transformer_{idx}.pt"
                    save_path = os.path.join(output_dir, fname)
                    torch.save(state_dict, save_path)
                    logger.info("Saved model checkpoint to %s", save_path)
                


        accelerator.register_save_state_pre_hook(save_model_hook)

    # Initialize the optimizer and learning rate scheduler
    logger.info("Initializing the optimizer and learning rate scheduler...\n")
    name_lr_mult = configs["train"].get("name_lr_mult", None)
    lr_mult = configs["train"].get("lr_mult", 1.0)
    params, params_lr_mult, names_lr_mult = [], [], []
    for name, param in transformer.named_parameters():
        if name_lr_mult is not None:
            for k in name_lr_mult.split(","):
                if k in name:
                    params_lr_mult.append(param)
                    names_lr_mult.append(name)
            if name not in names_lr_mult:
                params.append(param)
        else:
            params.append(param)

    
    optimizer = get_optimizer(
        params=[
            {"params": params, "lr": configs["optimizer"]["lr"]},
            {"params": params_lr_mult, "lr": configs["optimizer"]["lr"] * lr_mult}
        ],
        **configs["optimizer"]
    )
    if name_lr_mult is not None:
        logger.info(f"Learning rate x [{lr_mult}] parameter names: {names_lr_mult}\n")

    configs["lr_scheduler"]["total_steps"] = configs["train"]["epochs"] * math.ceil(
        len(train_loader) // accelerator.num_processes / args.gradient_accumulation_steps)  # only account updated steps
    configs["lr_scheduler"]["total_steps"] *= accelerator.num_processes  # for lr scheduler setting
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] *= accelerator.num_processes  # for lr scheduler setting
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, **configs["lr_scheduler"])
    configs["lr_scheduler"]["total_steps"] //= accelerator.num_processes  # reset for multi-gpu
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] //= accelerator.num_processes  # reset for multi-gpu

    # Prepare everything with `accelerator`
    transformer, optimizer, lr_scheduler, train_loader, val_loader, random_val_loader = accelerator.prepare(
        transformer, optimizer, lr_scheduler, train_loader, val_loader, random_val_loader
    )

    # Set classes explicitly for everything
    transformer: DistributedDataParallel
    optimizer: AcceleratedOptimizer
    lr_scheduler: AcceleratedScheduler
    train_loader: DataLoaderShard
    val_loader: DataLoaderShard
    random_val_loader: DataLoaderShard

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Initialize EMA model (optional)
    ema_model = None
    if args.use_ema:
        ema_kwargs = dict(configs["train"].get("ema_kwargs", {}))
        if args.ema_device == "cpu":
            ema_kwargs["device"] = "cpu"
        ema_model = MyEMAModel(
            accelerator.unwrap_model(transformer).parameters(),
            **ema_kwargs,
        )
        logger.info(f"EMA model initialized with kwargs: {ema_kwargs}\n")

    # Training configs after distribution and accumulation setup
    updated_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updated_steps = configs["lr_scheduler"]["total_steps"]
    if args.max_train_steps is None:
        args.max_train_steps = total_updated_steps
    assert configs["train"]["epochs"] * updated_steps_per_epoch == total_updated_steps
    if accelerator.num_processes > 1 and accelerator.is_main_process:
        print()
    accelerator.wait_for_everyone()
    logger.info(f"Total batch size: [{total_batch_size}]")
    logger.info(f"Learning rate: [{configs['optimizer']['lr']}]")
    logger.info(f"Gradient Accumulation steps: [{args.gradient_accumulation_steps}]")
    logger.info(f"Total epochs: [{configs['train']['epochs']}]")
    logger.info(f"Total steps: [{total_updated_steps}]")
    logger.info(f"Steps for updating per epoch: [{updated_steps_per_epoch}]")
    logger.info(f"Steps for validation: [{len(val_loader)}]\n")

    # (Optional) Load checkpoint
    global_update_step = 0
    if args.resume_from_iter is not None:
        if args.resume_from_iter < 0:
            args.resume_from_iter = int(sorted(os.listdir(ckpt_dir))[-1])
        logger.info(f"Load checkpoint from iteration [{args.resume_from_iter}]\n")
        # Load everything
        if version.parse(torch.__version__) >= version.parse("2.4.0"):
            torch.serialization.add_safe_globals([
                int, list, dict, 
                defaultdict,
                Any,
                DictConfig, ListConfig, Metadata, ContainerMetadata, AnyNode
            ]) # avoid deserialization error when loading optimizer state
        accelerator.load_state(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}"))  # torch < 2.4.0 here for `weights_only=False`
        global_update_step = int(args.resume_from_iter)

        # Resume EMA state if available
        if args.use_ema and ema_model is not None:
            ema_ckpt_path = os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}_ema.pt")
            if os.path.exists(ema_ckpt_path):
                ema_state = torch.load(ema_ckpt_path, map_location="cpu")
                ema_model.load_state_dict(ema_state)
                logger.info(f"EMA state loaded from {ema_ckpt_path}\n")
            else:
                logger.warning(f"EMA checkpoint not found at {ema_ckpt_path}, starting EMA from scratch.\n")

    # Save all experimental parameters and model architecture of this run to a file (args and configs)
    if accelerator.is_main_process:
        exp_params = save_experiment_params(args, configs, exp_dir)
        save_model_architecture(accelerator.unwrap_model(transformer), exp_dir)

    accelerator.wait_for_everyone()

    # WandB logger
    if accelerator.is_main_process:
        
        if wandb is None:
            logger.warning("wandb not installed; skipping experiment logging.")
        else:
            if args.offline_wandb:
                os.environ["WANDB_MODE"] = "offline"
            wandb.init(
                project=PROJECT_NAME, name=args.tag,
                config=exp_params, dir=exp_dir,
                resume=True
            )
            # Wandb artifact for logging experiment information
            arti_exp_info = wandb.Artifact(args.tag, type="exp_info")
            arti_exp_info.add_file(os.path.join(exp_dir, "params.yaml"))
            arti_exp_info.add_file(os.path.join(exp_dir, "model.txt"))
            arti_exp_info.add_file(os.path.join(exp_dir, "log.txt"))  # only save the log before training
            wandb.log_artifact(arti_exp_info)

    accelerator.wait_for_everyone()

    # Start training
    if accelerator.is_main_process:
        print()
    logger.info(f"Start training into {exp_dir}\n")
    logger.logger.propagate = False  # not propagate to the root logger (console)
    progress_bar = tqdm(
        range(total_updated_steps),
        initial=global_update_step,
        desc="Training",
        ncols=125,
        disable=not accelerator.is_main_process
    )
    for batch in yield_forever(train_loader):
        # from pytorch3d.transforms import rotation_6d_to_matrix
        # print(rotation_6d_to_matrix(batch["pred_6drotation_normalized"]))
        # print(batch['uid'])
        # raise Exception("Debugging: check pred_6drotation_normalized output")
        
        if global_update_step == args.max_train_steps:
            progress_bar.close()
            logger.logger.propagate = True  # propagate to the root logger (console)
            if accelerator.is_main_process and wandb is not None:
                wandb.finish()
            logger.info("Training finished!\n")
            return
        
        transformer.train()

        # Update curriculum augmentation progress (shared memory -> passed to DataLoader workers)
        if args.use_curriculum_aug:
            _aug_progress.value = global_update_step / max(1, total_updated_steps)

        with accelerator.accumulate(transformer):
            # Move raw inputs to device and normalize to channel-first layout for the condition embedder
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(accelerator.device).float()

            condition_args, condition_kwargs = ss_condition_embedding(batch)
            cond = condition_args[0] if len(condition_args) > 0 else None

            with torch.no_grad():
                shape_latent = ss_encoder(batch["voxels"])['z']
                shape_latent = shape_latent.reshape(batch["voxels"].shape[0], 8,-1).transpose(1,2)  # [N, token_len, in_ch]

            latents = {"shape": shape_latent}

            # V2: dual cross-attention (image cond + pointmap cond); V1: standard single cond
            outputs = transformer(latents, cond, num_parts=batch["num_parts"].long())

            detail_losses = {}
            loss = torch.tensor(0.0, device=accelerator.device)

            trainable1 = batch['trainable'].to(dtype=torch.bool, device=accelerator.device)
            trainable2 = batch.get('trainable_precise', trainable1).to(dtype=torch.bool, device=accelerator.device)

            _zero = torch.tensor(0.0, device=accelerator.device)

            if args.use_logscale:
                outputs["scale"] = torch.exp(outputs["scale"])

            is_scannet = batch["is_scannet"].to(device=accelerator.device, dtype=torch.bool)
            has_scannet = bool(is_scannet.any().item())

            gt_xz2f = batch["xz2f_rot"][is_scannet]

            if has_scannet:
                pred_xz2f = outputs["xz2f_rot"][is_scannet]
                detail_losses["xz2f_rot"] = tF.l1_loss(pred_xz2f, gt_xz2f)

                if trainable1.any():
                    detail_losses["scale"] = abs(batch["scale"].unsqueeze(1)[trainable1]-outputs["scale"][trainable1]).mean()
                    detail_losses["pred_translation"] = abs(batch["pred_translation"].unsqueeze(1)[trainable1]-outputs["pred_translation"][trainable1]).mean()
                else:
                    detail_losses["scale"] = _zero
                    detail_losses["pred_translation"] = _zero
                if trainable2.any():
                    detail_losses["pred_6drotation_normalized"] = abs(batch["pred_6drotation_normalized"].unsqueeze(1)[trainable2]-outputs["pred_6drotation_normalized"][trainable2]).mean()
                else:
                    detail_losses["pred_6drotation_normalized"] = _zero

            else:
                pred_xz2f = None
                detail_losses["xz2f_rot"] = _zero
                detail_losses["scale"] = _zero
                detail_losses["pred_translation"] = _zero
                detail_losses["pred_6drotation_normalized"] = _zero

            
            loss += detail_losses["xz2f_rot"]
            loss += detail_losses["pred_translation"]
            loss += detail_losses["pred_6drotation_normalized"]
            loss += detail_losses["scale"]

            if not args.post_training:
                batch['pm_surface_pts'] = batch['pm_surface_pts'] @ rotation_6d_to_matrix(gt_xz2f)
                outputs['translation']   = outputs['pred_translation']
                outputs['6drotation_normalized'] = outputs['pred_6drotation_normalized']

            idx = 0 
            xz2f_rot_std = 0
            for n in batch["num_parts"].long():
                xz2f_rot_std = xz2f_rot_std+outputs["xz2f_rot"][idx:idx+n].std(0).abs().sum()
                idx += n
            
            detail_losses["xz2f_rot_std"] = xz2f_rot_std / batch["num_parts"].shape[0]
            loss += detail_losses["xz2f_rot_std"]


            # Pointmap visible-surface alignment loss (instance-mask-region pointmap <-> predicted mesh pose)
            if args.use_pm_surface_loss and "pm_surface_pts" in batch and (global_update_step > 100 or args.post_training):
                from src.loss_function import compute_pointmap_surface_loss
                pm_loss = compute_pointmap_surface_loss(
                    outputs, batch,
                    weight=args.pm_surface_loss_weight,
                    device=accelerator.device,
                )
                detail_losses["pm_surface_loss"] = pm_loss
                loss = loss + pm_loss


            optimizer.zero_grad()
            loss = loss.mean()
            if torch.isnan(loss) or torch.isinf(loss):

                torch.save(cond, "test_cond.pt")
                torch.save(outputs, "test_outputs.pt")
                torch.save(batch, "test_batch.pt")

                raise Exception(
                    f"Invalid loss value at step {global_update_step}: {detail_losses.items()}. {cond}. " +
                    "This may be due to exploding gradients or numerical instability. " +
                    "Consider reducing the learning rate or checking the data for anomalies."
                )
            
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

            lr_scheduler.step()
            optimizer.step()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            # EMA update
            if args.use_ema and ema_model is not None:
                ema_model.step(accelerator.unwrap_model(transformer).parameters())

            # Gather the losses across all processes for logging (if we use distributed training)
            loss = accelerator.gather(loss.detach()).mean()

            detail_logs = {}
            if isinstance(detail_losses, dict):
                for k, v in detail_losses.items():
                    if isinstance(v, torch.Tensor):
                        detail_logs[f"loss/{k}"] = v.detach().mean().item()
                    elif isinstance(v, (float, int)):
                        detail_logs[f"loss/{k}"] = float(v)

            logs = {
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            logs.update(detail_logs)
            

            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
            global_update_step += 1

            log_line = (
                f"[{global_update_step:06d} / {total_updated_steps:06d}] "
                f"loss: {logs['loss']:.4f}, lr: {logs['lr']:.2e}"
            )
            if args.use_ema and ema_model is not None:
                log_line += f", ema_decay: {ema_model.cur_decay_value:.6f}"
            logger.info(log_line)

            if global_update_step == 1 and accelerator.is_main_process:
                def _tensor_stats_short(t: torch.Tensor) -> str:
                    return f"shape={tuple(t.shape)}, mean={t.float().mean().item():.4f}, std={t.float().std().item():.4f}"
                logger.info("First-step input summary:")
                for key, tensor in latents.items():
                    logger.info("  %s: %s", key, _tensor_stats_short(tensor))
                if isinstance(cond, torch.Tensor):
                    logger.info("  condition: %s", _tensor_stats_short(cond))
                if detail_logs:
                    logger.info("First-step loss breakdown: %s", detail_logs)

            # Log the training progress
            if (
                global_update_step % configs["train"]["log_freq"] == 0 
                or global_update_step == 1
                or global_update_step % updated_steps_per_epoch == 0 # last step of an epoch
            ):  
                if accelerator.is_main_process and wandb is not None:
                    log_payload = {
                        "training/loss": logs["loss"],
                        "training/lr": logs["lr"],
                    }
                    for k, v in detail_logs.items():
                        log_payload[f"training/{k}"] = v
                    wandb.log(log_payload, step=global_update_step)

            # Save checkpoint
            if (
                global_update_step % configs["train"]["save_freq"] == 0  # 1. every `save_freq` steps
                or global_update_step % (configs["train"]["save_freq_epoch"] * updated_steps_per_epoch) == 0  # 2. every `save_freq_epoch` epochs
                or global_update_step == total_updated_steps # 3. last step of an epoch
                # or global_update_step == 1 # 4. first step
            ): 

                if accelerator.is_main_process:
                    torch.save(transformer.state_dict(), os.path.join(ckpt_dir, f"{global_update_step:06d}.pt"))
                    if args.use_ema and ema_model is not None:
                        torch.save(ema_model.state_dict(), os.path.join(ckpt_dir, f"{global_update_step:06d}_ema.pt"))
                accelerator.wait_for_everyone()  # ensure all processes have finished saving

            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    main()
