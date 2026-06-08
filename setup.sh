# =============================================================================
# GALP — Geometry-Aware Layout Prediction : Environment Setup
# =============================================================================
# Only the libraries actually required to run `demo.py` (inference) are
# installed. The dependency set below was derived by tracing the import graph
# starting from demo.py:
#
#   demo.py -> src.train (+ src.sam3d_objects), src.datasets.data_utils,
#              inference_utils (+ pytorch3d, moge)
#
# Notes from the trace:
#   * src.sam3d_objects sparse backend defaults to "spconv"  -> spconv REQUIRED
#     (imported at load time; set SPARSE_BACKEND/ATTN_BACKEND to change).
#   * Attention backend defaults to "flash_attn" -> flash-attn is REQUIRED
#     (override with ATTN_BACKEND=sdpa|xformers if you cannot install it).
#   * pytorch3d, lightning, timm, einops, matplotlib are on the demo path.
#   * nvdiffrast / kaolin / gsplat / torch_cluster / vox2seq / diffoctreerast /
#     diff_gaussian_rasterization are TRAINING-ONLY or unused -> NOT installed.
# =============================================================================

# ----------------------------- Read Arguments --------------------------------
TEMP=`getopt -o h --long help,new-env,basic,pytorch3d,spconv,moge,xformers,flash-attn,all -n 'setup.sh' -- "$@"`

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
PYTORCH3D=false
SPCONV=false
MOGE=false
XFORMERS=false
FLASHATTN=false
ALL=false
ERROR=false

if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --pytorch3d) PYTORCH3D=true ; shift ;;
        --spconv) SPCONV=true ; shift ;;
        --moge) MOGE=true ; shift ;;
        --xformers) XFORMERS=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --all) ALL=true ; shift ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: setup.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help        Display this help message"
    echo "  --new-env         Create a new conda environment 'galp' (python 3.10, torch 2.4.0 / cu121)"
    echo "  --basic           Install the core pip dependencies required by demo.py"
    echo "  --pytorch3d       Install PyTorch3D (prebuilt wheel for torch 2.4.0 + cu121)"
    echo "  --spconv          Install spconv (default sparse backend of sam3d_objects)"
    echo "  --moge            Install MoGe (monocular geometry estimation) used by inference"
    echo "  --flash-attn      Install flash-attn (default attention backend)"
    echo "  --xformers        [optional] Install xformers (set ATTN_BACKEND=xformers to use)"
    echo "  --all             Install everything needed for the demo"
    echo "                    ( = --basic --pytorch3d --spconv --moge --flash-attn )"
    echo ""
    echo "Quick start:"
    echo "  bash setup.sh --new-env"
    echo "  conda activate galp"
    echo "  bash setup.sh --all"
    return 0 2>/dev/null || exit 0
fi

# --all is a convenience flag that turns on every demo dependency.
if [ "$ALL" = true ] ; then
    BASIC=true
    PYTORCH3D=true
    SPCONV=true
    MOGE=true
    FLASHATTN=true
fi

# ----------------------------- System Information ----------------------------
WORKDIR=$(pwd)

# Pinned target stack (must match the prebuilt wheels installed below).
PYTORCH_VERSION="2.4.0"
CUDA_VERSION="12.1"
CUDA_MAJOR_VERSION="12"

if [ "$NEW_ENV" = true ] ; then
    conda create -n galp python=3.10 -y
    conda activate galp
    conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
fi

# Detect the live torch/CUDA versions when torch is already importable, so the
# wheel-specific installs below pick the matching build.
if python -c "import torch" 2>/dev/null ; then
    PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    DETECTED_CUDA=$(python -c "import torch; print(torch.version.cuda)")
    if [ -n "$DETECTED_CUDA" ] && [ "$DETECTED_CUDA" != "None" ] ; then
        CUDA_VERSION=$DETECTED_CUDA
        CUDA_MAJOR_VERSION=$(echo $CUDA_VERSION | cut -d'.' -f1)
    fi
    echo "[SYSTEM] PyTorch Version: $PYTORCH_VERSION, CUDA Version: $CUDA_VERSION"
fi

# ----------------------------- Basic Dependencies ----------------------------
if [ "$BASIC" = true ] ; then
    # Core array / image / mesh utilities
    pip install \
        numpy==1.26.4 \
        pillow \
        opencv-python-headless \
        matplotlib \
        trimesh \
        tqdm \
        packaging

    # Diffusion / transformer model stack
    pip install \
        diffusers \
        transformers \
        accelerate \
        safetensors \
        timm \
        einops

    # Config + checkpoint handling (hydra instantiate, lightning checkpoints)
    pip install \
        hydra-core \
        omegaconf \
        lightning \
        loguru \
        icecream

    # Optional (guarded by try/except in the code, but commonly used for I/O)
    pip install open3d wandb
fi

# ----------------------------- PyTorch3D -------------------------------------
# Required on the inference path (pytorch3d.ops / pytorch3d.transforms /
# pytorch3d.renderer). Use the matching prebuilt wheel to avoid a long source
# build. The URL below targets python 3.10 + cu121 + torch 2.4.0.
if [ "$PYTORCH3D" = true ] ; then
    PYT3D_TAG="py310_cu121_pyt240"
    echo "[PYTORCH3D] Installing prebuilt wheel: $PYT3D_TAG"
    pip install --no-index --no-cache-dir pytorch3d \
        -f "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/${PYT3D_TAG}/download.html" \
    || {
        echo "[PYTORCH3D] Prebuilt wheel not found for this stack. Building from source..."
        pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    }
fi

# ----------------------------- spconv (sparse backend) -----------------------
# sam3d_objects' sparse modules default to BACKEND="spconv" and import it at
# load time, so this is required for demo.py.
if [ "$SPCONV" = true ] ; then
    case $CUDA_MAJOR_VERSION in
        11) pip install spconv-cu118 ;;
        12) pip install spconv-cu120 ;;
        *)  echo "[SPCONV] Unsupported CUDA major version: $CUDA_MAJOR_VERSION (install spconv manually)" ;;
    esac
fi

# ----------------------------- MoGe ------------------------------------------
# Lazily loaded in inference_utils.run_moge() for scene pointmap estimation.
if [ "$MOGE" = true ] ; then
    # utils3d is a runtime dependency of MoGe; pin to the SceneGen-tested commit.
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
    pip install git+https://github.com/microsoft/MoGe.git
fi

# ----------------------------- Optional: xformers ----------------------------
# Not needed by default (attention backend = "sdpa"). Install only if you plan
# to run with ATTN_BACKEND=xformers.
if [ "$XFORMERS" = true ] ; then
    if [ "$CUDA_MAJOR_VERSION" = "12" ] ; then
        pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
    elif [ "$CUDA_MAJOR_VERSION" = "11" ] ; then
        pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
    else
        echo "[XFORMERS] Unsupported CUDA version: $CUDA_VERSION"
    fi
fi

# ----------------------------- flash-attn (default backend) ------------------
# Required: sam3d_objects attention defaults to BACKEND/ATTN = "flash_attn".
# (Override at runtime with ATTN_BACKEND=sdpa|xformers if it cannot be built.)
if [ "$FLASHATTN" = true ] ; then
    pip install flash-attn --no-build-isolation
fi

# numpy is pinned last to undo any accidental upgrade pulled by another package.
pip install numpy==1.26.4
