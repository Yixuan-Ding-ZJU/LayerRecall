# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import os
import sys
from collections.abc import Mapping

# torchrun no longer consistently prepends the script directory to sys.path,
# which breaks absolute project imports when launched from another cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# torchvision 0.27+ removed write_video/read_video. Several modules import the
# symbols at module import time, so patch them before importing project code.
import torchvision.io as _tv_io
if not hasattr(_tv_io, "write_video"):
    import imageio.v2 as _imageio_v2

    def _shim_write_video(filename, video_array, fps, **_unused):
        if hasattr(video_array, "detach"):
            video_array = video_array.detach().cpu().numpy()
        _imageio_v2.mimwrite(filename, video_array, fps=fps, codec="libx264", quality=8)

    _tv_io.write_video = _shim_write_video
if not hasattr(_tv_io, "read_video"):
    import imageio.v3 as _imageio_v3
    import torch as _torch_for_shim

    def _shim_read_video(filename, pts_unit="sec", output_format="THWC", **_unused):
        frames = _imageio_v3.imread(filename, plugin="pyav")
        tensor = _torch_for_shim.from_numpy(frames)
        if output_format == "TCHW":
            tensor = tensor.permute(0, 3, 1, 2)
        return tensor, None, {}

    _tv_io.read_video = _shim_read_video

import argparse
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import CausalDiffusionInferencePipeline
from utils.dataset import MultiTextConcatDataset, MultiVideoConcatDataset, eval_collate_fn, multi_video_collate_fn
from utils.misc import set_seed
from utils.config import normalize_config, section_get, wan_default_config
from utils.inference_utils import clean_fsdp_state_dict_keys, unwrap_generator_state_dict

from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller


def save_prompts_to_txt(prompts_for_sample, prompt_txt_path: str, is_main_process: bool):
    """Save per-block prompts alongside the video.

    Consecutive identical prompts are merged, e.g.:
        [0] a, [1] a, [2] b  =>  [0,1] a\\n[2] b\\n
    """
    try:
        with open(prompt_txt_path, "w", encoding="utf-8") as f:
            if len(prompts_for_sample) == 0:
                return
            current_prompt = prompts_for_sample[0]
            current_indices = [0]
            for seg_idx in range(1, len(prompts_for_sample)):
                p = prompts_for_sample[seg_idx]
                if p == current_prompt:
                    current_indices.append(seg_idx)
                else:
                    indices_str = ",".join(str(i) for i in current_indices)
                    f.write(f"[{indices_str}] {current_prompt}\n")
                    current_prompt = p
                    current_indices = [seg_idx]
            indices_str = ",".join(str(i) for i in current_indices)
            f.write(f"[{indices_str}] {current_prompt}\n")
    except Exception as e:
        if is_main_process:
            print(f"Warning: failed to save prompts to {prompt_txt_path}: {e}")

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
parser.add_argument(
    "--layer-recall-selection-mode",
    choices=("hard", "soft"),
    default=None,
    help="Override LayerRecall inference selection without editing the YAML.",
)
args = parser.parse_args()

config = normalize_config(OmegaConf.load(args.config_path))
if args.layer_recall_selection_mode is not None:
    if config.get("layer_recall", None) is None:
        raise ValueError(
            "--layer-recall-selection-mode requires a layer_recall config section"
        )
    config.layer_recall.layer_recall_selection_mode = (
        args.layer_recall_selection_mode
    )

if not hasattr(config, "sampling_steps") or config.sampling_steps is None:
    raise ValueError("sampling_steps must be defined in the inference config")

if not hasattr(config, "guidance_scale") or config.guidance_scale is None:
    config.guidance_scale = 1.0

config.use_ema = section_get(config, "inference", "use_ema", getattr(config, "use_ema", False))
config.output_folder = section_get(config, "inference", "output_folder", getattr(config, "output_folder", "videos/longlive2"))
config.num_samples = section_get(config, "inference", "num_samples", getattr(config, "num_samples", 1))
config.num_output_frames = getattr(config, "num_output_frames", config.image_or_video_shape[1])
config.save_with_index = getattr(config, "save_with_index", False)
config.inference_iter = getattr(config, "inference_iter", -1)


def _maybe_to_dict(value):
    if value is None:
        return None
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return dict(value)


def _config_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


_MISSING = object()


def _nested_get(mapping, *keys, default=None):
    value = mapping
    for key in keys:
        if value is None:
            return default
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, Mapping):
            if key not in value:
                return default
            value = value[key]
        else:
            value = getattr(value, key, _MISSING)
            if value is _MISSING:
                return default
    return value


def _config_layer_recall_enabled(config):
    return _config_bool(
        _nested_get(config, "layer_recall", "layer_recall_enabled", default=False)
    )


def _extract_layer_recall_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("LayerRecall checkpoint must be a mapping")
    if checkpoint.get("trainer") != "chpm":
        raise ValueError(
            "LayerRecall inference only accepts CHPM checkpoints; "
            f"got trainer={checkpoint.get('trainer')!r}"
        )
    if int(checkpoint.get("checkpoint_version", -1)) != 3:
        raise ValueError(
            "Unsupported CHPM checkpoint version; expected 3, "
            f"got {checkpoint.get('checkpoint_version')!r}"
        )
    state = checkpoint.get("layer_recall_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("CHPM checkpoint is missing layer_recall_state_dict")
    invalid = [key for key, value in state.items() if not torch.is_tensor(value)]
    if invalid:
        raise ValueError(
            "layer_recall_state_dict contains non-tensor values: "
            f"{invalid[:8]}"
        )
    return state

def _preflight_layer_recall_state_dict(model_state, layer_recall_state):
    model_keys = set(model_state)
    checkpoint_keys = set(layer_recall_state)
    model_layer_recall_keys = {key for key in model_keys if "layer_recall" in str(key)}
    missing_in_model = sorted(checkpoint_keys - model_keys)
    missing_in_checkpoint = sorted(model_layer_recall_keys - checkpoint_keys)
    shape_mismatch = [
        (key, tuple(layer_recall_state[key].shape), tuple(model_state[key].shape))
        for key in sorted(checkpoint_keys.intersection(model_keys))
        if tuple(layer_recall_state[key].shape) != tuple(model_state[key].shape)
    ]
    if missing_in_model or missing_in_checkpoint or shape_mismatch:
        details = []
        if missing_in_model:
            details.append(f"unexpected checkpoint keys: {missing_in_model[:8]}")
        if missing_in_checkpoint:
            details.append(f"missing checkpoint keys: {missing_in_checkpoint[:8]}")
        if shape_mismatch:
            details.append(f"shape mismatches: {shape_mismatch[:8]}")
        raise ValueError(
            "The LayerRecall checkpoint does not exactly match the initialized inference "
            "LayerRecall architecture; " + "; ".join(details)
        )


def _load_layer_recall_checkpoint_if_configured(pipeline, config, local_rank):
    layer_recall_ckpt_path = section_get(
        config,
        "checkpoints",
        "layer_recall_ckpt",
        getattr(config, "layer_recall_ckpt", None),
    )
    if not layer_recall_ckpt_path:
        return None
    if not _config_layer_recall_enabled(config):
        raise ValueError(
            "checkpoints.layer_recall_ckpt is set, but layer_recall.layer_recall_enabled is false"
        )

    checkpoint = torch.load(layer_recall_ckpt_path, map_location="cpu", weights_only=False)
    layer_recall_state = _extract_layer_recall_state_dict(checkpoint)
    step = checkpoint.get("step")

    current_keys = [key for key in layer_recall_state if "layer_recall_current_" in key]
    if not current_keys:
        raise ValueError(
            "CHPM checkpoint does not contain current-conditioned LayerRecall parameters"
        )
    if not _config_bool(
        _nested_get(
            config,
            "layer_recall",
            "layer_recall_current_conditioned_enabled",
            default=False,
        )
    ):
        raise ValueError(
            "CHPM checkpoint is current-conditioned, but the inference config disables it"
        )

    model_state = pipeline.generator.state_dict()
    _preflight_layer_recall_state_dict(model_state, layer_recall_state)

    incompatible = pipeline.generator.load_state_dict(layer_recall_state, strict=False)
    missing_layer_recall = [key for key in incompatible.missing_keys if "layer_recall" in key]
    unexpected_layer_recall = [key for key in incompatible.unexpected_keys if "layer_recall" in key]
    if missing_layer_recall or unexpected_layer_recall:
        raise ValueError(
            "Failed to load a complete LayerRecall checkpoint overlay: "
            f"missing_layer_recall={missing_layer_recall[:8]}, unexpected_layer_recall={unexpected_layer_recall[:8]}"
        )
    if local_rank == 0:
        suffix = f" at step {step}" if step is not None else ""
        print(f"[layer_recall] loaded LayerRecall checkpoint overlay{suffix}: {layer_recall_ckpt_path}")
        print(f"[layer_recall] loaded {len(layer_recall_state)} LayerRecall tensors; current_conditioned={bool(current_keys)}")
    return {
        "path": layer_recall_ckpt_path,
        "num_tensors": len(layer_recall_state),
        "trainer": "chpm",
        "step": step,
    }


def _expected_inference_samples(config):
    inference_iter = int(getattr(config, "inference_iter", -1))
    if inference_iter >= 0:
        return inference_iter + 1
    return None


def _resolve_torch_compile(config):
    setting = getattr(config, "torch_compile", False)
    if isinstance(setting, str) and setting.strip().lower() == "auto":
        min_samples = int(getattr(config, "torch_compile_min_samples", 2))
        expected_samples = _expected_inference_samples(config)
        if expected_samples is not None and expected_samples < min_samples:
            return (
                False,
                "auto disabled because expected samples "
                f"({expected_samples}) < torch_compile_min_samples ({min_samples})",
            )
        return True, "auto enabled for repeated inference"
    return _config_bool(setting, default=False), "explicit setting"


def configure_generator_torch_compile(pipeline, config):
    compile_enabled, reason = _resolve_torch_compile(config)
    if not compile_enabled:
        if local_rank == 0 and str(getattr(config, "torch_compile", "false")).lower() == "auto":
            print(f"[torch.compile] skipped: {reason}")
        return
    target = str(getattr(config, "torch_compile_target", "generator_model")).lower()
    if target not in {"generator_model", "model"}:
        if local_rank == 0:
            print(f"[torch.compile][warn] Unsupported target={target}; expected generator_model")
        return
    if not hasattr(pipeline.generator, "configure_torch_compile"):
        if local_rank == 0:
            print("[torch.compile][warn] Current generator does not expose configure_torch_compile; skipping")
        return
    compiled = pipeline.generator.configure_torch_compile(
        backend=str(getattr(config, "torch_compile_backend", "inductor")),
        mode=getattr(config, "torch_compile_mode", "max-autotune-no-cudagraphs"),
        fullgraph=_config_bool(getattr(config, "torch_compile_fullgraph", False)),
        dynamic=_config_bool(getattr(config, "torch_compile_dynamic", False)),
        options=_maybe_to_dict(getattr(config, "torch_compile_options", None)),
        suppress_errors=_config_bool(getattr(config, "torch_compile_suppress_errors", True), default=True),
    )
    if local_rank == 0:
        status = "enabled" if compiled else "not enabled"
        print(f"[torch.compile] {status}: target={target}")

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(config.seed + local_rank)
    config.distributed = True  # Mark as distributed for pipeline
else:
    local_rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False  # Mark as non-distributed

print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40

torch.set_grad_enabled(False)


# Initialize pipeline
pipeline = CausalDiffusionInferencePipeline(config, device=device)

# --------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import configure_lora_for_model
import peft

merge_lora = bool(getattr(config, "merge_lora", False))
has_lora_adapter = bool(getattr(config, "adapter", None) and configure_lora_for_model is not None)
generator_checkpoint = None
generator_lora_state = None
generator_ckpt_path = getattr(config, "generator_ckpt", None)
if generator_ckpt_path:
    generator_checkpoint = torch.load(generator_ckpt_path, map_location="cpu")
    is_lora_only_checkpoint = (
        isinstance(generator_checkpoint, dict)
        and "generator_lora" in generator_checkpoint
        and not any(key in generator_checkpoint for key in ("generator", "generator_ema", "model"))
    )
    if is_lora_only_checkpoint:
        generator_lora_state = generator_checkpoint["generator_lora"]
        if local_rank == 0:
            print(f"Found LoRA generator weights in {generator_ckpt_path}")
    else:
        raw_gen_state_dict = unwrap_generator_state_dict(generator_checkpoint, use_ema=config.use_ema)
        if config.use_ema:
            raw_gen_state_dict = clean_fsdp_state_dict_keys(raw_gen_state_dict)
        if config.use_ema:
            missing, unexpected = pipeline.generator.load_state_dict(raw_gen_state_dict, strict=False)
            if local_rank == 0:
                if len(missing) > 0:
                    print(f"[Warning] {len(missing)} parameters are missing when loading checkpoint: {missing[:8]} ...")
                if len(unexpected) > 0:
                    print(f"[Warning] {len(unexpected)} unexpected parameters encountered when loading checkpoint: {unexpected[:8]} ...")
        else:
            print(f"Loading generator from {generator_ckpt_path}")
            layer_recall_enabled = bool(getattr(config, "layer_recall_enabled", False) or getattr(getattr(config, "layer_recall", None), "layer_recall_enabled", False))
            missing, unexpected = pipeline.generator.load_state_dict(raw_gen_state_dict, strict=not layer_recall_enabled)
            if layer_recall_enabled and local_rank == 0:
                yx_missing = [key for key in missing if "layer_recall" in key]
                other_missing = [key for key in missing if "layer_recall" not in key]
                if yx_missing:
                    print(f"[layer_recall] initialized new parameters not found in checkpoint: {yx_missing[:12]}")
                if other_missing:
                    print(f"[layer_recall][Warning] non-LayerRecall missing checkpoint keys: {other_missing[:12]}")
                if unexpected:
                    print(f"[layer_recall][Warning] unexpected checkpoint keys: {unexpected[:12]}")

_load_layer_recall_checkpoint_if_configured(pipeline, config, local_rank)

pipeline.is_lora_enabled = False
pipeline.is_lora_merged = False

if has_lora_adapter:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
        if merge_lora:
            print("LoRA weights will be merged into the base model before inference")
    # Apply LoRA to the generator transformer after loading base weights.
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # Load LoRA weights from lora_ckpt. If omitted, fall back to generator_ckpt
    # when it directly contains generator_lora.
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA weights from lora_ckpt: {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    elif generator_lora_state is not None:
        if local_rank == 0:
            print(f"Loading LoRA weights from generator_ckpt: {generator_ckpt_path}")
        peft.set_peft_model_state_dict(pipeline.generator.model, generator_lora_state)  # type: ignore
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint configured; using initialized LoRA adapters")

    if merge_lora:
        if local_rank == 0:
            print("Merging LoRA weights into generator before inference...")
        pipeline.generator.model = pipeline.generator.model.merge_and_unload(safe_merge=True)
        pipeline.is_lora_merged = True
    else:
        pipeline.is_lora_enabled = True
elif merge_lora and local_rank == 0:
    print("merge_lora=True requested but no adapter config was found; continuing without LoRA merge")

del generator_checkpoint


# Move pipeline to appropriate dtype and device
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)

pipeline.generator.model.eval().requires_grad_(False)
configure_generator_torch_compile(pipeline, config)

vae_device_str = getattr(config, "vae_device", None)
use_dedicated_vae_device = bool(getattr(config, "streaming_vae", False)) and bool(vae_device_str)
if use_dedicated_vae_device:
    vae_device = torch.device(vae_device_str)
    pipeline.vae.to(device="cpu")
    pipeline.vae.to(device=vae_device)
    if hasattr(pipeline.vae, "mean"):
        pipeline.vae.mean = pipeline.vae.mean.to(device=vae_device)
        pipeline.vae.std = pipeline.vae.std.to(device=vae_device)
    if local_rank == 0:
        print(f"[inference] VAE on {vae_device}, diffusion on {device}")
else:
    pipeline.vae.to(device=device)
    if vae_device_str and local_rank == 0:
        print(f"[inference] Ignoring vae_device={vae_device_str} because streaming_vae is false")

# Create dataset
nfpb = getattr(config, 'num_frame_per_block', 8)
data_path = config.data_path
chunks_per_shot = getattr(config, 'chunks_per_shot', 0)
scene_cut_prefix = getattr(config, 'scene_cut_prefix', "The scene transitions. ")
if getattr(config, "i2v", False):
    model_name = config.model_kwargs.model_name
    frame_raw_height = list(config.image_or_video_shape)[3] * wan_default_config[model_name]["spatial_compression_ratio"]
    frame_raw_width = list(config.image_or_video_shape)[4] * wan_default_config[model_name]["spatial_compression_ratio"]
    temporal_compression_ratio = wan_default_config[model_name]["temporal_compression_ratio"]
    total_frames = (config.num_output_frames - 1) * temporal_compression_ratio + 1
    dataset = MultiVideoConcatDataset(
        data_dir=data_path,
        video_size=(frame_raw_height, frame_raw_width),
        total_frames=total_frames,
        deterministic=True,
        num_frame_per_block=nfpb,
        temporal_compression_ratio=temporal_compression_ratio,
        target_fps=24 if "5B" in model_name else 16,
        allow_padding=getattr(config, "allow_padding", False),
        min_latent_frames=getattr(config, "min_latent_frames", 0),
        single_video_only=getattr(config, "uniform_prompt", False),
        independent_first_frame=getattr(config, "independent_first_frame", False),
        return_image=True,
        max_chunks_per_shot=getattr(config, "max_chunks_per_shot", 0),
        scene_cut_prefix=scene_cut_prefix,
    )
    collate_fn = multi_video_collate_fn
    num_blocks = config.num_output_frames // nfpb
else:
    num_blocks = config.num_output_frames // nfpb
    dataset = MultiTextConcatDataset(
        data_path=data_path,
        num_blocks=num_blocks,
        chunks_per_shot=chunks_per_shot,
        scene_cut_prefix=scene_cut_prefix,
        deterministic=True,
    )
    collate_fn = eval_collate_fn
if local_rank == 0:
    print(f"[data] data_path={data_path}, mode={getattr(dataset, '_mode', dataset.__class__.__name__)}, num_blocks={num_blocks}")
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0,
                        drop_last=False, collate_fn=collate_fn)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []

    # MultiTextConcatDataset + eval_collate_fn: prompts[0] is List[str].
    block_prompts = list(batch['prompts'][0])
    prompt = block_prompts[0]  # for filename
    prompts = [block_prompts] * config.num_samples

    shape = config.image_or_video_shape
    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, shape[2], shape[3], shape[4]], device=device, dtype=torch.bfloat16
    )
    initial_latent = None
    if getattr(config, "i2v", False):
        image = batch["image"].to(device=device, dtype=torch.bfloat16)
        if image.ndim == 4:
            image = image.unsqueeze(2)
        elif image.ndim != 5:
            raise ValueError(f"Expected i2v image with shape [B,C,H,W] or [B,C,T,H,W], got {tuple(image.shape)}")
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        if initial_latent.shape[0] != config.num_samples:
            initial_latent = initial_latent.repeat(config.num_samples, 1, 1, 1, 1)
        if config.num_output_frames <= initial_latent.shape[1]:
            raise ValueError(
                f"num_output_frames must exceed the i2v conditioning frames; "
                f"got {config.num_output_frames} and {initial_latent.shape[1]}"
            )
    print("sampled_noise.device", sampled_noise.device)
    print("prompts", prompts)
    print('sampled_noise.shape', sampled_noise.shape, 'prompts', prompts)
    save_latents_only = section_get(
        config,
        "inference",
        "save_latents_only",
        getattr(config, "save_latents_only", getattr(config, "save_latent_only", False)),
        aliases=("save_latent_only", "return_latents"),
    )
    inference_kwargs = dict(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=save_latents_only,
    )
    if initial_latent is not None:
        inference_kwargs["initial_latent"] = initial_latent
    with torch.inference_mode():
        generated = pipeline.inference(**inference_kwargs)

    if not save_latents_only:
        current_video = rearrange(generated, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)

        # Final output video
        video = 255.0 * torch.cat(all_video, dim=1)

        # Clear VAE cache
        pipeline.vae.model.clear_cache()
    else:
        latents = generated

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        # Determine model type for filename
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "lora"
        elif getattr(config, 'use_ema', False):
            model_type = "ema"
        else:
            model_type = "regular"
            
        for seed_idx in range(config.num_samples):
            if config.save_with_index:
                base_name = f'YX_rank{rank}-{idx}-{seed_idx}_{model_type}'
            else:
                base_name = f'YX_rank{rank}-{prompt[:100]}-{seed_idx}_{model_type}'

            if save_latents_only:
                latent_path = os.path.join(config.output_folder, f'{base_name}.pt')
                torch.save(latents[seed_idx].cpu(), latent_path)
            else:
                output_path = os.path.join(config.output_folder, f'{base_name}.mp4')
                fps = 24 if '5B' in config.model_kwargs.model_name else 16
                write_video(output_path, video[seed_idx], fps=fps)

            prompt_txt_path = os.path.join(config.output_folder, f'{base_name}_prompts.txt')
            save_prompts_to_txt(
                prompts[seed_idx] if isinstance(prompts[seed_idx], list) else [prompts[seed_idx]],
                prompt_txt_path,
                is_main_process=(rank == 0),
            )

    if config.inference_iter != -1 and i >= config.inference_iter:
        break
