import json
import math
import numbers
import os
import re
import threading
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


_YX_THREAD_STATE = threading.local()

_YX_WAN22_5B_DEFAULTS = {
    "model_name": "Wan2.2-TI2V-5B",
    "num_heads": 24,
    "head_dim": 128,
    "num_layers": 30,
    "patch_size": (1, 2, 2),
}


def _YX_to_plain(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _YX_to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_YX_to_plain(v) for v in value]
    return value


def _YX_as_int_tuple(value: Any, default: Tuple[int, ...]) -> Tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.replace("x", ",").split(",")]
        value = [piece for piece in pieces if piece]
    try:
        parsed = tuple(int(item) for item in value)
    except TypeError:
        parsed = (int(value),)
    if len(parsed) == 0:
        return default
    return parsed


def _YX_validate_num_layers(num_layers: Any) -> int:
    if isinstance(num_layers, bool) or not isinstance(num_layers, numbers.Integral):
        raise TypeError(
            f"num_layers must be a positive integer, got {type(num_layers).__name__}"
        )
    parsed = int(num_layers)
    if parsed <= 0:
        raise ValueError(f"num_layers must be positive, got {parsed}")
    return parsed


def parse_layer_ids(
    value: Any,
    *,
    field_name: str = "layer_ids",
    num_layers: Optional[int] = None,
) -> Tuple[int, ...]:
    """Parse integer layer IDs and inclusive ranges without dropping bad input."""
    if num_layers is not None:
        num_layers = _YX_validate_num_layers(num_layers)
    if value is None:
        return ()

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        raw_tokens = re.split(r"[,;]", text)
        if any(not token.strip() for token in raw_tokens):
            raise ValueError(f"{field_name} contains an empty token: {value!r}")
        tokens: List[Any] = raw_tokens
    else:
        if isinstance(value, bool):
            raise TypeError(f"{field_name} must contain integer layer IDs, got bool")
        try:
            tokens = list(value)
        except TypeError:
            tokens = [value]

    parsed: List[int] = []
    for raw_token in tokens:
        if isinstance(raw_token, bool):
            raise TypeError(f"{field_name} contains invalid bool token {raw_token!r}")
        if isinstance(raw_token, numbers.Integral):
            token_ids = (int(raw_token),)
        elif isinstance(raw_token, str):
            token = raw_token.strip()
            integer_match = re.fullmatch(r"[0-9]+", token)
            range_match = re.fullmatch(r"([0-9]+)\s*-\s*([0-9]+)", token)
            if integer_match is not None:
                token_ids = (int(token),)
            elif range_match is not None:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                if end < start:
                    raise ValueError(
                        f"{field_name} range must be ascending, got {token!r}"
                    )
                token_ids = tuple(range(start, end + 1))
            else:
                raise ValueError(f"{field_name} contains invalid token {raw_token!r}")
        else:
            raise TypeError(
                f"{field_name} contains non-integer token {raw_token!r} "
                f"({type(raw_token).__name__})"
            )

        for layer_id in token_ids:
            if layer_id < 0:
                raise ValueError(f"{field_name} layer ID must be non-negative, got {layer_id}")
            if num_layers is not None and layer_id >= num_layers:
                raise ValueError(
                    f"{field_name} layer ID {layer_id} is out of range for "
                    f"num_layers={num_layers}"
                )
            parsed.append(layer_id)

    return tuple(sorted(set(parsed)))


def set_layer_recall_context(**YX_context: Any) -> None:
    _YX_THREAD_STATE.layer_recall_context = YX_context


def get_layer_recall_context() -> Dict[str, Any]:
    return dict(getattr(_YX_THREAD_STATE, "layer_recall_context", {}) or {})


def clear_layer_recall_context() -> None:
    _YX_THREAD_STATE.layer_recall_context = {}


def get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if getter is not None:
        try:
            return getter(key, default)
        except Exception:
            pass
    return getattr(config, key, default)


def infer_frame_seq_length(
    *,
    YX_frame_seq_length: Optional[int] = None,
    YX_image_or_video_shape: Optional[Sequence[int]] = None,
    YX_grid_size: Optional[Sequence[int]] = None,
    YX_patch_size: Sequence[int] = (1, 2, 2),
    YX_require: bool = False,
) -> Optional[int]:
    """Infer tokens per latent frame without relying on LongLive v1 constants."""
    if YX_frame_seq_length is not None and int(YX_frame_seq_length) > 0:
        return int(YX_frame_seq_length)

    if YX_grid_size is not None:
        grid = [int(item) for item in YX_grid_size]
        if len(grid) >= 3:
            return int(grid[-2] * grid[-1])
        if len(grid) == 2:
            return int(grid[0] * grid[1])

    if YX_image_or_video_shape is not None:
        shape = [int(item) for item in YX_image_or_video_shape]
        if len(shape) < 2:
            raise ValueError(f"Expected image_or_video_shape with H/W, got {shape}")
        latent_h, latent_w = int(shape[-2]), int(shape[-1])
        patch = _YX_as_int_tuple(YX_patch_size, _YX_WAN22_5B_DEFAULTS["patch_size"])
        patch_h = int(patch[-2]) if len(patch) >= 2 else 1
        patch_w = int(patch[-1]) if len(patch) >= 1 else 1
        if latent_h % patch_h != 0 or latent_w % patch_w != 0:
            raise ValueError(
                f"Latent H/W {(latent_h, latent_w)} must be divisible by patch H/W {(patch_h, patch_w)}"
            )
        return int((latent_h // patch_h) * (latent_w // patch_w))

    if YX_require:
        raise ValueError("Unable to infer frame_seq_length; pass runtime shape or explicit value")
    return None


def infer_chunk_token_size(
    *,
    YX_chunk_token_size: Optional[int] = None,
    YX_num_frames: Optional[int] = None,
    YX_frame_seq_length: Optional[int] = None,
    YX_kv_cache: Optional[Dict[str, Any]] = None,
    YX_require: bool = False,
) -> Optional[int]:
    if YX_chunk_token_size is not None and int(YX_chunk_token_size) > 0:
        return int(YX_chunk_token_size)
    if YX_kv_cache is not None and "block_token_size" in YX_kv_cache:
        return int(YX_kv_cache["block_token_size"])
    if YX_num_frames is not None and YX_frame_seq_length is not None:
        return int(YX_num_frames) * int(YX_frame_seq_length)
    if YX_require:
        raise ValueError("Unable to infer chunk token size")
    return None


def get_kv_cache_capacity_tokens(YX_kv_cache: Dict[str, Any]) -> int:
    return int(YX_kv_cache["k"].shape[1])


@dataclass
class LayerRecallRuntimeSizes:
    YX_frame_seq_length: Optional[int] = None
    YX_chunk_token_size: Optional[int] = None
    YX_max_attention_tokens: Optional[int] = None
    YX_kv_cache_tokens: Optional[int] = None


@dataclass
class LayerRecallConfig:
    layer_recall_enabled: bool = False
    layer_recall_normalize_scores: bool = True
    layer_recall_selection_mode: str = "hard"
    layer_recall_temperature: float = 1.0
    layer_recall_candidate_pool_size: int = 8
    layer_recall_log_path: str = ""
    layer_recall_log_selection_detail: bool = False
    layer_recall_selection_sample_interval: int = 0
    layer_recall_rank: int = 0
    layer_recall_debug_train: bool = False
    layer_recall_debug_interval: int = 10
    layer_recall_debug_grad_detail: bool = False
    layer_recall_debug_stage: bool = False
    layer_recall_debug_memory: bool = False
    layer_recall_debug_stage_interval: int = 1
    layer_recall_model_name: str = "Wan2.2-TI2V-5B"
    layer_recall_num_heads: int = 24
    layer_recall_head_dim: int = 128
    layer_recall_num_layers: int = 30
    layer_recall_patch_size: Tuple[int, int, int] = (1, 2, 2)
    layer_recall_frame_seq_length: int = 0
    layer_recall_chunk_token_size: int = 0
    layer_recall_require_equal_chunk_tokens: bool = True
    layer_recall_current_conditioned_enabled: bool = False
    layer_recall_current_hidden_dim: int = 512
    layer_recall_current_alpha: float = 0.1
    layer_recall_current_detach_summary: bool = True
    layer_recall_current_zero_init: bool = True
    layer_recall_use_layer_gamma: bool = True
    layer_recall_score_mode: str = "cosine"
    layer_recall_physical_cache_frames: int = 0
    memory_sensitive_layers: Tuple[int, ...] = (4, 9, 10, 12, 13, 15, 16, 17, 18, 26)

    def __post_init__(self) -> None:
        num_layers = _YX_validate_num_layers(self.layer_recall_num_layers)
        self.memory_sensitive_layers = parse_layer_ids(
            self.memory_sensitive_layers,
            field_name="memory_sensitive_layers",
            num_layers=num_layers,
        )
        self.validate(num_layers)

    def validate(self, num_layers: Optional[int] = None) -> None:
        if num_layers is None:
            num_layers = self.layer_recall_num_layers
        num_layers = _YX_validate_num_layers(num_layers)
        if self.layer_recall_selection_mode not in {
            "hard",
            "soft",
            "straight_through_topk",
        }:
            raise ValueError(
                "layer_recall_selection_mode must be hard, soft, or "
                f"straight_through_topk, got {self.layer_recall_selection_mode!r}"
            )
        if self.layer_recall_score_mode != "cosine":
            raise ValueError(
                "LayerRecall currently supports cosine scoring only, got "
                f"{self.layer_recall_score_mode!r}"
            )
        parse_layer_ids(
            self.memory_sensitive_layers,
            field_name="memory_sensitive_layers",
            num_layers=num_layers,
        )

    @classmethod
    def from_repo_config(cls, repo_config: Any, YX_rank: int = 0) -> "LayerRecallConfig":
        raw = get_config_value(repo_config, "layer_recall", None)
        if raw is None:
            raw = repo_config if repo_config is not None else {}

        model_kwargs = get_config_value(repo_config, "model_kwargs", None)
        model_name = str(
            get_config_value(
                raw,
                "layer_recall_model_name",
                get_config_value(model_kwargs, "model_name", _YX_WAN22_5B_DEFAULTS["model_name"]),
            )
        )
        defaults = _YX_WAN22_5B_DEFAULTS if model_name == _YX_WAN22_5B_DEFAULTS["model_name"] else {}
        patch_size = _YX_as_int_tuple(
            get_config_value(raw, "layer_recall_patch_size", defaults.get("patch_size", (1, 2, 2))),
            _YX_WAN22_5B_DEFAULTS["patch_size"],
        )
        if len(patch_size) == 2:
            patch_size = (1, int(patch_size[0]), int(patch_size[1]))
        num_layers = int(get_config_value(raw, "layer_recall_num_layers", defaults.get("num_layers", 30)))
        return cls(
            layer_recall_enabled=bool(get_config_value(raw, "layer_recall_enabled", False)),
            layer_recall_normalize_scores=bool(get_config_value(
                raw,
                "layer_recall_normalize_scores",
                True,
            )),
            layer_recall_selection_mode=str(get_config_value(raw, "layer_recall_selection_mode", "hard")).lower(),
            layer_recall_temperature=max(1e-6, float(get_config_value(raw, "layer_recall_temperature", 1.0))),
            layer_recall_candidate_pool_size=int(get_config_value(raw, "layer_recall_candidate_pool_size", 8)),
            layer_recall_log_path=str(get_config_value(raw, "layer_recall_log_path", "")),
            layer_recall_log_selection_detail=bool(get_config_value(raw, "layer_recall_log_selection_detail", False)),
            layer_recall_selection_sample_interval=int(get_config_value(raw, "layer_recall_selection_sample_interval", 0)),
            layer_recall_rank=YX_rank,
            layer_recall_debug_train=bool(get_config_value(raw, "layer_recall_debug_train", False)),
            layer_recall_debug_interval=max(1, int(get_config_value(raw, "layer_recall_debug_interval", 10))),
            layer_recall_debug_grad_detail=bool(get_config_value(raw, "layer_recall_debug_grad_detail", False)),
            layer_recall_debug_stage=bool(get_config_value(raw, "layer_recall_debug_stage", False)),
            layer_recall_debug_memory=bool(get_config_value(raw, "layer_recall_debug_memory", False)),
            layer_recall_debug_stage_interval=max(1, int(get_config_value(raw, "layer_recall_debug_stage_interval", 1))),
            layer_recall_model_name=model_name,
            layer_recall_num_heads=int(get_config_value(raw, "layer_recall_num_heads", defaults.get("num_heads", 24))),
            layer_recall_head_dim=int(get_config_value(raw, "layer_recall_head_dim", defaults.get("head_dim", 128))),
            layer_recall_num_layers=int(num_layers),
            layer_recall_patch_size=(int(patch_size[0]), int(patch_size[1]), int(patch_size[2])),
            layer_recall_frame_seq_length=int(get_config_value(raw, "layer_recall_frame_seq_length", 0) or 0),
            layer_recall_chunk_token_size=int(get_config_value(raw, "layer_recall_chunk_token_size", 0) or 0),
            layer_recall_require_equal_chunk_tokens=bool(get_config_value(raw, "layer_recall_require_equal_chunk_tokens", True)),
            layer_recall_current_conditioned_enabled=bool(get_config_value(raw, "layer_recall_current_conditioned_enabled", False)),
            layer_recall_current_hidden_dim=int(get_config_value(raw, "layer_recall_current_hidden_dim", 512)),
            layer_recall_current_alpha=float(get_config_value(raw, "layer_recall_current_alpha", 0.1)),
            layer_recall_current_detach_summary=bool(get_config_value(raw, "layer_recall_current_detach_summary", True)),
            layer_recall_current_zero_init=bool(get_config_value(raw, "layer_recall_current_zero_init", True)),
            layer_recall_use_layer_gamma=bool(get_config_value(raw, "layer_recall_use_layer_gamma", True)),
            layer_recall_score_mode=str(get_config_value(raw, "layer_recall_score_mode", "cosine")).lower(),
            layer_recall_physical_cache_frames=int(get_config_value(raw, "layer_recall_physical_cache_frames", 0) or 0),
            memory_sensitive_layers=parse_layer_ids(
                get_config_value(
                    raw,
                    "memory_sensitive_layers",
                    (4, 9, 10, 12, 13, 15, 16, 17, 18, 26),
                ),
                field_name="memory_sensitive_layers",
                num_layers=num_layers,
            ),
        )

    def resolve_runtime_sizes(
        self,
        *,
        YX_image_or_video_shape: Optional[Sequence[int]] = None,
        YX_grid_size: Optional[Sequence[int]] = None,
        YX_num_frames: Optional[int] = None,
        YX_local_attn_size: Optional[int] = None,
        YX_kv_cache: Optional[Dict[str, Any]] = None,
    ) -> LayerRecallRuntimeSizes:
        frame_seq_length = infer_frame_seq_length(
            YX_frame_seq_length=self.layer_recall_frame_seq_length,
            YX_image_or_video_shape=YX_image_or_video_shape,
            YX_grid_size=YX_grid_size,
            YX_patch_size=self.layer_recall_patch_size,
            YX_require=False,
        )
        chunk_token_size = infer_chunk_token_size(
            YX_chunk_token_size=self.layer_recall_chunk_token_size,
            YX_num_frames=YX_num_frames,
            YX_frame_seq_length=frame_seq_length,
            YX_kv_cache=YX_kv_cache,
            YX_require=False,
        )
        max_attention_tokens = None
        if YX_local_attn_size is not None and int(YX_local_attn_size) != -1 and frame_seq_length is not None:
            max_attention_tokens = int(YX_local_attn_size) * int(frame_seq_length)
        kv_cache_tokens = get_kv_cache_capacity_tokens(YX_kv_cache) if YX_kv_cache is not None else None
        return LayerRecallRuntimeSizes(
            YX_frame_seq_length=frame_seq_length,
            YX_chunk_token_size=chunk_token_size,
            YX_max_attention_tokens=max_attention_tokens,
            YX_kv_cache_tokens=kv_cache_tokens,
        )


@dataclass
class HistoryChunkRecord:
    chunk_index: int
    start_frame: int
    num_frames: int
    cache_start_token: int
    cache_end_token: int
    summary: torch.Tensor
    global_start_token: Optional[int] = None
    global_end_token: Optional[int] = None

    @property
    def token_range(self) -> Tuple[int, int]:
        return int(self.cache_start_token), int(self.cache_end_token)

    @property
    def global_token_range(self) -> Tuple[int, int]:
        start = self.global_start_token if self.global_start_token is not None else self.cache_start_token
        end = self.global_end_token if self.global_end_token is not None else self.cache_end_token
        return int(start), int(end)

    @property
    def num_tokens(self) -> int:
        return int(self.cache_end_token) - int(self.cache_start_token)


class _YXStableTopKResult(NamedTuple):
    values: torch.Tensor
    indices: torch.Tensor


def stable_topk_indices(
    scores: torch.Tensor,
    records: Sequence[HistoryChunkRecord],
    k: int,
    largest: bool = True,
) -> torch.Tensor:
    """Return score-ranked indices, preferring smaller chunk IDs on exact ties."""
    if scores.ndim != 1:
        raise ValueError(f"scores must be one-dimensional, got shape {tuple(scores.shape)}")
    if len(records) != int(scores.numel()):
        raise ValueError(
            f"records and scores must have the same length, got {len(records)} and "
            f"{scores.numel()}"
        )

    topk = min(max(int(k), 0), int(scores.numel()))
    if topk == 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)

    chunk_ids = torch.tensor(
        [int(record.chunk_index) for record in records],
        dtype=torch.long,
        device=scores.device,
    )
    chunk_order = torch.argsort(chunk_ids, stable=True)
    score_order = torch.argsort(
        scores.detach()[chunk_order],
        descending=bool(largest),
        stable=True,
    )
    return chunk_order[score_order[:topk]]


def stable_topk(
    scores: torch.Tensor,
    records: Sequence[HistoryChunkRecord],
    k: int,
    largest: bool = True,
) -> _YXStableTopKResult:
    indices = stable_topk_indices(scores, records, k, largest=largest)
    return _YXStableTopKResult(values=scores[indices], indices=indices)


class LayerRecallMemoryBank:
    def __init__(self) -> None:
        self.YX_records_by_layer: Dict[int, List[HistoryChunkRecord]] = {}

    def clear(self) -> None:
        self.YX_records_by_layer.clear()

    def records_for_layer(self, YX_layer_index: int) -> List[HistoryChunkRecord]:
        return list(self.YX_records_by_layer.get(int(YX_layer_index), []))

    def add_or_replace(self, YX_layer_index: int, YX_record: HistoryChunkRecord) -> None:
        records = self.YX_records_by_layer.setdefault(int(YX_layer_index), [])
        for idx, old in enumerate(records):
            if int(old.chunk_index) == int(YX_record.chunk_index):
                records[idx] = YX_record
                break
        else:
            records.append(YX_record)
        records.sort(key=lambda item: item.chunk_index)

    def _eligible_records(
        self,
        *,
        YX_layer_index: int,
        YX_current_start_token: int,
        YX_use_global_tokens: bool = True,
    ) -> List[HistoryChunkRecord]:
        records = []
        for record in self.YX_records_by_layer.get(int(YX_layer_index), []):
            _, end = record.global_token_range if YX_use_global_tokens else record.token_range
            if int(end) <= int(YX_current_start_token):
                records.append(record)
        return records

    def score_all(
        self,
        *,
        YX_layer_index: int,
        YX_query: torch.Tensor,
        YX_current_start_token: int,
        YX_normalize: bool = False,
        YX_use_global_tokens: bool = True,
    ) -> Tuple[List[HistoryChunkRecord], torch.Tensor]:
        records = self._eligible_records(
            YX_layer_index=YX_layer_index,
            YX_current_start_token=YX_current_start_token,
            YX_use_global_tokens=YX_use_global_tokens,
        )
        if len(records) == 0:
            return [], YX_query.new_empty((0,), dtype=torch.float32)

        summaries = torch.stack(
            [
                record.summary.detach().to(device=YX_query.device, dtype=torch.float32)
                for record in records
            ],
            dim=0,
        )
        query = YX_query.to(dtype=torch.float32)
        if YX_normalize:
            summaries = F.normalize(summaries, dim=-1)
            query = F.normalize(query, dim=-1)
        scores = torch.matmul(summaries, query)
        return records, scores

    def select(
        self,
        *,
        YX_layer_index: int,
        YX_query: torch.Tensor,
        YX_topk: int,
        YX_current_start_token: int,
        YX_normalize: bool = False,
        YX_use_global_tokens: bool = True,
    ) -> Tuple[List[HistoryChunkRecord], torch.Tensor]:
        records, scores = self.score_all(
            YX_layer_index=YX_layer_index,
            YX_query=YX_query,
            YX_current_start_token=YX_current_start_token,
            YX_normalize=YX_normalize,
            YX_use_global_tokens=YX_use_global_tokens,
        )
        if int(YX_topk) <= 0 or len(records) == 0:
            return [], scores[:0]
        topk = min(int(YX_topk), int(scores.numel()))
        selected_scores, selected_indices = stable_topk(
            scores,
            records,
            topk,
        )
        selected = [records[int(idx)] for idx in selected_indices.detach().cpu().tolist()]
        return selected, selected_scores

    def apply_cache_roll(
        self,
        *,
        YX_layer_index: int,
        YX_sink_tokens: int,
        YX_num_evicted_tokens: int,
    ) -> None:
        if int(YX_num_evicted_tokens) <= 0:
            return
        next_records: List[HistoryChunkRecord] = []
        sink_tokens = int(YX_sink_tokens)
        evict_end = sink_tokens + int(YX_num_evicted_tokens)
        for record in self.YX_records_by_layer.get(int(YX_layer_index), []):
            start, end = record.token_range
            if end <= sink_tokens:
                next_records.append(record)
            elif start >= evict_end:
                next_records.append(replace(
                    record,
                    cache_start_token=start - int(YX_num_evicted_tokens),
                    cache_end_token=end - int(YX_num_evicted_tokens),
                ))
        self.YX_records_by_layer[int(YX_layer_index)] = next_records

    def prune_by_cache_capacity(self, *, YX_layer_index: int, YX_cache_tokens: int) -> None:
        cache_tokens = int(YX_cache_tokens)
        self.YX_records_by_layer[int(YX_layer_index)] = [
            record for record in self.YX_records_by_layer.get(int(YX_layer_index), [])
            if int(record.cache_start_token) >= 0 and int(record.cache_end_token) <= cache_tokens
        ]


def pool_pre_rope_k(YX_k_pre_rope: torch.Tensor) -> torch.Tensor:
    if YX_k_pre_rope.ndim != 4:
        raise ValueError(f"Expected k_pre_rope shape [B, T, H, D], got {tuple(YX_k_pre_rope.shape)}")
    return YX_k_pre_rope.detach().float().mean(dim=(0, 1, 2))


def straight_through_hard_value(
    YX_hard_value: torch.Tensor,
    YX_soft_value: torch.Tensor,
) -> torch.Tensor:
    """Use the hard value in forward while routing gradients through soft value."""
    if YX_hard_value.shape != YX_soft_value.shape:
        raise ValueError(
            "YX straight-through hard/soft shapes must match, got "
            f"{tuple(YX_hard_value.shape)} and {tuple(YX_soft_value.shape)}"
        )
    # Subtract first. In BF16, ``hard + soft - soft.detach()`` perturbs the
    # forward value because the first addition rounds before the subtraction.
    return YX_hard_value + (YX_soft_value - YX_soft_value.detach())


def materialize_layer_recall_slot(
    hard_value: torch.Tensor,
    soft_value: torch.Tensor,
    selection_mode: str,
) -> torch.Tensor:
    """Materialize one fixed-budget history slot for inference or training."""
    mode = str(selection_mode).strip().lower()
    if mode == "hard":
        return hard_value
    if mode == "soft":
        return soft_value
    if mode == "straight_through_topk":
        return straight_through_hard_value(hard_value, soft_value)
    raise ValueError(f"Unsupported LayerRecall selection mode: {selection_mode!r}")


def validate_query_shape(YX_query: torch.Tensor, YX_config: LayerRecallConfig) -> None:
    if int(YX_query.numel()) != int(YX_config.layer_recall_head_dim):
        raise ValueError(
            f"LayerRecall query has {YX_query.numel()} values, expected head_dim={YX_config.layer_recall_head_dim}"
        )


def is_layer_recall_enabled_for_layer(
    YX_config: Optional[LayerRecallConfig],
    YX_layer_index: int,
    num_layers: Optional[int] = None,
) -> Tuple[bool, str]:
    if isinstance(YX_layer_index, bool) or not isinstance(YX_layer_index, numbers.Integral):
        raise TypeError(
            "YX_layer_index must be an integer, "
            f"got {type(YX_layer_index).__name__}"
        )
    layer_index = int(YX_layer_index)
    effective_num_layers = num_layers
    if effective_num_layers is None and YX_config is not None:
        effective_num_layers = getattr(YX_config, "layer_recall_num_layers", None)
    if effective_num_layers is not None:
        effective_num_layers = _YX_validate_num_layers(effective_num_layers)
        if layer_index < 0 or layer_index >= effective_num_layers:
            raise ValueError(
                f"YX_layer_index {layer_index} is out of range for "
                f"num_layers={effective_num_layers}"
            )
    elif layer_index < 0:
        raise ValueError(f"YX_layer_index must be non-negative, got {layer_index}")

    if YX_config is None:
        return False, "layer_recall_not_configured"
    YX_config.validate(effective_num_layers)
    if layer_index in set(YX_config.memory_sensitive_layers):
        return True, "memory_sensitive_layer"
    return False, "original_window_layer"


def should_use_layer_recall_selection(
    *,
    YX_local_attn_size: Any,
    YX_current_start_token: int,
    YX_max_attention_size: int,
) -> Tuple[bool, str]:
    try:
        local_attn_value = int(YX_local_attn_size)
    except TypeError:
        local_values = [int(v) for v in YX_local_attn_size if int(v) != -1]
        local_attn_value = max(local_values) if len(local_values) > 0 else -1
    if local_attn_value == -1:
        return False, "global_attention_local_attn_size_-1"
    if int(YX_current_start_token) <= int(YX_max_attention_size):
        return False, "history_within_original_local_window"
    return True, "history_exceeds_original_local_window"


def _YX_merge_ranges(YX_ranges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    clean = sorted((int(start), int(end)) for start, end in YX_ranges if int(end) > int(start))
    merged: List[Tuple[int, int]] = []
    for start, end in clean:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _YX_range_len(YX_ranges: Sequence[Tuple[int, int]]) -> int:
    return sum(int(end) - int(start) for start, end in YX_ranges)


def _YX_subtract_ranges(
    YX_ranges: Sequence[Tuple[int, int]],
    YX_blocked: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    result = list(_YX_merge_ranges(YX_ranges))
    for block_start, block_end in _YX_merge_ranges(YX_blocked):
        next_result: List[Tuple[int, int]] = []
        for start, end in result:
            if block_end <= start or block_start >= end:
                next_result.append((start, end))
                continue
            if start < block_start:
                next_result.append((start, block_start))
            if block_end < end:
                next_result.append((block_end, end))
        result = next_result
    return result



def assemble_slot_visible_plan(
    *,
    YX_selected_records: Sequence[HistoryChunkRecord],
    YX_sink_tokens: int,
    YX_current_start_token: int,
    YX_current_end_token: int,
    YX_max_attention_size: int,
    YX_chunk_token_size: int,
) -> Dict[str, Any]:
    """Fill the attention-visible budget with sink, selected history, and current."""
    max_tokens = int(YX_max_attention_size)
    chunk_tokens = int(YX_chunk_token_size)
    if chunk_tokens <= 0:
        raise ValueError(f"Expected positive chunk_token_size, got {chunk_tokens}")

    current_range = (int(YX_current_start_token), int(YX_current_end_token))
    current_tokens = current_range[1] - current_range[0]
    sink_ranges = [(0, int(YX_sink_tokens))] if int(YX_sink_tokens) > 0 else []
    forced_ranges = _YX_merge_ranges(sink_ranges + [current_range])
    forced_tokens = _YX_range_len(forced_ranges)
    if forced_tokens > max_tokens:
        raise RuntimeError(
            "YX LayerRecall slot forced tokens exceed max_attention_size: "
            f"forced={forced_tokens}, max={max_tokens}, "
            f"sink={YX_sink_tokens}, current={current_tokens}"
        )

    slot_budget_tokens = max(0, max_tokens - forced_tokens)
    requested_slots = slot_budget_tokens // chunk_tokens

    selected_ranges: List[Tuple[int, int]] = []
    selected_ids: List[int] = []
    selected_records: List[HistoryChunkRecord] = []
    for record in list(YX_selected_records):
        if len(selected_records) >= requested_slots:
            break
        start, end = record.token_range
        if int(end) - int(start) != chunk_tokens:
            continue
        if _YX_range_len(_YX_subtract_ranges([record.token_range], forced_ranges + selected_ranges)) != chunk_tokens:
            continue
        selected_records.append(record)
        selected_ranges.append(record.token_range)
        selected_ids.append(int(record.chunk_index))

    filled_slots = len(selected_records)
    unfilled_slots = max(0, requested_slots - filled_slots)
    underfill_tokens = unfilled_slots * chunk_tokens
    visible_tokens = forced_tokens + filled_slots * chunk_tokens
    if visible_tokens > max_tokens:
        raise RuntimeError(f"YX LayerRecall slot visible tokens {visible_tokens} exceed max_attention_size {max_tokens}")

    return {
        "YX_visible_layout": "sink_selected_current",
        "YX_visible_ranges": _YX_merge_ranges(sink_ranges + selected_ranges + [current_range]),
        "YX_sink_ranges": _YX_merge_ranges(sink_ranges),
        "YX_selected_ranges": list(selected_ranges),
        "YX_recent_ranges": [],
        "YX_current_range": current_range,
        "YX_selected_chunk_ids": selected_ids,
        "YX_selected_records": selected_records,
        "YX_visible_tokens": visible_tokens,
        "YX_sink_tokens": _YX_range_len(sink_ranges),
        "YX_selected_tokens": filled_slots * chunk_tokens,
        "YX_recent_tokens": 0,
        "YX_current_tokens": current_tokens,
        "YX_forced_tokens": forced_tokens,
        "YX_memory_slot_budget_tokens": slot_budget_tokens,
        "YX_memory_slots_requested": requested_slots,
        "YX_memory_slots_filled": filled_slots,
        "YX_memory_slots_unfilled": unfilled_slots,
        "YX_visible_underfill_tokens": underfill_tokens,
    }



class LayerRecallSelectionLogger:
    def __init__(self, YX_config: LayerRecallConfig) -> None:
        self.YX_config = YX_config
        self.YX_detail_enabled = bool(
            YX_config.layer_recall_enabled
            and YX_config.layer_recall_log_selection_detail
            and YX_config.layer_recall_log_path
        )
        self.YX_enabled = self.YX_detail_enabled
        self.reset_counters()
        if self.YX_detail_enabled and int(YX_config.layer_recall_rank) == 0:
            os.makedirs(os.path.dirname(YX_config.layer_recall_log_path) or ".", exist_ok=True)

    def reset_counters(self) -> None:
        self.YX_counters = {
            "YX_log_events": 0,
            "YX_context_update_events": 0,
            "YX_gate_active_events": 0,
            "YX_soft_or_st_events": 0,
            "YX_soft_memory_positive_events": 0,
            "YX_soft_memory_tokens_total": 0,
            "YX_candidate_chunks_total": 0,
            "YX_candidate_chunks_max": 0,
            "YX_selection_metric_events": 0,
            "YX_selection_entropy_sum": 0.0,
            "YX_selection_entropy_max": 0.0,
            "YX_selection_entropy_min": None,
            "YX_top1_weight_sum": 0.0,
            "YX_top1_margin_sum": 0.0,
            "YX_score_std_sum": 0.0,
            "YX_selected_recent_count": 0,
            "memory_sensitive_layer_events": 0,
            "original_window_layer_events": 0,
            "layer_recall_layout_applied_events": 0,
        }
        self.YX_selection_modes: Dict[str, int] = {}
        self.YX_gate_reasons: Dict[str, int] = {}
        self.YX_selected_chunk_hist: Dict[str, int] = {}
        self.YX_layer_selection_stats: Dict[str, Dict[str, Any]] = {}

    def snapshot_counters(self) -> Dict[str, Any]:
        snapshot = dict(self.YX_counters)
        if snapshot.get("YX_selection_entropy_min") is None:
            snapshot["YX_selection_entropy_min"] = 0.0
        snapshot["YX_selection_modes"] = dict(self.YX_selection_modes)
        snapshot["YX_gate_reasons"] = dict(self.YX_gate_reasons)
        snapshot["YX_selected_chunk_hist"] = dict(self.YX_selected_chunk_hist)
        snapshot["YX_selected_chunk_unique"] = len(self.YX_selected_chunk_hist)
        snapshot["YX_layer_selection_stats"] = {
            layer: {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in stats.items()
            }
            for layer, stats in self.YX_layer_selection_stats.items()
        }
        return snapshot

    @staticmethod
    def _to_float_tensor(value: Any) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value.detach().float().cpu().flatten()
        else:
            try:
                tensor = torch.tensor(value, dtype=torch.float32).flatten()
            except Exception:
                return None
        if tensor.numel() == 0:
            return None
        return tensor

    def _update_selection_diagnostics(self, YX_payload: Dict[str, Any]) -> None:
        weights = self._to_float_tensor(YX_payload.get("YX_candidate_weights"))
        scores = self._to_float_tensor(YX_payload.get("YX_candidate_scores"))
        candidate_ids = YX_payload.get("YX_candidate_chunk_ids", [])
        if weights is None or scores is None:
            return
        count = min(int(weights.numel()), int(scores.numel()))
        if count <= 0:
            return
        weights = weights[:count]
        scores = scores[:count]
        if not torch.isfinite(weights).all() or not torch.isfinite(scores).all():
            return
        weight_sum = float(weights.sum().item())
        if weight_sum <= 0:
            return
        probs = (weights / weight_sum).clamp_min(1e-12)
        entropy = float(-(probs * probs.log()).sum().item())
        sorted_weights, sorted_indices = torch.sort(probs, descending=True)
        top1_weight = float(sorted_weights[0].item())
        top2_weight = float(sorted_weights[1].item()) if count > 1 else 0.0
        top1_margin = top1_weight - top2_weight
        score_std = float(torch.std(scores, unbiased=False).item()) if count > 1 else 0.0
        top1_pos = int(sorted_indices[0].item())
        try:
            selected_chunk = int(candidate_ids[top1_pos])
        except Exception:
            selected_chunk = -1
        try:
            numeric_ids = [int(chunk_id) for chunk_id in candidate_ids[:count]]
        except Exception:
            numeric_ids = []
        selected_recent = bool(numeric_ids and selected_chunk == max(numeric_ids))

        self.YX_counters["YX_selection_metric_events"] += 1
        self.YX_counters["YX_selection_entropy_sum"] += entropy
        self.YX_counters["YX_selection_entropy_max"] = max(float(self.YX_counters["YX_selection_entropy_max"]), entropy)
        current_min = self.YX_counters.get("YX_selection_entropy_min")
        self.YX_counters["YX_selection_entropy_min"] = entropy if current_min is None else min(float(current_min), entropy)
        self.YX_counters["YX_top1_weight_sum"] += top1_weight
        self.YX_counters["YX_top1_margin_sum"] += top1_margin
        self.YX_counters["YX_score_std_sum"] += score_std
        if selected_recent:
            self.YX_counters["YX_selected_recent_count"] += 1
        if selected_chunk >= 0:
            key = str(selected_chunk)
            self.YX_selected_chunk_hist[key] = self.YX_selected_chunk_hist.get(key, 0) + 1

        layer_key = str(int(YX_payload.get("YX_layer_index", -1)))
        layer_stats = self.YX_layer_selection_stats.setdefault(
            layer_key,
            {
                "events": 0,
                "entropy_sum": 0.0,
                "top1_weight_sum": 0.0,
                "top1_margin_sum": 0.0,
                "score_std_sum": 0.0,
                "selected_recent_count": 0,
                "top1_chunk_hist": {},
            },
        )
        layer_stats["events"] += 1
        layer_stats["entropy_sum"] += entropy
        layer_stats["top1_weight_sum"] += top1_weight
        layer_stats["top1_margin_sum"] += top1_margin
        layer_stats["score_std_sum"] += score_std
        if selected_recent:
            layer_stats["selected_recent_count"] += 1
        if selected_chunk >= 0:
            hist = layer_stats["top1_chunk_hist"]
            key = str(selected_chunk)
            hist[key] = hist.get(key, 0) + 1

    def _update_counters(self, YX_payload: Dict[str, Any]) -> None:
        if int(self.YX_config.layer_recall_rank) != 0:
            return
        self.YX_counters["YX_log_events"] += 1
        if str(YX_payload.get("YX_call_type", "")) == "context_update":
            self.YX_counters["YX_context_update_events"] += 1
        if bool(YX_payload.get("layer_recall_gate_active", False)):
            self.YX_counters["YX_gate_active_events"] += 1
        if bool(YX_payload.get("layer_recall_layout_applied", False)):
            self.YX_counters["layer_recall_layout_applied_events"] += 1

        mode = str(YX_payload.get("layer_recall_selection_mode", "unknown")).lower()
        self.YX_selection_modes[mode] = self.YX_selection_modes.get(mode, 0) + 1
        if bool(YX_payload.get("layer_recall_gate_active", False)) and mode in {
            "soft",
            "straight_through_topk",
        }:
            self.YX_counters["YX_soft_or_st_events"] += 1

        reason = str(YX_payload.get("layer_recall_gate_reason", "unknown"))
        self.YX_gate_reasons[reason] = self.YX_gate_reasons.get(reason, 0) + 1

        if bool(YX_payload.get("memory_sensitive_layer", False)):
            self.YX_counters["memory_sensitive_layer_events"] += 1
        else:
            self.YX_counters["original_window_layer_events"] += 1

        candidate_ids = YX_payload.get("YX_candidate_chunk_ids", [])
        try:
            candidate_count = len(candidate_ids)
        except TypeError:
            candidate_count = 0
        self.YX_counters["YX_candidate_chunks_total"] += int(candidate_count)
        self.YX_counters["YX_candidate_chunks_max"] = max(
            int(self.YX_counters["YX_candidate_chunks_max"]),
            int(candidate_count),
        )

        soft_tokens = int(YX_payload.get("YX_soft_memory_tokens", 0) or 0)
        self.YX_counters["YX_soft_memory_tokens_total"] += soft_tokens
        if soft_tokens > 0:
            self.YX_counters["YX_soft_memory_positive_events"] += 1
        self._update_selection_diagnostics(YX_payload)

    def log(self, YX_payload: Dict[str, Any]) -> None:
        if bool(getattr(self.YX_config, "layer_recall_enabled", False)):
            self._update_counters(YX_payload)
        if not self.YX_detail_enabled or int(self.YX_config.layer_recall_rank) != 0:
            return
        sample_interval = int(getattr(self.YX_config, "layer_recall_selection_sample_interval", 0) or 0)
        if sample_interval > 1 and int(self.YX_counters.get("YX_log_events", 0)) % sample_interval != 0:
            return
        with open(self.YX_config.layer_recall_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_YX_to_plain(YX_payload), sort_keys=True) + "\n")


__all__ = [
    "HistoryChunkRecord",
    "LayerRecallConfig",
    "LayerRecallSelectionLogger",
    "LayerRecallMemoryBank",
    "LayerRecallRuntimeSizes",
    "assemble_slot_visible_plan",
    "clear_layer_recall_context",
    "get_config_value",
    "get_kv_cache_capacity_tokens",
    "get_layer_recall_context",
    "infer_chunk_token_size",
    "infer_frame_seq_length",
    "is_layer_recall_enabled_for_layer",
    "parse_layer_ids",
    "pool_pre_rope_k",
    "set_layer_recall_context",
    "should_use_layer_recall_selection",
    "stable_topk",
    "stable_topk_indices",
    "straight_through_hard_value",
    "validate_query_shape",
    "materialize_layer_recall_slot",
]
