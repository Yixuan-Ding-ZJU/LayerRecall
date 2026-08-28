#!/usr/bin/env python3
"""Audit uninterrupted versus interrupted/resumed LayerRecall prediction training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    """Return no events when the corresponding detail logger was disabled."""
    if not path.is_file():
        return []
    return _load_jsonl(path)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_compare(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
            "max_abs": None,
        }
    equal = bool(torch.equal(left, right))
    max_abs = 0.0
    if left.numel() and (left.is_floating_point() or left.is_complex()):
        max_abs = float((left - right).abs().max().item())
    elif left.numel() and not equal:
        max_abs = float((left.to(torch.int64) - right.to(torch.int64)).abs().max().item())
    return {"equal": equal, "max_abs": max_abs}


def _recursive_differences(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return [{"path": path, "reason": "tensor_type_mismatch"}]
        result = _tensor_compare(left, right)
        return [] if result["equal"] else [{"path": path, **result}]
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
            return [{"path": path, "reason": "ndarray_type_mismatch"}]
        equal = (
            left.shape == right.shape
            and left.dtype == right.dtype
            and bool(np.array_equal(left, right))
        )
        if equal:
            return []
        max_abs = None
        if left.shape == right.shape and left.size and np.issubdtype(left.dtype, np.number):
            max_abs = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        return [{
            "path": path,
            "reason": "ndarray_mismatch",
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
            "max_abs": max_abs,
        }]
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left).union(right), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                differences.append({"path": child, "reason": "missing_key"})
            else:
                differences.extend(_recursive_differences(left[key], right[key], child))
        return differences
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [{"path": path, "reason": "length_mismatch", "left": len(left), "right": len(right)}]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(
                _recursive_differences(left_item, right_item, f"{path}[{index}]")
            )
        return differences
    if type(left) is not type(right) or left != right:
        return [{"path": path, "reason": "value_mismatch", "left": repr(left), "right": repr(right)}]
    return []


def _checkpoint(run: Path, step: int) -> tuple[Path, dict[str, Any]]:
    directory = run / f"checkpoint_model_{step:06d}"
    model_path = directory / "model.pt"
    marker_path = directory / "COMPLETE"
    if not model_path.is_file() or not marker_path.is_file():
        raise FileNotFoundError(f"incomplete checkpoint: {directory}")
    return model_path, torch.load(model_path, map_location="cpu", weights_only=False)


def _trace_rows(run: Path, *, after_step: int) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run.glob("rank*_exact_resume_trace.jsonl")):
        for row in _load_jsonl(path):
            if int(row["optimizer_step_after"]) <= after_step:
                continue
            normalized = dict(row)
            normalized.pop("time", None)
            normalized.pop("resume_checkpoint_path", None)
            rows.append(normalized)
    return sorted(rows, key=lambda row: (row["global_rank"], row["optimizer_step_after"]))


_METRIC_FIELDS = (
    "step",
    "chpm/loss",
    "chpm/loss_pred",
    "chpm/loss_reg",
    "chpm/grad_norm",
    "chpm/layer_recall_gate_active_events",
    "chpm/layer_recall_soft_or_st_events",
    "chpm/memory_sensitive_layer_events",
    "chpm/original_window_layer_events",
    "chpm/layer_recall_active_events",
    "raw/dataset_indices_csv",
    "raw/data_epoch",
    "raw/data_sample_cursor",
    "raw/global_micro_step_before_commit",
    "raw/micro_step_seed",
    "raw/pred_chunk_indices_csv",
)


def _metric_rows(run: Path, *, after_step: int) -> list[dict[str, Any]]:
    rows = []
    for row in _load_jsonl(run / "chpm_metrics.jsonl"):
        if int(row["step"]) > after_step:
            rows.append({key: row.get(key) for key in _METRIC_FIELDS})
    return rows


def _sample_rows(run: Path, *, after_step: int) -> list[dict[str, Any]]:
    ignored = {"time"}
    return [
        {key: value for key, value in row.items() if key not in ignored}
        for row in _load_jsonl(run / "chpm_samples.jsonl")
        if int(row["step"]) > after_step
    ]


def _selection_signature(run: Path, *, after_step: int) -> dict[str, Any]:
    rows = [
        row
        for row in _load_optional_jsonl(run / "layer_recall_selection.jsonl")
        if int(row.get("YX_step", -1)) > after_step
    ]
    return {"count": len(rows), "sha256": _canonical_hash(rows)}


_DISCRETE_SELECTION_FIELDS = (
    "YX_step",
    "YX_call_type",
    "YX_chunk_index",
    "YX_chunk_start_frame",
    "YX_denoising_step_index",
    "YX_layer_index",
    "YX_candidate_chunk_ids",
    "YX_selected_chunk_ids",
    "YX_hard_top_ids",
    "YX_score_top_ids",
    "YX_top1_chunk_id",
    "layer_recall_gate_active",
    "layer_recall_gate_reason",
    "memory_sensitive_layer",
    "layer_role",
    "original_window_layer_active",
    "layer_recall_layout_applied",
    "YX_effective_visible_layout",
    "YX_candidate_global_token_ranges",
    "YX_selected_ranges",
    "YX_memory_slots_requested",
    "YX_memory_slots_filled",
    "YX_memory_slots_unfilled",
    "YX_soft_memory_tokens",
    "YX_visible_tokens",
    "YX_cache_start_frame",
    "YX_local_start_index",
    "YX_local_end_index",
)


def _discrete_selection_signature(run: Path, *, after_step: int) -> dict[str, Any]:
    rows = []
    for row in _load_optional_jsonl(run / "layer_recall_selection.jsonl"):
        if int(row.get("YX_step", -1)) <= after_step:
            continue
        rows.append({key: row.get(key) for key in _DISCRETE_SELECTION_FIELDS})
    return {"count": len(rows), "sha256": _canonical_hash(rows)}


def _max_abs_from_differences(differences: list[dict[str, Any]]) -> float | None:
    values = [
        float(item["max_abs"])
        for item in differences
        if item.get("max_abs") is not None
    ]
    return max(values) if values else (0.0 if not differences else None)


def _all_tensor_differences_within(
    differences: list[dict[str, Any]], threshold: float
) -> bool:
    return all(
        item.get("max_abs") is not None
        and float(item["max_abs"]) <= threshold
        for item in differences
    )


def _metric_differences_within(
    differences: list[dict[str, Any]], threshold: float
) -> bool:
    for item in differences:
        if not str(item.get("path", "")).endswith(".chpm/grad_norm"):
            return False
        try:
            left = float(item["left"])
            right = float(item["right"])
        except (KeyError, TypeError, ValueError):
            return False
        if abs(left - right) > threshold:
            return False
    return True


def _checkpoint_summary(state: dict[str, Any]) -> dict[str, Any]:
    layer_recall_state = state["layer_recall_state_dict"]
    optimizer_state = state["student_optimizer"]["state"]
    return {
        "trainer": state.get("trainer"),
        "checkpoint_version": state.get("checkpoint_version"),
        "global_step": state.get("global_step"),
        "global_micro_step": state.get("global_micro_step"),
        "accumulation_step": state.get("accumulation_step"),
        "layer_recall_tensor_count": len(layer_recall_state),
        "layer_recall_scalar_count": sum(int(value.numel()) for value in layer_recall_state.values()),
        "optimizer_state_count": len(optimizer_state),
        "data_stream_rank_count": len(state.get("data_stream_states", [])),
        "rng_rank_count": len(state.get("rng_states", [])),
        "dataset_manifest_hash": state.get("dataset_manifest_hash"),
        "critical_resume_fingerprint": state.get("critical_resume_fingerprint"),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    control = Path(args.control_run).resolve()
    interrupted = Path(args.interrupted_run).resolve()
    resumed = Path(args.resumed_run).resolve()
    control_input_path, control_input = _checkpoint(control, args.resume_step)
    interrupted_input_path, interrupted_input = _checkpoint(interrupted, args.resume_step)
    control_output_path, control_output = _checkpoint(control, args.final_step)
    resumed_output_path, resumed_output = _checkpoint(resumed, args.final_step)

    control_selection_full = _selection_signature(
        control, after_step=args.resume_step
    )
    resumed_selection_full = _selection_signature(
        resumed, after_step=args.resume_step
    )
    control_selection_discrete = _discrete_selection_signature(
        control, after_step=args.resume_step
    )
    resumed_selection_discrete = _discrete_selection_signature(
        resumed, after_step=args.resume_step
    )
    comparisons = {
        "resume_input_layer_recall": _recursive_differences(
            control_input["layer_recall_state_dict"], interrupted_input["layer_recall_state_dict"]
        ),
        "resume_input_optimizer": _recursive_differences(
            control_input["student_optimizer"], interrupted_input["student_optimizer"]
        ),
        "final_layer_recall": _recursive_differences(
            control_output["layer_recall_state_dict"], resumed_output["layer_recall_state_dict"]
        ),
        "final_optimizer": _recursive_differences(
            control_output["student_optimizer"], resumed_output["student_optimizer"]
        ),
        "final_data_stream": _recursive_differences(
            control_output["data_stream_states"], resumed_output["data_stream_states"]
        ),
        "final_rng": _recursive_differences(
            control_output["rng_states"], resumed_output["rng_states"]
        ),
        "trace_suffix": _recursive_differences(
            _trace_rows(control, after_step=args.resume_step),
            _trace_rows(resumed, after_step=args.resume_step),
        ),
        "metric_suffix": _recursive_differences(
            _metric_rows(control, after_step=args.resume_step),
            _metric_rows(resumed, after_step=args.resume_step),
        ),
        "sample_suffix": _recursive_differences(
            _sample_rows(control, after_step=args.resume_step),
            _sample_rows(resumed, after_step=args.resume_step),
        ),
        "selection_discrete_suffix": _recursive_differences(
            control_selection_discrete,
            resumed_selection_discrete,
        ),
    }
    acceptance = {
        "resume_input_layer_recall_exact": not comparisons["resume_input_layer_recall"],
        "resume_input_optimizer_exact": not comparisons["resume_input_optimizer"],
        "final_layer_recall_within_tolerance": _all_tensor_differences_within(
            comparisons["final_layer_recall"], args.layer_recall_max_abs
        ),
        "final_optimizer_within_tolerance": _all_tensor_differences_within(
            comparisons["final_optimizer"], args.optimizer_max_abs
        ),
        "final_data_stream_exact": not comparisons["final_data_stream"],
        "final_rng_exact": not comparisons["final_rng"],
        "trace_suffix_exact": not comparisons["trace_suffix"],
        "metric_suffix_within_tolerance": _metric_differences_within(
            comparisons["metric_suffix"], args.metric_max_abs
        ),
        "sample_suffix_exact": not comparisons["sample_suffix"],
        "selection_discrete_suffix_exact": not comparisons[
            "selection_discrete_suffix"
        ],
    }
    failures = [name for name, passed in acceptance.items() if not passed]
    return {
        "status": "pass" if not failures else "fail",
        "resume_step": args.resume_step,
        "final_step": args.final_step,
        "paths": {
            "control_run": str(control),
            "interrupted_run": str(interrupted),
            "resumed_run": str(resumed),
            "control_input_checkpoint": str(control_input_path),
            "interrupted_input_checkpoint": str(interrupted_input_path),
            "control_output_checkpoint": str(control_output_path),
            "resumed_output_checkpoint": str(resumed_output_path),
        },
        "control_output": _checkpoint_summary(control_output),
        "resumed_output": _checkpoint_summary(resumed_output),
        "thresholds": {
            "layer_recall_max_abs": args.layer_recall_max_abs,
            "optimizer_max_abs": args.optimizer_max_abs,
            "metric_max_abs": args.metric_max_abs,
        },
        "observed": {
            "final_layer_recall_max_abs": _max_abs_from_differences(comparisons["final_layer_recall"]),
            "final_optimizer_max_abs": _max_abs_from_differences(
                comparisons["final_optimizer"]
            ),
            "full_selection_floating_fields_exact": (
                control_selection_full == resumed_selection_full
            ),
        },
        "acceptance": acceptance,
        "selection_full_control": control_selection_full,
        "selection_full_resumed": resumed_selection_full,
        "selection_discrete_control": control_selection_discrete,
        "selection_discrete_resumed": resumed_selection_discrete,
        "comparison_difference_counts": {
            name: len(differences) for name, differences in comparisons.items()
        },
        "comparison_first_differences": {
            name: differences[:10]
            for name, differences in comparisons.items()
            if differences
        },
        "failures": failures,
    }


def _write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "YX_exact_resume_audit.json"
    md_path = output_dir / "YX_exact_resume_audit.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# LayerRecall Prediction Distillation Exact Resume 审计",
        "",
        f"- 结果：`{report['status']}`",
        f"- 恢复点：step `{report['resume_step']}`",
        f"- 最终对照：step `{report['final_step']}`",
        f"- control：`{report['paths']['control_run']}`",
        f"- interrupted：`{report['paths']['interrupted_run']}`",
        f"- resumed：`{report['paths']['resumed_run']}`",
        "",
        "## 差异计数",
        "",
        "| 对象 | 差异数 |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{name}` | {count} |"
        for name, count in report["comparison_difference_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Checkpoint 摘要",
            "",
            "```json",
            json.dumps(report["resumed_output"], indent=2, ensure_ascii=False, sort_keys=True),
            "```",
        ]
    )
    if report["failures"]:
        lines.extend(
            [
                "",
                "## 失败项",
                "",
                ", ".join(f"`{item}`" for item in report["failures"]),
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-run", required=True)
    parser.add_argument("--interrupted-run", required=True)
    parser.add_argument("--resumed-run", required=True)
    parser.add_argument("--resume-step", type=int, default=2)
    parser.add_argument("--final-step", type=int, default=4)
    parser.add_argument("--layer_recall-max-abs", type=float, default=5.0e-6)
    parser.add_argument("--optimizer-max-abs", type=float, default=5.0e-6)
    parser.add_argument("--metric-max-abs", type=float, default=5.0e-6)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = audit(args)
    _write_report(report, Path(args.output_dir))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
