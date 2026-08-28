import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.layer_recall import (
    HistoryChunkRecord,
    LayerRecallMemoryBank,
    stable_topk,
    stable_topk_indices,
)


def _record(chunk_index: int, summary: torch.Tensor | None = None) -> HistoryChunkRecord:
    return HistoryChunkRecord(
        chunk_index=chunk_index,
        start_frame=chunk_index,
        num_frames=1,
        cache_start_token=chunk_index,
        cache_end_token=chunk_index + 1,
        summary=torch.zeros(2) if summary is None else summary,
    )


def _score(
    records: list[HistoryChunkRecord],
    query: torch.Tensor,
    *,
    normalize: bool = False,
) -> torch.Tensor:
    bank = LayerRecallMemoryBank()
    for record in records:
        bank.add_or_replace(0, record)
    _, scores = bank.score_all(
        YX_layer_index=0,
        YX_query=query,
        YX_current_start_token=100,
        YX_normalize=normalize,
    )
    return scores


def _selected_chunk_ids(
    indices: torch.Tensor,
    records: list[HistoryChunkRecord],
) -> list[int]:
    return [records[index].chunk_index for index in indices.detach().cpu().tolist()]


def test_score_all_uses_fp32_for_bfloat16_and_keeps_only_query_gradient() -> None:
    summary = torch.tensor([2.0, -1.0], dtype=torch.bfloat16, requires_grad=True)
    query = torch.tensor([0.5, 3.0], dtype=torch.bfloat16, requires_grad=True)

    scores = _score([_record(0, summary)], query)

    assert scores.dtype == torch.float32
    torch.testing.assert_close(scores, torch.tensor([-2.0]), rtol=0.0, atol=0.0)
    scores.sum().backward()
    assert query.grad is not None
    torch.testing.assert_close(
        query.grad.float(),
        summary.detach().float(),
        rtol=0.0,
        atol=0.0,
    )
    assert summary.grad is None


def test_score_all_cosine_is_computed_in_fp32() -> None:
    summary = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    query = torch.tensor([2.0, -1.0], dtype=torch.bfloat16, requires_grad=True)

    scores = _score([_record(0, summary)], query, normalize=True)
    expected = torch.nn.functional.cosine_similarity(
        summary.float().unsqueeze(0),
        query.detach().float().unsqueeze(0),
    )

    assert scores.dtype == torch.float32
    torch.testing.assert_close(scores, expected, rtol=1e-6, atol=1e-6)
    scores.sum().backward()
    assert query.grad is not None


def test_stable_topk_exact_ties_prefer_smaller_chunk_id() -> None:
    records = [_record(8), _record(2), _record(5)]
    scores = torch.tensor([3.0, 3.0, 3.0])

    result = stable_topk(scores, records, 2)

    assert _selected_chunk_ids(result.indices, records) == [2, 5]
    assert torch.equal(result.values, torch.tensor([3.0, 3.0]))


def test_stable_topk_near_scores_keep_score_order_without_perturbation() -> None:
    lower = torch.tensor(1.0, dtype=torch.float32)
    higher = torch.nextafter(lower, torch.tensor(float("inf")))
    records = [_record(1), _record(99), _record(2)]
    scores = torch.stack((lower, higher, torch.tensor(0.0)))
    before = scores.clone()
    softmax_before = torch.softmax(scores, dim=0)

    result = stable_topk(scores, records, 2)

    assert _selected_chunk_ids(result.indices, records) == [99, 1]
    assert torch.equal(result.values, scores[result.indices])
    assert torch.equal(scores, before)
    assert torch.equal(torch.softmax(scores, dim=0), softmax_before)


def test_stable_topk_caps_k_and_supports_smallest_score_order() -> None:
    records = [_record(7), _record(1), _record(4)]
    scores = torch.tensor([2.0, 2.0, 1.0])

    indices = stable_topk_indices(scores, records, 10, largest=False)

    assert indices.device == scores.device
    assert _selected_chunk_ids(indices, records) == [4, 1, 7]
    assert stable_topk_indices(scores, records, 0).numel() == 0


def test_stable_topk_is_independent_of_record_order_for_ties() -> None:
    records_a = [_record(9), _record(3), _record(6), _record(1)]
    records_b = [_record(1), _record(6), _record(9), _record(3)]
    scores_a = torch.ones(len(records_a))
    scores_b = torch.ones(len(records_b))

    indices_a = stable_topk_indices(scores_a, records_a, 3)
    indices_b = stable_topk_indices(scores_b, records_b, 3)

    assert _selected_chunk_ids(indices_a, records_a) == [1, 3, 6]
    assert _selected_chunk_ids(indices_b, records_b) == [1, 3, 6]


def test_stable_topk_preserves_device_and_value_gradient() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = [_record(2), _record(1)]
    scores = torch.tensor([1.0, 1.0], device=device, requires_grad=True)

    result = stable_topk(scores, records, 1)

    assert result.indices.device == scores.device
    assert result.values.device == scores.device
    result.values.sum().backward()
    torch.testing.assert_close(
        scores.grad,
        torch.tensor([0.0, 1.0], device=device),
        rtol=0.0,
        atol=0.0,
    )
