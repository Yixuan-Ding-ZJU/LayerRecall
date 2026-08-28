import unittest

import torch

from utils.layer_recall import HistoryChunkRecord, assemble_slot_visible_plan


def _record(chunk_index: int, start: int, end: int) -> HistoryChunkRecord:
    return HistoryChunkRecord(
        chunk_index=chunk_index,
        start_frame=chunk_index * 8,
        num_frames=8,
        cache_start_token=start,
        cache_end_token=end,
        summary=torch.zeros(128),
    )


class LayerRecallVisiblePolicyTest(unittest.TestCase):
    def test_32_frame_budget_reserves_two_history_slots(self):
        plan = assemble_slot_visible_plan(
            YX_selected_records=[
                _record(1, 8, 16),
                _record(2, 16, 24),
                _record(3, 32, 40),
            ],
            YX_sink_tokens=8,
            YX_current_start_token=24,
            YX_current_end_token=32,
            YX_max_attention_size=32,
            YX_chunk_token_size=8,
        )
        self.assertEqual(plan["YX_memory_slots_requested"], 2)
        self.assertEqual(plan["YX_memory_slots_filled"], 2)
        self.assertEqual(plan["YX_selected_chunk_ids"], [1, 2])
        self.assertEqual(plan["YX_visible_tokens"], 32)
        self.assertEqual(plan["YX_recent_tokens"], 0)

    def test_candidate_underfill_never_adds_a_recent_window(self):
        plan = assemble_slot_visible_plan(
            YX_selected_records=[_record(1, 8, 16)],
            YX_sink_tokens=8,
            YX_current_start_token=24,
            YX_current_end_token=32,
            YX_max_attention_size=32,
            YX_chunk_token_size=8,
        )
        self.assertEqual(plan["YX_memory_slots_requested"], 2)
        self.assertEqual(plan["YX_memory_slots_filled"], 1)
        self.assertEqual(plan["YX_memory_slots_unfilled"], 1)
        self.assertEqual(plan["YX_visible_underfill_tokens"], 8)
        self.assertEqual(plan["YX_recent_ranges"], [])

    def test_overlapping_or_wrong_length_records_are_skipped(self):
        plan = assemble_slot_visible_plan(
            YX_selected_records=[
                _record(1, 24, 32),
                _record(2, 8, 12),
                _record(3, 8, 16),
            ],
            YX_sink_tokens=8,
            YX_current_start_token=24,
            YX_current_end_token=32,
            YX_max_attention_size=32,
            YX_chunk_token_size=8,
        )
        self.assertEqual(plan["YX_selected_chunk_ids"], [3])
        self.assertEqual(plan["YX_memory_slots_filled"], 1)

    def test_forced_sink_and_current_cannot_exceed_budget(self):
        with self.assertRaisesRegex(RuntimeError, "forced tokens exceed"):
            assemble_slot_visible_plan(
                YX_selected_records=[],
                YX_sink_tokens=16,
                YX_current_start_token=24,
                YX_current_end_token=40,
                YX_max_attention_size=24,
                YX_chunk_token_size=8,
            )


if __name__ == "__main__":
    unittest.main()
