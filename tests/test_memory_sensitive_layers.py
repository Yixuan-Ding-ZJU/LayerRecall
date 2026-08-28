import unittest

from utils.layer_recall import (
    LayerRecallConfig,
    LayerRecallSelectionLogger,
    is_layer_recall_enabled_for_layer,
    parse_layer_ids,
)


class MemorySensitiveLayerTest(unittest.TestCase):
    def test_layer_id_parser_supports_ranges_and_deduplicates(self):
        self.assertEqual(
            parse_layer_ids("0,2-4;3,6", num_layers=7),
            (0, 2, 3, 4, 6),
        )
        self.assertEqual(parse_layer_ids("", num_layers=7), ())

    def test_layer_id_parser_rejects_invalid_values(self):
        invalid_values = (
            ("1,two", ValueError, "invalid token"),
            ("1,,2", ValueError, "empty token"),
            ("4-2", ValueError, "ascending"),
            ([1, 2.5], TypeError, "non-integer token"),
            ([True], TypeError, "bool token"),
        )
        for value, exception_type, match in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(exception_type, match):
                    parse_layer_ids(value, field_name="test_layers")

    def test_config_reads_the_public_memory_sensitive_layer_list(self):
        config = LayerRecallConfig.from_repo_config(
            {
                "layer_recall": {
                    "layer_recall_num_layers": 30,
                    "memory_sensitive_layers": "4,9-10,12,26",
                }
            }
        )
        self.assertEqual(config.memory_sensitive_layers, (4, 9, 10, 12, 26))
        self.assertEqual(
            is_layer_recall_enabled_for_layer(config, 10),
            (True, "memory_sensitive_layer"),
        )
        self.assertEqual(
            is_layer_recall_enabled_for_layer(config, 11),
            (False, "original_window_layer"),
        )

    def test_default_policy_matches_the_paper_top10(self):
        config = LayerRecallConfig()
        self.assertEqual(
            config.memory_sensitive_layers,
            (4, 9, 10, 12, 13, 15, 16, 17, 18, 26),
        )

    def test_config_rejects_out_of_range_layers(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            LayerRecallConfig.from_repo_config(
                {
                    "layer_recall": {
                        "layer_recall_num_layers": 4,
                        "memory_sensitive_layers": "0,4",
                    }
                }
            )

    def test_logger_separates_memory_sensitive_and_original_window_events(self):
        logger = LayerRecallSelectionLogger(
            LayerRecallConfig(layer_recall_enabled=True)
        )
        logger.log(
            {
                "memory_sensitive_layer": True,
                "layer_recall_layout_applied": True,
                "layer_recall_gate_active": True,
                "layer_recall_selection_mode": "hard",
            }
        )
        logger.log(
            {
                "memory_sensitive_layer": False,
                "layer_recall_layout_applied": False,
                "layer_recall_gate_active": False,
                "layer_recall_selection_mode": "hard",
            }
        )
        snapshot = logger.snapshot_counters()
        self.assertEqual(snapshot["YX_log_events"], 2)
        self.assertEqual(snapshot["memory_sensitive_layer_events"], 1)
        self.assertEqual(snapshot["original_window_layer_events"], 1)
        self.assertEqual(snapshot["layer_recall_layout_applied_events"], 1)


if __name__ == "__main__":
    unittest.main()
