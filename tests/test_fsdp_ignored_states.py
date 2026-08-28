import inspect
import unittest
from unittest import mock

import torch

from utils import distributed


class YXFSDPIgnoredStatesConfigTest(unittest.TestCase):
    def _capture_fsdp_wrap(self, ignored_states=None, pass_argument=False):
        module = torch.nn.Linear(2, 2)
        wrapped = object()
        captured = {}

        def capture_constructor(module_arg, **kwargs):
            captured["module"] = module_arg
            captured.update(kwargs)
            return wrapped

        with mock.patch.object(
            distributed, "FSDP", side_effect=capture_constructor
        ), mock.patch.object(distributed.torch.cuda, "current_device", return_value=0):
            if pass_argument:
                result = distributed.fsdp_wrap(
                    module, ignored_states=ignored_states
                )
            else:
                result = distributed.fsdp_wrap(module)

        self.assertIs(result, wrapped)
        self.assertIs(captured["module"], module)
        return captured

    def test_current_torch_fsdp_supports_ignored_states(self):
        signature = inspect.signature(distributed.FSDP.__init__)

        self.assertIn("ignored_states", signature.parameters)
        self.assertIsNone(signature.parameters["ignored_states"].default)

    def test_default_ignored_states_is_forwarded_as_none(self):
        captured = self._capture_fsdp_wrap()

        self.assertIn("ignored_states", captured)
        self.assertIsNone(captured["ignored_states"])

    def test_parameter_list_is_forwarded_unchanged(self):
        ignored_states = [
            torch.nn.Parameter(torch.ones(1)),
            torch.nn.Parameter(torch.zeros(1)),
        ]

        captured = self._capture_fsdp_wrap(
            ignored_states=ignored_states, pass_argument=True
        )

        self.assertIs(captured["ignored_states"], ignored_states)


if __name__ == "__main__":
    unittest.main()
