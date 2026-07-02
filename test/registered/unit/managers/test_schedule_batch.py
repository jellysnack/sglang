import unittest

from sglang.srt.managers.schedule_batch import (
    _split_cached_tokens_by_source,
    _swa_recompute_commit_len,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestScheduleBatch(CustomTestCase):
    def test_swa_recompute_commit_len_is_regular_swa_tail(self):
        self.assertEqual(
            _swa_recompute_commit_len(
                recompute_len=5376, sliding_window_size=128, page_size=256
            ),
            256,
        )
        self.assertEqual(
            _swa_recompute_commit_len(
                recompute_len=96, sliding_window_size=128, page_size=16
            ),
            96,
        )
        self.assertEqual(
            _swa_recompute_commit_len(
                recompute_len=0, sliding_window_size=128, page_size=256
            ),
            0,
        )

    def test_swa_recompute_overlap_consumes_storage_suffix_first(self):
        self.assertEqual(
            _split_cached_tokens_by_source(
                pre_len=50,
                host_hit_length=100,
                storage_hit_length=80,
                swa_recompute_len=50,
            ),
            (0, 20, 30),
        )


if __name__ == "__main__":
    unittest.main()
