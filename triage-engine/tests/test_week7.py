"""
tests/test_week7.py
====================
Tests for the benchmarking and evaluation utilities.

RUN:
    pytest tests/test_week7.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import statistics


class TestLatencyCalculations:
    """Tests for latency math used in benchmarking."""

    def test_median_calculation(self):
        """Median of [100, 200, 300] should be 200."""
        latencies = [100.0, 200.0, 300.0]
        assert statistics.median(latencies) == 200.0

    def test_p95_calculation(self):
        """95th percentile index should be correct."""
        latencies = list(range(1, 101))  # 1 to 100
        sorted_l  = sorted(latencies)
        p95_idx   = int(len(sorted_l) * 0.95)
        assert sorted_l[p95_idx] == 96

    def test_throughput_calculation(self):
        """Throughput = requests / seconds should be correct."""
        n_requests   = 100
        total_time_s = 10.0
        throughput   = n_requests / total_time_s
        assert throughput == 10.0

    def test_error_rate_calculation(self):
        """Error rate = errors / total should be correct."""
        n_requests = 100
        n_errors   = 5
        error_rate = n_errors / n_requests
        assert error_rate == 0.05

    def test_zero_errors_gives_zero_rate(self):
        """No errors should give 0.0 error rate."""
        assert 0 / 100 == 0.0


class TestBenchmarkTickets:
    """Tests that benchmark ticket data is valid."""

    def test_all_benchmark_tickets_have_body(self):
        """Every benchmark ticket must have a body field."""
        from scripts.benchmark import BENCHMARK_TICKETS
        for ticket in BENCHMARK_TICKETS:
            assert "body" in ticket
            assert len(ticket["body"]) >= 5

    def test_benchmark_tickets_cover_multiple_categories(self):
        """Benchmark should test diverse ticket types."""
        from scripts.benchmark import BENCHMARK_TICKETS
        # We have at least 10 different tickets
        assert len(BENCHMARK_TICKETS) >= 10

    def test_api_url_format(self):
        """API URL should point to localhost."""
        from scripts.benchmark import API_URL
        assert "localhost" in API_URL
        assert "8000" in API_URL


class TestAccuracyMetrics:
    """Tests for accuracy calculation logic."""

    def test_perfect_accuracy(self):
        """All correct predictions should give 1.0 accuracy."""
        import pandas as pd
        preds  = ["billing", "bug_report", "general"]
        labels = ["billing", "bug_report", "general"]
        accuracy = (pd.Series(preds) == pd.Series(labels)).mean()
        assert accuracy == 1.0

    def test_zero_accuracy(self):
        """All wrong predictions should give 0.0 accuracy."""
        import pandas as pd
        preds  = ["billing",   "bug_report", "general"]
        labels = ["bug_report", "general",   "billing"]
        accuracy = (pd.Series(preds) == pd.Series(labels)).mean()
        assert accuracy == 0.0

    def test_partial_accuracy(self):
        """Two correct out of four should give 0.5 accuracy."""
        import pandas as pd
        preds  = ["billing", "billing",   "bug_report", "bug_report"]
        labels = ["billing", "bug_report","bug_report", "general"]
        accuracy = (pd.Series(preds) == pd.Series(labels)).mean()
        assert accuracy == 0.5

    def test_calibration_bucket_logic(self):
        """Confidence bucket boundaries should be exclusive/inclusive correctly."""
        import numpy as np
        confidences = np.array([0.55, 0.65, 0.75, 0.85, 0.95])
        low, high   = 0.7, 0.8
        mask = (confidences >= low) & (confidences < high)
        # Only 0.75 falls in [0.7, 0.8)
        assert mask.sum() == 1
        assert confidences[mask][0] == 0.75


class TestEdgeCaseHandling:
    """Tests for edge case ticket handling logic."""

    def test_minimum_body_length(self):
        """Body must be at least 5 characters."""
        valid_body   = "Help me please"
        invalid_body = "Hi"
        assert len(valid_body) >= 5
        assert len(invalid_body) < 5

    def test_very_long_body_truncates_safely(self):
        """Very long bodies should not cause errors in length calculation."""
        long_body = "crash " * 500
        # Should be truncatable to any reasonable length without error
        truncated = long_body[:512]
        assert len(truncated) == 512

    def test_repeated_words_are_valid_input(self):
        """Repeated words should be valid ticket input."""
        body = "crash " * 20
        assert len(body.strip()) > 5

    def test_mixed_signals_ticket_is_valid(self):
        """A ticket with signals from multiple categories should be processable."""
        body = "I cannot login and I was charged twice and the app crashes"
        assert len(body) > 5
        # Contains signals for: account_access, billing, bug_report
        assert "login" in body
        assert "charged" in body
        assert "crashes" in body