"""
scripts/benchmark.py
=====================
Comprehensive benchmarking of the triage system.

Produces:
  - API throughput (requests/minute)
  - Latency percentiles (p50, p95, p99)
  - Model accuracy on held-out test set
  - Per-category accuracy breakdown
  - Confidence calibration analysis
  - SHAP explanation coverage
  - Auto-route rate at different confidence thresholds

To Run: 
  Check that API is running first:
    python scripts/run_api.py   (in a separate terminal)

  Run benchmark:
    python scripts/benchmark.py

"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import statistics
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests

from src.data.preprocessor import TicketPreprocessor
from src.models.explainer import TicketExplainer
from src.utils.logger import logger

# API base URL
API_URL = "http://localhost:8000/api/v1"

# ----------------------------------------------------------------
# BENCHMARK TICKETS
# A diverse set for load testing — covers all categories,
# different lengths, and some ambiguous cases
# ----------------------------------------------------------------

BENCHMARK_TICKETS = [
    {"subject": "Double charge", "body": "I was charged twice for my Pro subscription this month. Please refund the duplicate $49 charge immediately."},
    {"subject": "App crashes", "body": "The application crashes every time I try to upload a file larger than 5MB. This started after the last update."},
    {"subject": "Dark mode request", "body": "Would love a dark mode option for the dashboard. Spending 8 hours a day and the white background is straining my eyes."},
    {"subject": "Cannot login", "body": "Locked out of my account. Password reset email never arrives. I have tried four times now."},
    {"subject": "Dashboard slow", "body": "The analytics dashboard takes over 30 seconds to load. It was instant before the recent update."},
    {"subject": "How does billing work", "body": "Quick question about how usage is calculated for billing purposes. Could not find a clear answer in the docs."},
    {"subject": "Refund request", "body": "I cancelled my subscription within the trial period but was still charged. I need a full refund processed."},
    {"subject": "Export broken", "body": "The CSV export feature fails every single time with a 500 error. This is blocking our weekly reporting process."},
    {"subject": "API integration", "body": "Do you have a Zapier integration? We use it heavily and connecting your platform would completely change our workflow."},
    {"subject": "2FA issue", "body": "Lost access to my authenticator app. Cannot get past two factor authentication. I have backup codes but they are not accepted."},
    {"subject": "Search timeout", "body": "Search results take over 45 seconds to appear after typing. Makes the product nearly unusable for our team daily."},
    {"subject": "Invoice needed", "body": "Can you send a PDF invoice for my last three payments? My company requires them for expense reporting."},
]


def check_api_health() -> bool:
    """Verify the API is running before starting benchmarks."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def single_request(ticket: Dict) -> Tuple[float, bool, Dict]:
    """
    Make one API request and return (latency_ms, success, response_data).
    Used by the load test to measure individual request performance.
    """
    start = time.time()
    try:
        response = requests.post(
            f"{API_URL}/tickets",
            json=ticket,
            timeout=15,
        )
        latency_ms = (time.time() - start) * 1000
        success = response.status_code == 201
        data = response.json() if success else {}
        return latency_ms, success, data
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        return latency_ms, False, {}


def run_load_test(
    n_requests: int = 100,
    concurrency: int = 10,
) -> Dict:
    """
    Send n_requests to the API with concurrency concurrent workers.
    Measures throughput and latency distribution.

    Args:
        n_requests:  Total number of requests to send
        concurrency: How many requests to send simultaneously
    Returns:
        Dict with throughput, latency percentiles, error rate
    """
    logger.info(f"Load test: {n_requests} requests, {concurrency} concurrent")

    # Build request list by cycling through benchmark tickets
    tickets = [BENCHMARK_TICKETS[i % len(BENCHMARK_TICKETS)] for i in range(n_requests)]

    latencies  = []
    successes  = []
    start_time = time.time()

    # ThreadPoolExecutor sends requests concurrently
    # max_workers controls how many run simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_request, ticket) for ticket in tickets]

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            latency_ms, success, _ = future.result()
            latencies.append(latency_ms)
            successes.append(success)

            # Progress indicator every 25 requests
            if (i + 1) % 25 == 0:
                logger.info(f"  Progress: {i+1}/{n_requests} requests completed")

    total_time_s = time.time() - start_time
    n_success    = sum(successes)
    n_errors     = n_requests - n_success

    # Calculate latency percentiles
    latencies_sorted = sorted(latencies)

    results = {
        "n_requests":     n_requests,
        "concurrency":    concurrency,
        "total_time_s":   round(total_time_s, 2),
        "throughput_rps": round(n_requests / total_time_s, 1),
        "throughput_rpm": round(n_requests / total_time_s * 60, 0),
        "n_success":      n_success,
        "n_errors":       n_errors,
        "error_rate":     round(n_errors / n_requests, 4),
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(latencies_sorted[int(len(latencies) * 0.95)], 1),
        "latency_p99_ms": round(latencies_sorted[int(len(latencies) * 0.99)], 1),
        "latency_min_ms": round(min(latencies), 1),
        "latency_max_ms": round(max(latencies), 1),
        "latency_mean_ms":round(statistics.mean(latencies), 1),
    }

    return results


def run_accuracy_evaluation() -> Dict:
    """
    Evaluate model accuracy on the held-out test set.
    Produces per-category breakdown and confidence calibration.

    This is independent of the API — runs the model directly
    so we get detailed metrics beyond what the API exposes.
    """
    logger.info("Running accuracy evaluation on test set...")

    test_path  = Path("data/processed/test.csv")
    model_path = Path("models/baseline_v1.0.0.joblib")

    if not test_path.exists() or not model_path.exists():
        logger.warning("Test data or model not found. Skipping accuracy evaluation.")
        return {}

    test_df  = pd.read_csv(test_path)
    pipeline = joblib.load(model_path)

    texts      = test_df["processed_text"].fillna("").tolist()
    true_labels = test_df["category"].tolist()

    # Get predictions and probabilities
    pred_labels = pipeline.predict(texts)
    pred_probas = pipeline.predict_proba(texts)

    # Overall accuracy
    overall_accuracy = (pd.Series(pred_labels) == pd.Series(true_labels)).mean()

    # Per-category accuracy
    categories = list(pipeline.classes_)
    per_category = {}
    for cat in categories:
        mask = [t == cat for t in true_labels]
        if sum(mask) == 0:
            continue
        cat_true = [t for t, m in zip(true_labels, mask) if m]
        cat_pred = [p for p, m in zip(pred_labels, mask) if m]
        cat_accuracy = (pd.Series(cat_pred) == pd.Series(cat_true)).mean()
        per_category[cat] = {
            "accuracy":  round(float(cat_accuracy), 4),
            "n_samples": sum(mask),
        }

    # Confidence statistics
    max_probas = pred_probas.max(axis=1)
    auto_route_75 = (max_probas >= 0.75).mean()
    auto_route_80 = (max_probas >= 0.80).mean()
    auto_route_90 = (max_probas >= 0.90).mean()

    # Confidence calibration
    # Group predictions by confidence bucket and check if accuracy matches
    calibration = {}
    buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for low, high in buckets:
        mask = (max_probas >= low) & (max_probas < high)
        if mask.sum() == 0:
            continue
        bucket_true = [t for t, m in zip(true_labels, mask) if m]
        bucket_pred = [p for p, m in zip(pred_labels, mask) if m]
        bucket_accuracy = (pd.Series(bucket_pred) == pd.Series(bucket_true)).mean()
        calibration[f"{int(low*100)}-{int(high*100)}%"] = {
            "n_samples": int(mask.sum()),
            "avg_confidence": round(float(max_probas[mask].mean()), 3),
            "accuracy": round(float(bucket_accuracy), 3),
        }

    # SHAP explanation coverage
    explainer_path = Path("models/shap_explainer.joblib")
    shap_coverage = None
    if explainer_path.exists():
        explainer_data = joblib.load(explainer_path)
        explainer = TicketExplainer(
            pipeline=pipeline,
            category_names=explainer_data["category_names"],
            n_top_tokens=5,
        )
        explainer.shap_explainer  = explainer_data["shap_explainer"]
        explainer.background_mean = explainer_data["background_mean"]
        explainer.feature_names   = explainer_data["feature_names"]
        explainer.is_fitted       = True

        # Sample 50 tickets for SHAP evaluation (full set is slow)
        sample_texts = texts[:50]
        explanations = explainer.explain_batch(sample_texts)
        strong = sum(
            1 for e in explanations
            if len(e.top_positive) >= 2 and e.top_positive[0].shap_value > 0.05
        )
        shap_coverage = round(strong / len(explanations), 3)

    return {
        "overall_accuracy":  round(float(overall_accuracy), 4),
        "n_test_samples":    len(test_df),
        "per_category":      per_category,
        "auto_route_rate_75": round(float(auto_route_75), 4),
        "auto_route_rate_80": round(float(auto_route_80), 4),
        "auto_route_rate_90": round(float(auto_route_90), 4),
        "avg_confidence":    round(float(max_probas.mean()), 4),
        "calibration":       calibration,
        "shap_coverage":     shap_coverage,
    }


def run_edge_case_tests() -> Dict:
    """
    Test API behavior on edge cases:
    - Very short tickets
    - Very long tickets
    - All punctuation / numbers
    - Repeated words
    - Mixed category signals
    """
    logger.info("Running edge case tests...")

    edge_cases = [
        {
            "name":    "minimum_length",
            "ticket":  {"body": "Help me please with this"},
            "expect":  "valid_prediction",
        },
        {
            "name":    "very_long_ticket",
            "ticket":  {"body": "I need help. " * 100},
            "expect":  "valid_prediction",
        },
        {
            "name":    "repeated_word",
            "ticket":  {"body": "crash crash crash crash crash crash crash crash"},
            "expect":  "valid_prediction",
        },
        {
            "name":    "mixed_signals",
            "ticket":  {"body": "I cannot login and I was charged twice and the app crashes and it is very slow"},
            "expect":  "valid_prediction",
        },
        {
            "name":    "only_subject_no_body_context",
            "ticket":  {"subject": "Billing issue", "body": "Please help me with this problem I am having"},
            "expect":  "valid_prediction",
        },
        {
            "name":    "numbers_only_context",
            "ticket":  {"body": "I was charged $99.99 twice. Reference: 12345678. Date: 2026-01-15."},
            "expect":  "valid_prediction",
        },
    ]

    results = []
    for case in edge_cases:
        try:
            latency, success, data = single_request(case["ticket"])
            result = {
                "name":              case["name"],
                "success":           success,
                "latency_ms":        round(latency, 1),
                "predicted_category": data.get("predicted_category", "N/A"),
                "confidence":        round(data.get("confidence", 0), 3),
                "auto_routed":       data.get("auto_routed", False),
                "has_explanation":   bool(data.get("explanation", {}).get("top_positive")),
            }
        except Exception as e:
            result = {
                "name":    case["name"],
                "success": False,
                "error":   str(e),
            }
        results.append(result)
        logger.info(f"  {case['name']}: {'✓' if result.get('success') else '✗'} "
                   f"{result.get('predicted_category', 'ERROR')} "
                   f"({result.get('confidence', 0):.0%})")

    return {"edge_cases": results}


def print_benchmark_report(
    load_results: Dict,
    accuracy_results: Dict,
    edge_results: Dict,
) -> None:
    """Print a clean formatted benchmark report."""

    print("\n" + "="*65)
    print("  TRIAGE ENGINE — BENCHMARK REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65)

    if load_results:
        print(f"\n{'─'*65}")
        print("  LOAD TEST RESULTS")
        print(f"{'─'*65}")
        print(f"  Requests sent:       {load_results['n_requests']}")
        print(f"  Concurrency:         {load_results['concurrency']} simultaneous")
        print(f"  Total time:          {load_results['total_time_s']}s")
        print(f"")
        print(f"  Throughput:          {load_results['throughput_rps']} req/sec")
        print(f"                       {int(load_results['throughput_rpm'])} req/min")
        print(f"  Error rate:          {load_results['error_rate']:.1%}")
        print(f"")
        print(f"  Latency p50:         {load_results['latency_p50_ms']}ms")
        print(f"  Latency p95:         {load_results['latency_p95_ms']}ms  ← quote this to interviewers")
        print(f"  Latency p99:         {load_results['latency_p99_ms']}ms")
        print(f"  Latency mean:        {load_results['latency_mean_ms']}ms")

    if accuracy_results:
        print(f"\n{'─'*65}")
        print("  ACCURACY EVALUATION")
        print(f"{'─'*65}")
        print(f"  Test samples:        {accuracy_results.get('n_test_samples', 'N/A')}")
        print(f"  Overall accuracy:    {accuracy_results.get('overall_accuracy', 0):.1%}")
        print(f"  Avg confidence:      {accuracy_results.get('avg_confidence', 0):.1%}")
        print(f"")
        print(f"  Auto-route rate:")
        print(f"    At 75% threshold:  {accuracy_results.get('auto_route_rate_75', 0):.1%}")
        print(f"    At 80% threshold:  {accuracy_results.get('auto_route_rate_80', 0):.1%}")
        print(f"    At 90% threshold:  {accuracy_results.get('auto_route_rate_90', 0):.1%}")

        if accuracy_results.get("per_category"):
            print(f"\n  Per-category accuracy:")
            for cat, data in accuracy_results["per_category"].items():
                bar = "█" * int(data["accuracy"] * 20)
                print(f"    {cat:<22} {data['accuracy']:.1%}  {bar}  (n={data['n_samples']})")

        if accuracy_results.get("calibration"):
            print(f"\n  Confidence calibration:")
            print(f"    {'Conf range':<15} {'Samples':<10} {'Avg conf':<12} {'Accuracy'}")
            for bucket, data in accuracy_results["calibration"].items():
                print(f"    {bucket:<15} {data['n_samples']:<10} {data['avg_confidence']:.1%}{'':6} {data['accuracy']:.1%}")

        if accuracy_results.get("shap_coverage") is not None:
            print(f"\n  SHAP explanation coverage: {accuracy_results['shap_coverage']:.1%}")
            print(f"  (fraction of tickets with ≥2 strong explanation tokens)")

    if edge_results and edge_results.get("edge_cases"):
        print(f"\n{'─'*65}")
        print("  EDGE CASE RESULTS")
        print(f"{'─'*65}")
        for case in edge_results["edge_cases"]:
            status = "✓" if case.get("success") else "✗"
            cat    = case.get("predicted_category", "ERROR")
            conf   = case.get("confidence", 0)
            print(f"  {status} {case['name']:<30} → {cat:<18} ({conf:.0%})")

    print(f"\n{'─'*65}")
    print("  RESUME METRICS SUMMARY")
    print(f"{'─'*65}")
    if load_results:
        print(f"  Throughput:     {int(load_results['throughput_rpm'])} tickets/minute")
        print(f"  p95 latency:    {load_results['latency_p95_ms']}ms per request")
        print(f"  Error rate:     {load_results['error_rate']:.1%}")
    if accuracy_results:
        print(f"  Accuracy:       {accuracy_results.get('overall_accuracy', 0):.1%} on held-out test set")
        print(f"  Auto-route:     {accuracy_results.get('auto_route_rate_75', 0):.1%} of tickets (≥75% confidence)")
    if accuracy_results.get("shap_coverage"):
        print(f"  SHAP coverage:  {accuracy_results['shap_coverage']:.1%} strong explanations")
    print("="*65 + "\n")


def main():
    logger.info("="*60)
    logger.info("WEEK 7: COMPREHENSIVE BENCHMARKING")
    logger.info("="*60)

    # ---- Check API is running ----
    logger.info("\nChecking API health...")
    if not check_api_health():
        logger.error(
            "API is not running. Start it first:\n"
            "  python scripts/run_api.py"
        )
        logger.info("\nSkipping load test and edge cases. Running accuracy evaluation only...")
        accuracy_results = run_accuracy_evaluation()
        print_benchmark_report({}, accuracy_results, {})
        return

    logger.info("API is healthy — running full benchmark suite")

    # ---- Load test ----
    logger.info("\n[1/3] Running load test...")
    load_results = run_load_test(n_requests=100, concurrency=10)

    # ---- Accuracy evaluation ----
    logger.info("\n[2/3] Running accuracy evaluation...")
    accuracy_results = run_accuracy_evaluation()

    # ---- Edge cases ----
    logger.info("\n[3/3] Running edge case tests...")
    edge_results = run_edge_case_tests()

    # ---- Print report ----
    print_benchmark_report(load_results, accuracy_results, edge_results)

    # ---- Save results to JSON ----
    output = {
        "generated_at":    datetime.now().isoformat(),
        "load_test":       load_results,
        "accuracy":        accuracy_results,
        "edge_cases":      edge_results,
    }
    output_path = Path("data/processed/benchmark_results.json")
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Full results saved to {output_path}")


if __name__ == "__main__":
    main()