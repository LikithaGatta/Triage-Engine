"""
scripts/train_distilbert.py
============================
Main script for Week 2 — fine-tunes DistilBERT and compares
results against the Week 1 TF-IDF baseline.

HOW TO RUN:
    source venv/bin/activate
    python scripts/train_distilbert.py

WHAT YOU WILL SEE:
    - DistilBERT downloading from HuggingFace (first run only)
    - 3 epoch progress bars with live accuracy
    - Side-by-side comparison vs Week 1 baseline
    - Your resume metrics printed at the end

HOW LONG DOES THIS TAKE?
    Apple Silicon Mac (M1/M2/M3): 3-8 minutes
    Intel Mac (CPU only):         15-30 minutes
    With NVIDIA GPU:              2-5 minutes
"""

import sys
from pathlib import Path

# Add project root to Python path before any src imports
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import argparse
import joblib
import pandas as pd
from dotenv import load_dotenv

from src.data.loader import DatasetLoader
from src.data.preprocessor import TicketPreprocessor
from src.models.distilbert_classifier import DistilBertClassifier
from src.models.baseline_classifier import BaselineClassifier
from src.utils.logger import logger

load_dotenv()


def parse_args():
    """Command line arguments for customizing the training run."""
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for ticket classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs. More = better accuracy but slower.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Tickets per training step. Reduce to 8 if you run out of memory.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate. 2e-5 is the standard for DistilBERT fine-tuning.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=150,
        help="Synthetic tickets per category for dataset generation.",
    )
    return parser.parse_args()


def load_existing_splits() -> tuple:
    """
    Load the train/test splits saved during Week 1 training.

    We use the SAME splits as Week 1 so the comparison is fair.
    If we used different splits, any accuracy difference might be
    due to which tickets ended up in train vs test, not model quality.

    Returns:
        Tuple of (train_df, test_df) or (None, None) if not found
    """
    train_path = Path("data/processed/train.csv")
    test_path  = Path("data/processed/test.csv")

    if train_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)
        logger.info(f"Loaded existing splits: {len(train_df)} train, {len(test_df)} test")
        return train_df, test_df

    return None, None


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("WEEK 2: DISTILBERT FINE-TUNING")
    logger.info(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.learning_rate}")
    logger.info("=" * 60)


    # ============================================================
    # STEP 1: LOAD DATA
    # We use the same train/test split from Week 1 for a fair comparison.
    # If those files do not exist yet, we generate and split fresh data.
    # ============================================================

    logger.info("\n[Step 1 of 4] Loading data...")

    train_df, test_df = load_existing_splits()

    if train_df is None:
        logger.info("No existing splits found. Generating fresh data...")
        loader = DatasetLoader()
        df = loader.load_or_generate(n_synthetic_per_category=args.n_samples)
        preprocessor = TicketPreprocessor()
        df = preprocessor.process_dataframe(df)
        df = df[~df["too_short"]]
        train_df, test_df = loader.get_train_test_split(df)

    logger.info(f"Train: {len(train_df):,} tickets | Test: {len(test_df):,} tickets")


    # ============================================================
    # STEP 2: LOAD WEEK 1 BASELINE RESULTS FOR COMPARISON
    # We want to show the improvement number on the resume.
    # Load the saved Week 1 model and get its test accuracy.
    # ============================================================

    logger.info("\n[Step 2 of 4] Loading Week 1 baseline for comparison...")

    baseline_accuracy = None
    baseline_model_path = Path("models/baseline_v1.0.0.joblib")

    if baseline_model_path.exists() and "processed_text" in train_df.columns:
        try:
            # Load the saved Week 1 TF-IDF model
            baseline = BaselineClassifier(model_version="1.0.0")
            baseline.load(str(baseline_model_path))

            # Run it on the test set to get its accuracy
            baseline_metrics = baseline.evaluate(test_df)
            baseline_accuracy = baseline_metrics["test_accuracy"]
            logger.info(f"Week 1 baseline accuracy: {baseline_accuracy:.1%}")
        except Exception as e:
            logger.warning(f"Could not load baseline model: {e}")
            logger.info("Will skip comparison — baseline accuracy unknown")
    else:
        logger.info("No baseline model found. Run train_baseline.py first for comparison.")


    # ============================================================
    # STEP 3: FINE-TUNE DISTILBERT
    # This is the main event. 3 epochs of fine-tuning.
    # You will see live progress bars showing loss and accuracy.
    # ============================================================

    logger.info("\n[Step 3 of 4] Fine-tuning DistilBERT...")
    logger.info("First run will download DistilBERT (~250MB). Please wait...")

    # Create the classifier
    clf = DistilBertClassifier(model_version="2.0.0")

    # Fine-tune and evaluate
    # This returns the final test metrics after all epochs complete
    metrics = clf.train(
        train_df=train_df,
        test_df=test_df,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


    # ============================================================
    # STEP 4: RESULTS AND COMPARISON
    # Print the numbers you will put on your resume.
    # ============================================================

    logger.info("\n[Step 4 of 4] Results")

    distilbert_accuracy = metrics["test_accuracy"]
    distilbert_f1       = metrics["f1_macro"]

    logger.info("\n" + "=" * 60)
    logger.info("WEEK 2 RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Training tickets:        {metrics['n_train']:,}")
    logger.info(f"  Test tickets:            {metrics['n_test']:,}")
    logger.info(f"")
    logger.info(f"  DistilBERT accuracy:     {distilbert_accuracy:.1%}")
    logger.info(f"  DistilBERT Macro F1:     {distilbert_f1:.1%}")
    logger.info(f"")

    # Show the improvement over baseline if we have it
    if baseline_accuracy is not None:
        improvement = distilbert_accuracy - baseline_accuracy
        logger.info(f"  Week 1 baseline:         {baseline_accuracy:.1%}")
        logger.info(f"  Week 2 DistilBERT:       {distilbert_accuracy:.1%}")
        logger.info(f"  Improvement:             +{improvement:.1%} ({improvement*100:.1f} percentage points)")
        logger.info(f"")
        logger.info(f"  YOUR RESUME BULLET:")
        logger.info(f"  Fine-tuned DistilBERT on {metrics['n_train']} labeled support")
        logger.info(f"  tickets, achieving {distilbert_accuracy:.0%} routing accuracy —")
        logger.info(f"  a {improvement*100:.0f}-point improvement over the TF-IDF baseline.")

    logger.info("=" * 60)


    # ============================================================
    # DEMO: SHOW LIVE PREDICTIONS
    # These are examples you can show to recruiters
    # ============================================================

    logger.info("\nDEMO: Live ticket predictions")
    logger.info("-" * 40)

    demo_tickets = [
        "I was charged twice this month for my Pro subscription. Please refund.",
        "The app crashes every single time I try to upload a file. This is a bug.",
        "Would love a dark mode option — my eyes hurt after 8 hours on this screen.",
        "I am completely locked out of my account and the reset email never arrives.",
        "The dashboard takes 45 seconds to load. It was instant last week.",
    ]

    for ticket in demo_tickets:
        result = clf.predict_single(ticket)
        conf_bar = "=" * int(result["confidence"] * 20)
        logger.info(f"  Ticket:    '{ticket[:55]}...'")
        logger.info(f"  Predicted: {result['predicted_category']:<20} "
                    f"Confidence: {result['confidence']:.1%}  [{conf_bar}]")
        logger.info(f"  Auto-routed: {result['auto_routed']}")
        logger.info("")


    # ============================================================
    # NEXT STEPS
    # ============================================================

    logger.info("=" * 60)
    logger.info("NEXT STEPS")
    logger.info("=" * 60)
    logger.info("  1. View both experiments in MLflow:")
    logger.info("       mlflow ui  →  http://localhost:5000")
    logger.info("       Compare triage_baseline vs triage_distilbert side by side")
    logger.info("")
    logger.info("  2. Commit to GitHub:")
    logger.info("       git add .")
    logger.info('       git commit -m "Week 2: DistilBERT fine-tuned, XX% accuracy"')
    logger.info("       git push")
    logger.info("")
    logger.info("  3. Update your README with the new accuracy numbers")
    logger.info("")
    logger.info("  4. When ready → Week 3: SHAP explainability layer")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()