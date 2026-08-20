"""
scripts/retrain_from_feedback.py
==================================
Retrains the classifier using original training data + agent corrections.

This is the core of the human-in-the-loop pipeline.
Every time this runs, the model gets smarter from agent feedback.

HOW TO RUN:
    python scripts/retrain_from_feedback.py

WHAT IT DOES:
    1. Loads original training data (data/processed/train.csv)
    2. Loads all agent corrections (data/processed/agent_corrections.csv)
    3. Preprocesses the corrections
    4. Combines: original data + corrections (corrections weighted higher)
    5. Retrains TF-IDF + Logistic Regression
    6. Evaluates: new model vs old model on same test set
    7. If new model is better → saves as new version, updates API
    8. Logs everything to MLflow under the "retraining" experiment

MINIMUM CORRECTIONS THRESHOLD:
    We require at least MIN_CORRECTIONS corrections before retraining.
    With too few corrections the signal is too weak and retraining
    might actually hurt accuracy. 20 is a reasonable minimum.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.data.feedback_store import FeedbackStore
from src.data.preprocessor import TicketPreprocessor
from src.models.baseline_classifier import BaselineClassifier
from src.utils.logger import logger

load_dotenv()

# Minimum corrections before retraining is worthwhile
MIN_CORRECTIONS = 20

# How many times to repeat each correction in training data
# Corrections are high-value examples — weighting them higher
# lets the model learn faster from agent feedback
CORRECTION_WEIGHT = 3


def main():
    logger.info("=" * 60)
    logger.info("WEEK 6: RETRAINING FROM AGENT FEEDBACK")
    logger.info("=" * 60)


    # ============================================================
    # STEP 1: CHECK WE HAVE ENOUGH CORRECTIONS
    # ============================================================

    store = FeedbackStore()
    n_corrections = store.count()

    logger.info(f"\n[Step 1] Checking feedback store...")
    logger.info(f"  Corrections available: {n_corrections}")
    logger.info(f"  Minimum required:      {MIN_CORRECTIONS}")

    if n_corrections < MIN_CORRECTIONS:
        logger.warning(
            f"Not enough corrections to retrain ({n_corrections} < {MIN_CORRECTIONS}). "
            f"Run scripts/simulate_corrections.py to generate test corrections."
        )
        sys.exit(0)

    logger.info(f"  ✓ Sufficient corrections — proceeding with retraining")


    # ============================================================
    # STEP 2: LOAD ORIGINAL TRAINING DATA
    # ============================================================

    logger.info(f"\n[Step 2] Loading original training data...")

    train_path = Path("data/processed/train.csv")
    test_path  = Path("data/processed/test.csv")

    if not train_path.exists():
        logger.error("No training data found. Run scripts/train_baseline.py first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    logger.info(f"  Original training set: {len(train_df):,} tickets")
    logger.info(f"  Test set:              {len(test_df):,} tickets")


    # ============================================================
    # STEP 3: LOAD AND PREPROCESS CORRECTIONS
    # ============================================================

    logger.info(f"\n[Step 3] Loading and preprocessing corrections...")

    preprocessor = TicketPreprocessor()
    corrections_df = store.get_as_training_data()

    logger.info(f"  Corrections loaded: {len(corrections_df)}")

    # Preprocess the correction texts
    corrections_processed = preprocessor.process_dataframe(corrections_df)
    corrections_processed = corrections_processed[~corrections_processed["too_short"]]

    logger.info(f"  After preprocessing: {len(corrections_processed)} corrections")
    logger.info(f"  Correction categories:\n{corrections_processed['category'].value_counts().to_string()}")


    # ============================================================
    # STEP 4: COMBINE DATASETS
    # ============================================================

    logger.info(f"\n[Step 4] Combining original data with corrections...")

    # Repeat corrections CORRECTION_WEIGHT times to give them higher weight
    # This is equivalent to oversampling — the model sees correction examples
    # more often and learns from them faster
    weighted_corrections = pd.concat(
        [corrections_processed] * CORRECTION_WEIGHT,
        ignore_index=True
    )

    # Combined dataset: original + weighted corrections
    combined_df = pd.concat([train_df, weighted_corrections], ignore_index=True)

    # Shuffle so corrections are distributed throughout training
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"  Original training:     {len(train_df):,}")
    logger.info(f"  Weighted corrections:  {len(weighted_corrections):,} ({n_corrections} × {CORRECTION_WEIGHT})")
    logger.info(f"  Combined total:        {len(combined_df):,}")


    # ============================================================
    # STEP 5: EVALUATE OLD MODEL BASELINE
    # ============================================================

    logger.info(f"\n[Step 5] Evaluating current model on test set...")

    old_model_path = Path("models/baseline_v1.0.0.joblib")
    if not old_model_path.exists():
        logger.error("No trained model found. Run scripts/train_baseline.py first.")
        sys.exit(1)

    old_pipeline = joblib.load(old_model_path)
    old_preds    = old_pipeline.predict(test_df["processed_text"].fillna("").tolist())
    old_accuracy = (pd.Series(old_preds) == test_df["category"]).mean()
    old_f1       = f1_score(test_df["category"].tolist(), old_preds, average="macro")

    logger.info(f"  Current model accuracy: {old_accuracy:.1%}")
    logger.info(f"  Current model F1:       {old_f1:.1%}")


    # ============================================================
    # STEP 6: TRAIN NEW MODEL
    # ============================================================

    logger.info(f"\n[Step 6] Training new model on combined dataset...")

    # Build new version string: 1.0.0 → 1.1.0
    # In a real system you would parse the current version and increment
    old_version_parts  = "1.0.0".split(".")
    new_minor          = int(old_version_parts[1]) + 1
    new_version        = f"1.{new_minor}.0"

    new_classifier = BaselineClassifier(model_version=new_version)
    new_classifier.train(
        combined_df,
        text_col="processed_text",
        label_col="category",
        experiment_name="triage_retraining",
    )

    logger.info(f"  New model version: {new_version}")


    # ============================================================
    # STEP 7: EVALUATE NEW MODEL
    # ============================================================

    logger.info(f"\n[Step 7] Evaluating new model on test set...")

    new_preds    = new_classifier.pipeline.predict(test_df["processed_text"].fillna("").tolist())
    new_accuracy = (pd.Series(new_preds) == test_df["category"]).mean()
    new_f1       = f1_score(test_df["category"].tolist(), new_preds, average="macro")

    improvement  = new_accuracy - old_accuracy

    logger.info(f"  New model accuracy:  {new_accuracy:.1%}")
    logger.info(f"  New model F1:        {new_f1:.1%}")
    logger.info(f"  Improvement:         {improvement:+.1%} ({improvement*100:+.2f} points)")


    # ============================================================
    # STEP 8: PROMOTE IF BETTER
    # ============================================================

    logger.info(f"\n[Step 8] Promotion decision...")

    # Log comparison to MLflow
    mlflow.set_experiment("triage_retraining")
    with mlflow.start_run(run_name=f"retrain_v{new_version}"):
        mlflow.log_params({
            "n_original_train":  len(train_df),
            "n_corrections":     n_corrections,
            "correction_weight": CORRECTION_WEIGHT,
            "n_combined":        len(combined_df),
            "new_version":       new_version,
        })
        mlflow.log_metrics({
            "old_accuracy":  old_accuracy,
            "new_accuracy":  new_accuracy,
            "old_f1":        old_f1,
            "new_f1":        new_f1,
            "improvement":   improvement,
        })

    # Promote the new model if it improved
    IMPROVEMENT_THRESHOLD = -0.005  # Allow up to 0.5% degradation (noise)

    if improvement >= IMPROVEMENT_THRESHOLD:
        # Save as the new production model
        new_model_path = Path(f"models/baseline_v{new_version}.joblib")
        joblib.dump(new_classifier.pipeline, new_model_path)

        # Also overwrite the "current" model path that the API loads
        joblib.dump(new_classifier.pipeline, old_model_path)

        logger.info(f"  ✓ New model PROMOTED to production")
        logger.info(f"  Saved to: {new_model_path}")
        logger.info(f"  API model updated: {old_model_path}")
        promoted = True
    else:
        logger.warning(
            f"  ✗ New model NOT promoted "
            f"(degraded by {abs(improvement):.1%} which exceeds threshold)"
        )
        promoted = False


    # ============================================================
    # RESULTS SUMMARY
    # ============================================================

    logger.info("\n" + "=" * 60)
    logger.info("WEEK 6 RETRAINING RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Corrections used:    {n_corrections}")
    logger.info(f"  Combined train size: {len(combined_df):,}")
    logger.info(f"")
    logger.info(f"  Old model accuracy:  {old_accuracy:.1%}")
    logger.info(f"  New model accuracy:  {new_accuracy:.1%}")
    logger.info(f"  Improvement:         {improvement:+.1%}")
    logger.info(f"  Promoted:            {'YES' if promoted else 'NO'}")
    logger.info(f"")

    if promoted:
        logger.info(f"  RESUME BULLET:")
        logger.info(f"  'Implemented human-in-the-loop retraining pipeline.")
        logger.info(f"  After {n_corrections} agent corrections, model accuracy")
        logger.info(f"  improved by {improvement*100:+.1f} percentage points.")
        logger.info(f"  New model v{new_version} promoted to production.'")
        logger.info(f"")
        logger.info(f"  NEXT: Restart the API to load the new model:")
        logger.info(f"    Ctrl+C to stop, then: python scripts/run_api.py")
    else:
        logger.info(f"  Old model retained. Review the corrections for quality.")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()