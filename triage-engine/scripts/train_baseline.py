"""
scripts/train_baseline.py
==========================
This is the MAIN script for Week 1. Run this file to:
  1. Generate synthetic ticket data
  2. Clean and preprocess the text
  3. Train a TF-IDF + Logistic Regression baseline classifier
  4. Evaluate accuracy on a test set
  5. Save the trained model to disk
  6. Log everything to MLflow for tracking

TO RUN:
  Make sure your virtual environment is activated first:
    source venv/bin/activate

  Then from the triage-engine/ folder run:
    python scripts/train_baseline.py


"""

# SYSTEM PATH SETUP
#
# Path(__file__) = full path to this script file
# .parent        = the scripts/ folder
# .parent        = the triage-engine/ folder (one more level up)
# str(...)       = convert to string because sys.path needs strings
# sys.path.insert(0, ...) = add to the FRONT of the search path

import sys
from pathlib import Path

# Add the project root (triage-engine/) to Python's module search path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import argparse    # Lets us accept arguments from the command line
                   # e.g. python train_baseline.py --n-samples 200

import os          # Operating system utilities
import pandas as pd
from dotenv import load_dotenv

from src.data.loader import DatasetLoader
from src.data.preprocessor import TicketPreprocessor
from src.models.baseline_classifier import BaselineClassifier
from src.utils.logger import logger

# Load .env file so environment variables are available
load_dotenv()



def parse_args():
    """
    Define and parse command-line arguments.

    Returns:
        Namespace object where args.n_samples, args.test_size, etc.
        contain the values the user passed in (or the defaults).
    """
    parser = argparse.ArgumentParser(
        description="Train the Week 1 TF-IDF baseline triage classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --n-samples: how many synthetic tickets per category to generate
    # type=int means Python converts the string "200" to integer 200
    # default=150 means if the user does not pass this flag, use 150
    parser.add_argument(
        "--n-samples",
        type=int,
        default=150,
        help="Number of synthetic tickets per category. 6 categories x 150 = 900 total.",
    )

    # --test-size: fraction of data held out for testing
    # 0.20 = 20 percent test, 80 percent train (industry standard)
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of data for the test set. 0.20 = 80 percent train, 20 percent test.",
    )

    # --model-version: a label for this run logged to MLflow
    parser.add_argument(
        "--model-version",
        type=str,
        default="1.0.0",
        help="Version string logged to MLflow for this training run.",
    )

    return parser.parse_args()


# MAIN TRAINING FUNCTION

def main():
    """
    The main training pipeline. Runs all 4 steps in sequence:
      Step 1: Load data
      Step 2: Preprocess text
      Step 3: Split into train/test
      Step 4: Train and evaluate the model
    """

    # Get the command-line arguments (or defaults if none provided)
    args = parse_args()

    logger.info("=" * 60)
    logger.info("WEEK 1: BASELINE CLASSIFIER TRAINING")
    logger.info(f"Settings: {args.n_samples} tickets/category, "
                f"{args.test_size*100:.0f}% test set, "
                f"model v{args.model_version}")
    logger.info("=" * 60)


    # STEP 1: LOAD DATA

    logger.info("\n[Step 1 of 4] Loading data...")

    # DatasetLoader manages generating and saving ticket data
    loader = DatasetLoader()

    # load_or_generate() checks if data/processed/full_dataset.csv exists
    #   If YES: loads the CSV (fast)
    #   If NO:  generates synthetic data and saves it to CSV
    df = loader.load_or_generate(n_synthetic_per_category=args.n_samples)

    # len(df) = number of rows (tickets)
    # df.shape = (rows, columns) tuple
    logger.info(f"Dataset loaded: {len(df):,} tickets, {df.shape[1]} columns")
    logger.info(f"\nCategory counts:\n{df['category'].value_counts().to_string()}")
    logger.info(f"\nUrgency counts:\n{df['urgency'].value_counts().to_string()}")


    # STEP 2: PREPROCESS TEXT
    # Clean raw ticket text before feeding it to the ML model.
    #   - Lowercase everything
    #   - Remove URLs, emails, HTML tags, punctuation
    #   - Remove common words like "the", "a", "is" (stopwords)
    #   - Lemmatize: "charged" becomes "charge", "crashes" becomes "crash"
    #   - Combine subject + body into one "processed_text" field

    logger.info("\n[Step 2 of 4] Preprocessing text...")

    # min_word_count=3 flags tickets with fewer than 3 words after cleaning
    preprocessor = TicketPreprocessor(min_word_count=3)

    # process_dataframe() adds these columns to df:
    #   processed_text: cleaned text ready for TF-IDF
    #   word_count:     words remaining after cleaning
    #   char_count:     original character count
    #   too_short:      True if fewer than min_word_count words remain
    df = preprocessor.process_dataframe(df)

    # Show a before and after example so we can verify it works
    sample_row = df.sample(1, random_state=99).iloc[0]
    logger.info(f"\nPREPROCESSING EXAMPLE:")
    logger.info(f"  Category:  {sample_row['category']}")
    logger.info(f"  Raw body:  {str(sample_row.get('body', ''))[:100]}...")
    logger.info(f"  Processed: {sample_row['processed_text'][:100]}...")
    logger.info(f"  Words kept: {sample_row['word_count']}")

    # Remove tickets that are too short — not enough text to classify
    # ~ means NOT, so ~df["too_short"] keeps rows where too_short is False
    tickets_before = len(df)
    df = df[~df["too_short"]]
    tickets_removed = tickets_before - len(df)

    if tickets_removed > 0:
        logger.warning(f"Removed {tickets_removed} tickets that were too short after cleaning")

    logger.info(f"Tickets remaining: {len(df):,}")


    # STEP 3: TRAIN / TEST SPLIT
    # Divide data into two non-overlapping sets:
    #
    #   TRAINING SET (80%): Model learns from these.
    #                       It sees the text AND the correct label.
    #
    #   TEST SET (20%):     Hidden from the model during training.
    #                       Used ONLY to measure real-world accuracy.
    
    logger.info("\n[Step 3 of 4] Splitting into train and test sets...")

    # Returns two DataFrames: train_df and test_df
    # Stratified split: each category has the same proportion in both sets
    train_df, test_df = loader.get_train_test_split(
        df,
        test_size=args.test_size,
    )

    # Save splits to CSV for reproducibility
    # index=False means do not write row numbers into the file
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    logger.info(f"Train: {len(train_df):,} tickets saved to data/processed/train.csv")
    logger.info(f"Test:  {len(test_df):,} tickets saved to data/processed/test.csv")


    # STEP 4: TRAIN AND EVALUATE
    # TF-IDF converts text to numbers:
    #   "I was charged twice" becomes a vector like [0.0, 0.91, 0.0, ...]
    #   where each number represents how important a word is.
    #
    # Logistic Regression learns from those numbers:
    #   High "refund" score + high "charge" score = predict "billing"
    #   High "crash" score + high "error" score   = predict "bug_report"
    #
    # Pipeline chains them: text → TF-IDF → numbers → classifier → label

    logger.info("\n[Step 4 of 4] Training and evaluating...")

    # Create the classifier with a version label for MLflow tracking
    classifier = BaselineClassifier(model_version=args.model_version)

    # TRAIN
    # text_col tells it which DataFrame column has the cleaned text
    # label_col tells it which column has the correct category answers
    # MLflow automatically logs parameters and metrics inside .train()
    logger.info("Training model on training set...")
    train_metrics = classifier.train(
        train_df,
        text_col="processed_text",
        label_col="category",
        experiment_name="triage_baseline",
    )
    logger.info(f"Training accuracy: {train_metrics['train_accuracy']:.1%}")

    # EVALUATE on test set (data the model has never seen)
    # .evaluate() logs all metrics to MLflow automatically
    logger.info("Evaluating on held-out test set...")
    test_metrics = classifier.evaluate(
        test_df,
        text_col="processed_text",
        label_col="category",
    )


    # RESULTS SUMMARY


    logger.info("\n" + "=" * 60)
    logger.info("WEEK 1 RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total tickets:      {len(df):,}")
    logger.info(f"  Training set:       {len(train_df):,}")
    logger.info(f"  Test set:           {len(test_df):,}")
    logger.info(f"")
    logger.info(f"  Test accuracy:      {test_metrics['test_accuracy']:.1%}")
    logger.info(f"  Macro F1 score:     {test_metrics['f1_macro']:.1%}")
    logger.info(f"  Avg confidence:     {test_metrics['avg_confidence']:.1%}")
    logger.info(f"  Auto-route rate:    {test_metrics['auto_route_rate_75']:.1%}")
    logger.info(f"  (tickets where model confidence >= 75%)")
    logger.info(f"")
    logger.info(f"  Write down your test accuracy.")
    logger.info(f"  Week 2 DistilBERT will beat it by ~25-30 points.")
    logger.info(f"  That improvement is your resume headline metric.")
    logger.info("=" * 60)


    logger.info("\nDEMO: Predicting a single example ticket")
    logger.info("-" * 40)

    demo_ticket = (
        "I was charged twice this month for my Pro subscription. "
        "I see two charges of $49 on my credit card. "
        "Please refund the duplicate charge immediately."
    )

    # Clean the demo ticket using the same steps as training data
    # The model expects preprocessed text, not raw text
    demo_cleaned = preprocessor.clean_text(demo_ticket)

    # Predict — returns dict with category, confidence, probabilities
    prediction = classifier.predict_single(demo_cleaned)

    logger.info(f"  Ticket:    '{demo_ticket[:70]}...'")
    logger.info(f"  Predicted: {prediction['predicted_category']}")
    logger.info(f"  Confidence:{prediction['confidence']:.1%}")
    logger.info(f"  Auto-routed: {prediction['auto_routed']}")
    logger.info(f"  All category probabilities:")

    # Sort from highest to lowest probability
    sorted_probs = sorted(
        prediction['all_probabilities'].items(),
        key=lambda x: x[1],   # Sort by the probability value (second item in tuple)
        reverse=True,          # Highest probability first
    )

    # Print a simple bar chart for each category
    for category_name, probability in sorted_probs:
        # int(probability * 20) turns 0.91 into 18, giving bar length
        bar = "=" * int(probability * 20)
        logger.info(f"    {category_name:<22} {probability:.1%}  [{bar}]")


    logger.info("\n" + "=" * 60)
    logger.info("NEXT STEPS")
    logger.info("=" * 60)
    logger.info("  1. View your MLflow experiment dashboard:")
    logger.info("       mlflow ui")
    logger.info("       Open: http://localhost:5000")
    logger.info("")
    logger.info("  2. Open the Jupyter notebook:")
    logger.info("       jupyter notebook notebooks/week1_eda.ipynb")
    logger.info("")
    logger.info("  3. Run the test suite:")
    logger.info("       pytest tests/ -v")
    logger.info("")
    logger.info("  4. Commit your work to GitHub:")
    logger.info("       git add .")
    logger.info('       git commit -m "Week 1: baseline classifier complete"')
    logger.info("       git push")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()