
from pathlib import Path
from typing import Dict, Optional

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.utils.logger import logger
from src.utils.schemas import Category

# Where we save trained model files
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


class BaselineClassifier:
    """
    TF-IDF + Logistic Regression classifier wrapped in an sklearn Pipeline.

    Usage:
        clf = BaselineClassifier()
        clf.train(train_df)
        metrics = clf.evaluate(test_df)
        result = clf.predict_single("I was charged twice")
    """

    def __init__(self, model_version: str = "1.0.0"):
        self.model_version = model_version
        self.is_trained    = False

        # ---- BUILD THE PIPELINE ----
        # A Pipeline chains steps so they run in sequence.
        # Step 1: TF-IDF converts text to number vectors
        # Step 2: Logistic Regression classifies those vectors
        self.pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    # Use single words AND two-word pairs
                    # "not working" as a pair is more informative than separately
                    ngram_range=(1, 2),

                    # Ignore words in more than 90% of tickets — too common to be useful
                    max_df=0.90,

                    # Ignore words in fewer than 2 tickets — too rare to generalize
                    min_df=2,

                    # Cap vocabulary at 15,000 words to control memory
                    max_features=15_000,

                    # Use log scaling so "error" appearing 10x is not
                    # 10x more important than "error" appearing once
                    sublinear_tf=True,
                )
            ),
            (
                "classifier",
                LogisticRegression(
                    # C controls regularization strength
                    # Higher C = fits training data more closely
                    # Lower C = more conservative, better generalization
                    C=5.0,

                    # lbfgs works well for multi-class problems
                    solver="lbfgs",

                    # Increase if you see "did not converge" warning
                    max_iter=1000,

                    # For reproducibility
                    random_state=42,

                    # "balanced" weights minority classes higher
                    # so the model does not ignore rare categories
                    class_weight="balanced",
                )
            ),
        ])

    def train(
        self,
        train_df: pd.DataFrame,
        text_col: str = "processed_text",
        label_col: str = "category",
        experiment_name: str = "triage_baseline",
    ) -> Dict:
        """
        Train the classifier and log the run to MLflow.

        MLflow records every training run with its parameters and metrics.
        Run "mlflow ui" in the terminal to see a dashboard of all runs.

        Args:
            train_df:        DataFrame with training tickets
            text_col:        Column name containing preprocessed text
            label_col:       Column name containing category labels
            experiment_name: MLflow experiment to log under
        Returns:
            Dict with training metrics
        """
        logger.info(f"Training on {len(train_df):,} tickets...")

        # Extract the text and labels as Python lists
        X_train = train_df[text_col].tolist()    # List of cleaned ticket texts
        y_train = train_df[label_col].tolist()   # List of category strings

        # Set up the MLflow experiment (creates it if it does not exist)
        mlflow.set_experiment(experiment_name)

        # Everything inside this block gets logged to MLflow
        with mlflow.start_run(run_name=f"baseline_v{self.model_version}"):

            # Log the hyperparameters we used
            # These show up in the MLflow UI so we can compare runs
            mlflow.log_params({
                "model_type":    "TF-IDF + LogisticRegression",
                "ngram_range":   "(1, 2)",
                "max_features":  15_000,
                "C":             5.0,
                "class_weight":  "balanced",
                "n_train":       len(X_train),
                "model_version": self.model_version,
            })

            # FIT the pipeline
            # This runs TF-IDF.fit_transform() then LogisticRegression.fit()
            logger.info("Fitting TF-IDF + Logistic Regression pipeline...")
            self.pipeline.fit(X_train, y_train)
            self.is_trained = True

            # Quick check on training accuracy
            # (Real accuracy measured on test set in .evaluate())
            train_preds    = self.pipeline.predict(X_train)
            train_accuracy = (pd.Series(train_preds) == pd.Series(y_train)).mean()

            mlflow.log_metric("train_accuracy", train_accuracy)
            logger.info(f"Train accuracy: {train_accuracy:.1%} (not the real metric — see test accuracy)")

            # Save model to disk
            model_path = MODELS_DIR / f"baseline_v{self.model_version}.joblib"
            joblib.dump(self.pipeline, model_path)
            mlflow.log_artifact(str(model_path))
            logger.info(f"Model saved to {model_path}")

        return {"train_accuracy": train_accuracy, "n_samples": len(X_train)}

    def evaluate(
        self,
        test_df: pd.DataFrame,
        text_col: str = "processed_text",
        label_col: str = "category",
    ) -> Dict:
        """
        Evaluate the trained model on the held-out test set.

        This is where we get the REAL accuracy numbers.
        The test set was never seen during training, so these
        numbers reflect how the model will perform on new tickets.

        Args:
            test_df:   DataFrame with test tickets
            text_col:  Column with preprocessed text
            label_col: Column with true category labels
        Returns:
            Dict of all evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation. Call .train() first.")

        X_test = test_df[text_col].tolist()
        y_test = test_df[label_col].tolist()

        logger.info(f"Evaluating on {len(X_test):,} test tickets...")

        # Get the model's predictions
        y_pred = self.pipeline.predict(X_test)

        # Get probability scores for every category
        # Shape: (n_tickets, n_categories)
        # Each row sums to 1.0 — the model's confidence across categories
        y_proba = self.pipeline.predict_proba(X_test)

        # ---- CORE METRICS ----

        # Accuracy: fraction of tickets correctly classified
        accuracy = (pd.Series(y_pred) == pd.Series(y_test)).mean()

        # F1 macro: average F1 score across all categories
        # Macro means each category is weighted equally
        # (good when you care about all categories equally)
        f1_macro        = f1_score(y_test, y_pred, average="macro")
        precision_macro = precision_score(y_test, y_pred, average="macro")
        recall_macro    = recall_score(y_test, y_pred, average="macro")

        # Full per-category breakdown
        report_str = classification_report(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        logger.info(f"\n{'='*50}\nCLASSIFICATION REPORT\n{'='*50}\n{report_str}")
        logger.info(f"Overall accuracy: {accuracy:.1%}")
        logger.info(f"Macro F1 score:   {f1_macro:.1%}")

        # ---- CONFIDENCE ANALYSIS ----

        # For each ticket, the highest probability across all categories
        max_probas = y_proba.max(axis=1)

        # Average confidence — how sure is the model on average?
        avg_confidence = float(max_probas.mean())

        # Auto-route rate — what fraction would we route automatically?
        # We only auto-route when confidence >= 0.75 (our threshold)
        auto_route_rate = float((max_probas >= 0.75).mean())

        logger.info(f"Avg confidence:   {avg_confidence:.1%}")
        logger.info(f"Auto-route rate:  {auto_route_rate:.1%} of tickets have confidence >= 75%")

        # Collect all metrics into one dict
        metrics = {
            "test_accuracy":      float(accuracy),
            "f1_macro":           float(f1_macro),
            "precision_macro":    float(precision_macro),
            "recall_macro":       float(recall_macro),
            "avg_confidence":     avg_confidence,
            "auto_route_rate_75": auto_route_rate,
            "n_test_samples":     len(X_test),
        }

        # Add per-category F1 scores
        for cat in Category:
            if cat.value in report_dict:
                metrics[f"f1_{cat.value}"] = report_dict[cat.value]["f1-score"]

        # Save the classification report as a text file
        report_path = MODELS_DIR / "classification_report.txt"
        report_path.write_text(report_str)
        logger.info(f"Classification report saved to {report_path}")

        return metrics

    def predict_single(self, text: str) -> Dict:
        """
        Classify one ticket text and return the prediction.

        This is what runs in real-time when a ticket comes in.

        Args:
            text: Preprocessed ticket text
        Returns:
            Dict with predicted_category, confidence, all_probabilities, auto_routed
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call .train() first.")

        # predict_proba returns array of shape (1, n_classes)
        # [0] gets the first (only) row as a 1D array
        proba   = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_           # List of category names in order

        # Build a dict mapping category name to its probability
        proba_dict = {cls: float(prob) for cls, prob in zip(classes, proba)}

        # The predicted category is the one with the highest probability
        predicted = classes[proba.argmax()]
        confidence = float(proba.max())

        return {
            "predicted_category":  predicted,
            "confidence":          confidence,
            "all_probabilities":   proba_dict,
            # Auto-route if confidence is high enough that we trust it
            "auto_routed":         confidence >= 0.75,
        }

    def load(self, model_path: Optional[str] = None) -> None:
        """Load a previously saved model from disk."""
        if model_path is None:
            model_path = str(MODELS_DIR / f"baseline_v{self.model_version}.joblib")
        self.pipeline   = joblib.load(model_path)
        self.is_trained = True
        logger.info(f"Model loaded from {model_path}")