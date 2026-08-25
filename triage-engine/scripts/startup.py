"""
scripts/startup.py
==================
Runs on Render startup. Downloads NLTK data and trains models if needed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Download NLTK data FIRST before any other imports that need it
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from src.utils.logger import logger

def ensure_models_exist():
    model_path     = Path("models/baseline_v1.0.0.joblib")
    explainer_path = Path("models/shap_explainer.joblib")

    if not model_path.exists():
        logger.info("No model found — training baseline...")
        from src.data.loader import DatasetLoader
        from src.data.preprocessor import TicketPreprocessor
        from src.models.baseline_classifier import BaselineClassifier

        loader = DatasetLoader()
        df = loader.load_or_generate(n_synthetic_per_category=150)
        preprocessor = TicketPreprocessor()
        df = preprocessor.process_dataframe(df)
        df = df[~df["too_short"]]
        train_df, test_df = loader.get_train_test_split(df)
        clf = BaselineClassifier()
        clf.train(train_df)
        logger.info("Baseline model trained")

    if not explainer_path.exists():
        logger.info("No explainer found — training SHAP explainer...")
        import joblib
        import pandas as pd
        from src.models.explainer import TicketExplainer

        pipeline = joblib.load(model_path)
        train_df = pd.read_csv("data/processed/train.csv")
        explainer = TicketExplainer(
            pipeline=pipeline,
            category_names=list(pipeline.classes_),
            n_top_tokens=5,
        )
        explainer.fit(train_df["processed_text"].fillna("").tolist())
        explainer.save(str(explainer_path))
        logger.info("SHAP explainer trained")

    logger.info("All models ready")

if __name__ == "__main__":
    ensure_models_exist()