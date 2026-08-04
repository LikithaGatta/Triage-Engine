import sys
from pathlib import Path

from sklearn import pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
from dotenv import load_dotenv

from src.data.preprocessor import TicketPreprocessor
from src.models.explainer import TicketExplainer
from src.utils.logger import logger

load_dotenv()

def main():
    logger.info("=" * 60)
    logger.info("WEEK 3: SHAP EXPLAINABILITY LAYER")
    logger.info("=" * 60)

    logger.info("\n[Step 1 of 4] Loading trained model and data...")

    model_path = Path("models/baseline_v1.0.0.joblib")
    train_path = Path("data/processed/train.csv")
    test_path  = Path("data/processed/test.csv")

    if not model_path.exists():
        logger.error("No trained model found. Run scripts/train_baseline.py first.")
        sys.exit(1)
    
    pipeline = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    logger.info(f"Loaded {len(train_df)} train, {len(test_df)} test tickets")

    category_names = list(pipeline.classes_)
    logger.info(f"Categories: {category_names}")

    logger.info("\n[Step 2 of 4] Fitting SHAP explainer...")

    explainer = TicketExplainer(
        pipeline=pipeline,
        category_names=category_names,
        n_top_tokens=5,  
    )

    train_texts = train_df["processed_text"].fillna("").tolist()

    explainer.fit(
        background_texts=train_texts,
        n_background=100, 
    )

    explainer_path = "models/shap_explainer.joblib"
    explainer.save(explainer_path)


    logger.info("\n[Step 3 of 4] Global feature importance per category...")
    logger.info("These are the words most associated with each category:")
    logger.info("(From Logistic Regression coefficients — not SHAP values)")
    logger.info("-" * 50)

    global_features = explainer.get_category_top_features(n=8)

    for category, features in global_features.items():
        logger.info(f"\n  {category.upper().replace('_', ' ')}")
        pos = [(t, v) for t, v in features if v > 0][:5]
        neg = [(t, v) for t, v in features if v < 0][:3]

        if pos:
                pos_str = ", ".join([f'"{t}" (+{v:.2f})' for t, v in pos])
                logger.info(f"    Toward:  {pos_str}")
        if neg:
                neg_str = ", ".join([f'"{t}" ({v:.2f})' for t, v in neg])
                logger.info(f"    Against: {neg_str}")

    logger.info("\n[Step 4 of 4] Explaining test set predictions...")
    test_texts = test_df["processed_text"].fillna("").tolist()
    test_ids   = test_df["ticket_id"].astype(str).tolist()

    explanations = explainer.explain_batch(test_texts, ticket_ids=test_ids)
    strong_explanations = sum(
        1 for e in explanations
        if len(e.top_positive) >= 2 and e.top_positive[0].shap_value > 0.05
    )
    logger.info(f"Explained {len(explanations)} test tickets")
    logger.info(f"Strong explanations (>=2 clear signals): {strong_explanations} ({strong_explanations/len(explanations):.0%})")

    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Detailed explanations on real-sounding tickets")
    logger.info("=" * 60)

    preprocessor = TicketPreprocessor()

    demo_tickets = [
        {
            "id":   "DEMO-001",
            "text": "I was charged twice this month for my Pro subscription. "
                    "Please refund the duplicate charge immediately.",
            "note": "Clear billing — expect high confidence and strong signals"
        },
        {
            "id":   "DEMO-002",
            "text": "I cannot login and my subscription renews tomorrow. "
                    "I really need access before I get charged again.",
            "note": "AMBIGUOUS — billing + account_access signals mixed"
        },
        {
            "id":   "DEMO-003",
            "text": "The export feature crashes every time I try to download "
                    "more than 1000 rows. This is blocking our reporting.",
            "note": "AMBIGUOUS — bug_report + performance signals mixed"
        },
        {
            "id":   "DEMO-004",
            "text": "Would love a dark mode. Also the page loads really slowly "
                    "on my machine after the recent update.",
            "note": "AMBIGUOUS — feature_request + performance signals mixed"
        },
        {
            "id":   "DEMO-005",
            "text": "The app is completely broken. Nothing works. "
                    "500 errors everywhere. This started an hour ago.",
            "note": "Clear bug — expect high confidence"
        },
    ]


    for demo in demo_tickets:
        cleaned = preprocessor.clean_text(demo["text"])
        tokens  = preprocessor.tokenize_and_filter(cleaned)
        processed = " ".join(tokens)

        result = explainer.explain(
            text=processed,
            ticket_id=demo["id"],
        )

        logger.info(f"\n{'─'*55}")
        logger.info(f"Ticket {demo['id']}: {demo['note']}")
        logger.info(f"Text: '{demo['text'][:70]}...'")
        logger.info(f"")
        logger.info(f"  Predicted:  {result.predicted_category}")
        logger.info(f"  Confidence: {result.confidence:.1%}")
        logger.info(f"  Base rate:  {result.base_value:.1%} (avg across all tickets)")
        logger.info(f"")

        if result.top_positive:
            logger.info(f"  DROVE prediction toward {result.predicted_category}:")
            for token in result.top_positive:
                bar = "+" * max(1, int(abs(token.shap_value) * 50))
                logger.info(f"    '{token.token:<20}' {bar} +{token.shap_value:.4f}")

        if result.top_negative:
            logger.info(f"  PULLED AWAY from {result.predicted_category}:")
            for token in result.top_negative:
                bar = "-" * max(1, int(abs(token.shap_value) * 50))
                logger.info(f"    '{token.token:<20}' {bar} {token.shap_value:.4f}")

        logger.info(f"")
        logger.info(f"  Summary: {result.explanation_text}")


    logger.info("\n" + "=" * 60)
    logger.info("WEEK 3 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Explainer saved to: {explainer_path}")
    logger.info(f"  Test tickets explained: {len(explanations)}")
    logger.info(f"  Strong explanations:    {strong_explanations} ({strong_explanations/len(explanations):.0%})")
    logger.info(f"")
    logger.info(f"  RESUME BULLET (add to Week 2 bullet):")
    logger.info(f"  'Added SHAP explainability layer surfacing top-5 token")
    logger.info(f"  contributions per prediction, enabling support managers")
    logger.info(f"  to audit routing decisions and identify systematic errors.'")
    logger.info(f"")
    logger.info(f"  NEXT STEPS:")
    logger.info(f"  1. Commit to GitHub")
    logger.info(f"  2. The explanation results feed directly into the")
    logger.info(f"     Week 4 FastAPI — every /predict endpoint will")
    logger.info(f"     return SHAP tokens alongside the prediction.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

 


