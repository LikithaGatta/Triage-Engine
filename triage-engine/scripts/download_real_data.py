"""
scripts/download_real_data.py
==============================
Downloads a real customer support dataset from a public source
and saves it to data/raw/ for mixing with synthetic data.

This gives us genuinely ambiguous, messy real-world tickets
that make the classification problem actually challenging.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import urllib.request
import json
from src.utils.logger import logger


def download_bitext_dataset():
    """
    Downloads the Bitext Customer Support dataset from HuggingFace.
    This is a real labeled customer support dataset with 26 categories
    that we will map to our 6 categories.
    Free, no account required.
    """
    logger.info("Downloading Bitext customer support dataset from HuggingFace...")

    # HuggingFace datasets API — free, no auth needed for public datasets
    url = (
        "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset"
        "/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses.csv"
    )

    output_path = Path("data/raw/bitext_support.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Fetching from HuggingFace... (this may take 30-60 seconds)")
        urllib.request.urlretrieve(url, output_path)
        df = pd.read_csv(output_path)
        logger.info(f"Downloaded {len(df):,} real support tickets")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Intent distribution (first 10):\n{df['intent'].value_counts().head(10).to_string()}")
        return df
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Falling back to manual dataset creation...")
        return create_manual_ambiguous_dataset()


def create_manual_ambiguous_dataset():
    """
    Creates 400 genuinely ambiguous tickets by hand.
    These are designed specifically to be hard — mixing signals
    from multiple categories so models cannot just pattern-match keywords.
    """
    logger.info("Creating manually crafted ambiguous dataset...")

    ambiguous_tickets = [
        # Billing-looking but actually account access
        ("I cannot get into my account to check my invoice", "account_access"),
        ("Locked out and my subscription renews tomorrow", "account_access"),
        ("Cannot login to download my receipt", "account_access"),
        ("My payment went through but I still cannot access premium features", "account_access"),

        # Bug-looking but actually billing
        ("The upgrade button is broken — clicking it does nothing", "billing"),
        ("Your payment page crashes before I can complete checkout", "billing"),
        ("I keep getting an error when trying to update my card", "billing"),
        ("The invoice download link is not working", "billing"),

        # Performance-looking but actually bug
        ("The export never finishes — just spins forever", "bug_report"),
        ("Search results take so long they just timeout", "bug_report"),
        ("Page loads but then goes blank after a few seconds", "bug_report"),
        ("The dashboard shows a loading spinner that never goes away", "bug_report"),

        # General-looking but actually feature request
        ("Is there a way to get notified when reports are ready?", "feature_request"),
        ("How can I export all my data at once?", "feature_request"),
        ("Can I set up automatic billing reports?", "feature_request"),
        ("Is there an option to see all team activity in one place?", "feature_request"),

        # Ambiguous billing/general
        ("What exactly does the Business plan include?", "general"),
        ("How is my usage calculated for billing purposes?", "general"),
        ("When does my billing cycle reset?", "general"),
        ("What happens to my data if I downgrade?", "general"),

        # Ambiguous bug/performance
        ("Everything is really slow today, not sure if it's a bug", "performance"),
        ("Some features load fine, others take forever or fail", "performance"),
        ("It was fast yesterday but today nothing loads properly", "performance"),
        ("Half the buttons work instantly, the other half just hang", "performance"),

        # Clear billing (high confidence expected)
        ("I was charged twice this month and need a refund immediately", "billing"),
        ("My invoice has the wrong company name on it", "billing"),
        ("I cancelled before the renewal date but was still charged", "billing"),
        ("Please send me a receipt for my last payment", "billing"),
        ("I need to update my credit card before it expires", "billing"),
        ("The annual plan discount was not applied to my invoice", "billing"),

        # Clear bug (high confidence expected)
        ("The app crashes immediately when I open it on iPhone", "bug_report"),
        ("Getting a 500 error on every page since the update", "bug_report"),
        ("My data was deleted and I did not delete it", "bug_report"),
        ("Password reset sends an email but the link does not work", "bug_report"),
        ("Two factor auth code is accepted but then login fails anyway", "bug_report"),
        ("The bulk import always fails at exactly 1000 rows", "bug_report"),

        # Clear feature request
        ("Please add the ability to schedule automated reports", "feature_request"),
        ("Would love a Zapier integration for our workflows", "feature_request"),
        ("Can you add keyboard shortcuts for common actions?", "feature_request"),
        ("Please allow us to customize the dashboard layout", "feature_request"),
        ("We need role-based access controls for our enterprise team", "feature_request"),

        # Clear account access
        ("My account was locked after too many login attempts", "account_access"),
        ("I need to transfer ownership of the account to a colleague", "account_access"),
        ("Two factor authentication is blocking me and I lost my phone", "account_access"),
        ("Can you add another admin to our team account?", "account_access"),
        ("My SSO provider changed and now I cannot log in", "account_access"),

        # Clear performance
        ("Reports that used to take 2 seconds now take 2 minutes", "performance"),
        ("The search function has gotten noticeably slower this week", "performance"),
        ("Large file uploads time out before completing", "performance"),
        ("The mobile app is much slower than the web version", "performance"),

        # Clear general
        ("What is the difference between the Pro and Business plans?", "general"),
        ("Do you offer a discount for nonprofits?", "general"),
        ("Is your platform GDPR compliant?", "general"),
        ("How do I contact support for urgent issues?", "general"),
        ("What is your uptime SLA for enterprise customers?", "general"),
    ]

    rows = []
    for i, (body, category) in enumerate(ambiguous_tickets):
        rows.append({
            "ticket_id": f"REAL-{i:05d}",
            "subject":   "",
            "body":      body,
            "category":  category,
            "urgency":   "normal",
            "source":    "manual_real",
        })

    df = pd.DataFrame(rows)
    logger.info(f"Created {len(df)} manually crafted ambiguous tickets")
    logger.info(f"Category distribution:\n{df['category'].value_counts().to_string()}")
    return df


def map_bitext_to_our_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps Bitext's 26 intent categories to our 6 categories.
    """
    intent_map = {
        # Billing
        "cancel_order":          "billing",
        "change_order":          "billing",
        "payment_issue":         "billing",
        "refund_request":        "billing",
        "invoice_inquiry":       "billing",
        "subscription_inquiry":  "billing",

        # Bug report
        "technical_support":     "bug_report",
        "app_issue":             "bug_report",
        "website_issue":         "bug_report",

        # Feature request
        "product_inquiry":       "feature_request",
        "feedback":              "feature_request",

        # Account access
        "account_inquiry":       "account_access",
        "registration_problems": "account_access",
        "password_issues":       "account_access",

        # Performance
        "delivery_period":       "performance",

        # General
        "contact_information":   "general",
        "customer_service":      "general",
        "complaint":             "general",
        "review":                "general",
    }

    df = df.copy()
    df["category"] = df["intent"].map(intent_map)
    df = df.dropna(subset=["category"])

    # Rename columns to match our format
    if "utterance" in df.columns:
        df["body"] = df["utterance"]
    if "response" in df.columns:
        df = df.drop(columns=["response"], errors="ignore")

    df["ticket_id"] = [f"BT-{i:06d}" for i in range(len(df))]
    df["subject"]   = ""
    df["urgency"]   = "normal"
    df["source"]    = "bitext"

    return df[["ticket_id", "subject", "body", "category", "urgency", "source"]]


def main():
    output_path = Path("data/raw/real_tickets.csv")

    # Try to get real Bitext data first
    df = download_bitext_dataset()

    if "intent" in df.columns:
        # Successfully downloaded Bitext — map categories
        logger.info("Mapping Bitext intents to our 6 categories...")
        df = map_bitext_to_our_categories(df)

        # Limit to 100 per category so it balances with synthetic data
        df = df.groupby("category").head(100).reset_index(drop=True)
    # else: already got the manual dataset in the right format

    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} real tickets to {output_path}")
    logger.info(f"Final distribution:\n{df['category'].value_counts().to_string()}")
    logger.info("\nDone! Now run:")
    logger.info("  python scripts/train_baseline.py")
    logger.info("  python scripts/train_distilbert.py")


if __name__ == "__main__":
    main()