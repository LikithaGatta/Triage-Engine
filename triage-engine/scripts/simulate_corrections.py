"""
scripts/simulate_corrections.py
=================================
Simulates 50 realistic agent corrections to test the retraining pipeline.

In a real deployment, corrections come from agents using the dashboard.
For portfolio demonstration, we generate realistic corrections
that target the model's actual weak spots — ambiguous tickets where
multiple categories are plausible.

WHY SIMULATE?
  You cannot wait for real agents to use your demo app.
  Simulating corrections lets you demonstrate the full pipeline
  immediately. The simulation is honest — we create genuinely
  ambiguous tickets and assign defensible correct labels.

HOW TO RUN:
    python scripts/simulate_corrections.py

WHAT IT DOES:
    1. Creates 50 ambiguous tickets with agent corrections
    2. Saves them to data/processed/agent_corrections.csv
    3. Prints a summary showing what was corrected and why
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.feedback_store import FeedbackStore
from src.utils.logger import logger


# ----------------------------------------------------------------
# SIMULATED CORRECTIONS
# These are realistic ambiguous tickets where the model is likely
# to make mistakes. Each has a clear correct label and a note
# explaining why the model probably got it wrong.
# ----------------------------------------------------------------

SIMULATED_CORRECTIONS = [
    # Billing tickets that look like account access
    {
        "ticket_id": "SIM-001",
        "subject": "Cannot access my account",
        "body": "I can't get into my account and I'm worried about being charged while locked out. My subscription renews in 3 days.",
        "original_category": "account_access",
        "original_confidence": 0.61,
        "corrected_category": "billing",
        "agent_id": "agent_sarah",
        "correction_note": "Primary concern is the upcoming charge, not the login itself",
    },
    {
        "ticket_id": "SIM-002",
        "subject": "Charged but can't login",
        "body": "I've been charged $99 but I literally cannot login to use anything I paid for. This is unacceptable.",
        "original_category": "account_access",
        "original_confidence": 0.58,
        "corrected_category": "billing",
        "agent_id": "agent_sarah",
        "correction_note": "Billing dispute is the core issue here",
    },
    # Bug reports that look like performance
    {
        "ticket_id": "SIM-003",
        "subject": "Export never finishes",
        "body": "Every time I try to export my contacts list it just spins forever and never completes. This started after the update yesterday.",
        "original_category": "performance",
        "original_confidence": 0.54,
        "corrected_category": "bug_report",
        "agent_id": "agent_mike",
        "correction_note": "This is a bug introduced by the update, not a performance issue",
    },
    {
        "ticket_id": "SIM-004",
        "subject": "Search returns wrong results",
        "body": "The search is really slow AND it returns completely wrong results. Searching for 'John Smith' shows random contacts.",
        "original_category": "performance",
        "original_confidence": 0.51,
        "corrected_category": "bug_report",
        "agent_id": "agent_mike",
        "correction_note": "Wrong results = bug. Slowness is secondary.",
    },
    # Feature requests that look like general questions
    {
        "ticket_id": "SIM-005",
        "subject": "Is there a way to bulk export?",
        "body": "Hi, I was wondering if there's any way to export all my data at once? I've been doing it one page at a time which is very tedious.",
        "original_category": "general",
        "original_confidence": 0.55,
        "corrected_category": "feature_request",
        "agent_id": "agent_priya",
        "correction_note": "They want a feature that doesn't exist yet",
    },
    {
        "ticket_id": "SIM-006",
        "subject": "Can I get notified by email?",
        "body": "I keep missing updates because I have to check the dashboard manually. Is there any email notification option?",
        "original_category": "general",
        "original_confidence": 0.57,
        "corrected_category": "feature_request",
        "agent_id": "agent_priya",
        "correction_note": "Requesting a notification feature that doesn't exist",
    },
    # Account access that looks like bug report
    {
        "ticket_id": "SIM-007",
        "subject": "Two factor auth broken",
        "body": "2FA is completely broken for me. The code says invalid every time even though I'm using the right app. Started today.",
        "original_category": "bug_report",
        "original_confidence": 0.62,
        "corrected_category": "account_access",
        "agent_id": "agent_sarah",
        "correction_note": "2FA access issue — route to account team not engineering",
    },
    {
        "ticket_id": "SIM-008",
        "subject": "SSO login not working",
        "body": "Our company SSO stopped working this morning. Nobody on our team can log in. Error says authentication failed.",
        "original_category": "bug_report",
        "original_confidence": 0.59,
        "corrected_category": "account_access",
        "agent_id": "agent_sarah",
        "correction_note": "SSO configuration issue — account team handles this",
    },
    # Performance that looks like bug report
    {
        "ticket_id": "SIM-009",
        "subject": "Dashboard incredibly slow",
        "body": "The main dashboard now takes over 2 minutes to load. Everything else seems fine. Just the dashboard page.",
        "original_category": "bug_report",
        "original_confidence": 0.52,
        "corrected_category": "performance",
        "agent_id": "agent_mike",
        "correction_note": "This is purely a performance regression, no bug",
    },
    {
        "ticket_id": "SIM-010",
        "subject": "Reports timing out",
        "body": "Running any report with more than 500 rows times out. This used to work fine last month.",
        "original_category": "bug_report",
        "original_confidence": 0.48,
        "corrected_category": "performance",
        "agent_id": "agent_mike",
        "correction_note": "Timeout on large datasets = performance issue",
    },
]

# Generate 40 more corrections by repeating with variations
ADDITIONAL_TICKETS = [
    ("SIM-011", "Billing issue", "I was overcharged this month and need it corrected", "account_access", 0.53, "billing", "agent_sarah"),
    ("SIM-012", "Can't pay", "The payment page crashes when I try to add my card", "bug_report", 0.67, "billing", "agent_sarah"),
    ("SIM-013", "Invoice wrong", "My invoice shows the wrong company name. Need a corrected one for accounting", "general", 0.59, "billing", "agent_priya"),
    ("SIM-014", "Refund needed", "I cancelled within the trial period but was still charged. Please refund", "account_access", 0.54, "billing", "agent_sarah"),
    ("SIM-015", "Page broken", "The analytics page shows all zeros even though we have data", "performance", 0.56, "bug_report", "agent_mike"),
    ("SIM-016", "App crashes", "App crashes immediately when I open the reports section on iOS", "performance", 0.51, "bug_report", "agent_mike"),
    ("SIM-017", "Data missing", "All my contacts from last week disappeared. They were there yesterday", "performance", 0.49, "bug_report", "agent_mike"),
    ("SIM-018", "Login issue", "I reset my password three times and still cannot login", "bug_report", 0.61, "account_access", "agent_sarah"),
    ("SIM-019", "Team access", "How do I give my colleague admin access to our account?", "general", 0.58, "account_access", "agent_priya"),
    ("SIM-020", "Locked out", "My account seems to be locked. Too many failed login attempts", "bug_report", 0.55, "account_access", "agent_sarah"),
    ("SIM-021", "Slow search", "Search takes 30+ seconds to return results. Was instant before", "bug_report", 0.52, "performance", "agent_mike"),
    ("SIM-022", "API slow", "Your API response times have gotten much worse this week. 10+ seconds per call", "bug_report", 0.50, "performance", "agent_mike"),
    ("SIM-023", "Dark mode?", "Any plans to add a dark mode? Would really help with eye strain", "general", 0.61, "feature_request", "agent_priya"),
    ("SIM-024", "Zapier integration", "Do you have a Zapier integration? We really need this for our workflow", "general", 0.59, "feature_request", "agent_priya"),
    ("SIM-025", "Mobile app", "Is there a mobile app planned? Would love to check things on my phone", "general", 0.57, "feature_request", "agent_priya"),
    ("SIM-026", "Pricing question", "What happens to my data if I downgrade to the free plan?", "billing", 0.54, "general", "agent_sarah"),
    ("SIM-027", "How to export", "What file formats can I export data to?", "feature_request", 0.56, "general", "agent_priya"),
    ("SIM-028", "Compliance question", "Is your platform SOC 2 compliant? We need this for our enterprise deal", "feature_request", 0.53, "general", "agent_priya"),
    ("SIM-029", "Double billed", "Two charges appeared on my card this month. Need one refunded urgently", "account_access", 0.58, "billing", "agent_sarah"),
    ("SIM-030", "Card update", "My credit card expired. How do I update my payment info?", "account_access", 0.55, "billing", "agent_sarah"),
    ("SIM-031", "Receipt needed", "Can you send me a receipt for my last three payments for tax purposes?", "general", 0.57, "billing", "agent_priya"),
    ("SIM-032", "Upgrade broken", "The upgrade button on the pricing page is not working at all", "bug_report", 0.63, "billing", "agent_sarah"),
    ("SIM-033", "CSV import fails", "Every time I try to import a CSV file it says invalid format even though the file is correct", "performance", 0.54, "bug_report", "agent_mike"),
    ("SIM-034", "Notifications broken", "I stopped receiving email notifications two weeks ago. Settings look correct", "feature_request", 0.51, "bug_report", "agent_mike"),
    ("SIM-035", "Data not syncing", "My data stopped syncing between devices three days ago", "performance", 0.53, "bug_report", "agent_mike"),
    ("SIM-036", "MFA locked", "Lost my phone and now locked out of MFA. Have backup codes but they expired", "bug_report", 0.60, "account_access", "agent_sarah"),
    ("SIM-037", "Change email", "Need to change the email address on my account. Old one is no longer mine", "bug_report", 0.52, "account_access", "agent_sarah"),
    ("SIM-038", "Bulk delete", "Is there a way to delete multiple records at once instead of one by one?", "general", 0.58, "feature_request", "agent_priya"),
    ("SIM-039", "API access", "We need API access to pull data into our own systems. Is this available?", "general", 0.56, "feature_request", "agent_priya"),
    ("SIM-040", "Slow reports", "Generating any report now takes 5+ minutes. Used to be under 30 seconds", "bug_report", 0.54, "performance", "agent_mike"),
    ("SIM-041", "Charts not loading", "The charts on the analytics page never load. Just a spinner forever", "performance", 0.57, "bug_report", "agent_mike"),
    ("SIM-042", "Plan comparison", "What is the difference between Pro and Business in terms of API limits?", "billing", 0.55, "general", "agent_priya"),
    ("SIM-043", "Cancel subscription", "How do I cancel my subscription? I cannot find the option in settings", "billing", 0.53, "general", "agent_priya"),
    ("SIM-044", "Data export limit", "I hit the export limit but I need more data. Is there a way to increase it?", "general", 0.54, "feature_request", "agent_priya"),
    ("SIM-045", "Wrong charge amount", "I was charged the enterprise rate but I am on the Pro plan", "account_access", 0.56, "billing", "agent_sarah"),
    ("SIM-046", "App freezes", "The mobile app freezes completely after about 10 minutes of use", "performance", 0.55, "bug_report", "agent_mike"),
    ("SIM-047", "Import stuck", "CSV import has been stuck at 50% for two hours. Is it broken?", "performance", 0.52, "bug_report", "agent_mike"),
    ("SIM-048", "Team member access", "Need to add a new billing admin to our account", "bug_report", 0.54, "account_access", "agent_sarah"),
    ("SIM-049", "Password help", "Reset password link expired before I could use it", "bug_report", 0.58, "account_access", "agent_sarah"),
    ("SIM-050", "Keyboard shortcuts", "Would love keyboard shortcuts for common actions. Would save so much time", "general", 0.60, "feature_request", "agent_priya"),
]


def main():
    logger.info("=" * 60)
    logger.info("SIMULATING 50 AGENT CORRECTIONS")
    logger.info("=" * 60)

    store = FeedbackStore()

    # Check if we already have corrections
    existing = store.count()
    if existing > 0:
        logger.info(f"Found {existing} existing corrections. Adding to them.")

    # Save the detailed corrections first
    count = 0
    for c in SIMULATED_CORRECTIONS:
        store.save_correction(
            ticket_id=c["ticket_id"],
            body=c["body"],
            subject=c["subject"],
            original_category=c["original_category"],
            original_confidence=c["original_confidence"],
            corrected_category=c["corrected_category"],
            corrected_urgency="normal",
            agent_id=c["agent_id"],
            correction_note=c.get("correction_note", ""),
        )
        count += 1

    # Save the additional corrections
    for item in ADDITIONAL_TICKETS:
        tid, subj, body, orig_cat, orig_conf, corr_cat, agent = item
        store.save_correction(
            ticket_id=tid,
            body=body,
            subject=subj,
            original_category=orig_cat,
            original_confidence=orig_conf,
            corrected_category=corr_cat,
            corrected_urgency="normal",
            agent_id=agent,
            correction_note="",
        )
        count += 1

    logger.info(f"\nSaved {count} corrections to {store.path}")
    logger.info(f"Total corrections now: {store.count()}")

    # Show summary
    import pandas as pd
    df = store.load_corrections()
    logger.info(f"\nCorrection category distribution:")
    logger.info(f"{df['corrected_category'].value_counts().to_string()}")
    logger.info(f"\nTop correction patterns (model wrong → agent corrected to):")
    patterns = df.groupby(["original_category", "corrected_category"]).size().reset_index(name="count")
    patterns = patterns.sort_values("count", ascending=False)
    logger.info(f"{patterns.head(8).to_string(index=False)}")
    logger.info(f"\nRun scripts/retrain_from_feedback.py to retrain the model")


if __name__ == "__main__":
    main()