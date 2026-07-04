"""
This file : 
1. Generates realistic synthetic support ticket data for training and testing the triage engine.
2. Loads and splits the dataset into training and testing sets.

- Sort the dataset into 6 categories: billing, bug_report, feature_request, account_access, performance, general, other.
- For each category we have different message templates to generate realistic support tickets.
- Each is filled in with random values to create a diverse dataset. (amounts, product names, error codes, etc.)

"""

# Imports
import random # For generating random values for the synthetic data
from pathlib import Path # For handling file paths
from typing import Dict, Tuple, Optional # For type annotations

import pandas as pd    # For data manipulation and analysis

from sklearn.model_selection import train_test_split # For splitting the dataset into training and testing sets

from src.utils.logger import logger # For detailed reports on the events while application is running
from src.utils.schemas import Category, UrgencyLevel # For defining the categories and urgency levels of support tickets

# Define file paths for data storage
DATA_DIR = Path("data")                
RAW_DIR = DATA_DIR / "raw"            
PROCESSED_DIR = DATA_DIR / "processed"  

class SyntheticTicketGenerator:
    # Creates realistic support ticket data for training our ML model.

    TEMPLATES: Dict[str, list] = {
        # Each category has several ticket templates and then create variations

        # --- BILLING TICKETS ---
        # Tickets related to charges, refunds, invoices, and payment issues, etc.
        Category.BILLING: [
        ("double_charge",
         "I was charged twice for my {plan} subscription. "
         "Two identical charges of ${amount} appeared on my statement. Please refund one."),
        
        ("refund_request",
         "I cancelled {days} days ago but got billed again. This is the {complaint}. "
         "I need my ${amount} back immediately."),
        
        ("wrong_amount",
         "My invoice shows ${wrong_amount} but my plan is ${correct_amount} per month. "
         "Please fix this and refund the difference."),
        
        ("update_payment",
         "My card ending in {last4} expired. How do I update payment before the next cycle?"),
       
        ("invoice_question",
         "Can you send a PDF invoice for my last payment? "
         "My company needs it for expense reporting. Name on invoice: {company}."),
        
        ("unexpected_charge",
         "I have no idea what this ${amount} charge is for. "
         "I do not remember signing up for anything new. Can you explain this?"),
        
        ("downgrade_billing",
         "I downgraded my plan {time_ref} but I am still being charged the {plan} rate. "
         "This needs to be corrected and backdated."),
    ],

        # --- Bug REPORT TICKETS ---
        # Tickets related to software bugs, crashes, errors, and other technical issues that users are experiencing with the product.
        Category.BUG_REPORT: [
        ("crash",
         "The app crashes {time_ref} when I try to {action}. "
         "I am on {platform}. Error message: {error}"),
        
        ("feature_broken",
         "The {feature} stopped working completely. "
         "Clicking it does nothing. Tried different browsers — same issue."),
        
        ("data_not_saving",
         "My {data_type} disappear after saving. I get the green success message "
         "but everything is gone when I refresh. This is {complaint}."),
        
        ("login_broken",
         "Cannot get into my account at all. Password is correct, "
         "reset email never arrives. Tried {days} times now."),
        
        ("api_error",
         "All API calls to your service are returning 500 errors {time_ref}. "
         "This is blocking our entire production deployment. Error: {error}"),
        
        ("display_bug",
         "The numbers on the {page} dashboard are wrong. "
         "They do not match our actual data at all. Looks like a calculation error."),
        
        ("intermittent_bug",
         "Sometimes the {feature} works fine, sometimes it just fails silently. "
         "No error message. Cannot reproduce consistently but it happens daily."),
    ],

 
        # --- FEATURE REQUEST TICKETS ---
        # Tickets related to users requesting new features, improvements, or enhancements to the product.
        Category.FEATURE_REQUEST: [
        ("dark_mode",
         "Any plans for a dark mode? Spending all day in the tool and the bright "
         "white interface is really straining my eyes."),
        
        ("export",
         "We desperately need the ability to export {data_type} to {format}. "
         "Currently copying everything manually which takes hours."),
        
        ("bulk_action",
         "Please add bulk {action} for {items}. Doing it one at a time for 500+ records "
         "is {complaint}."),
        
        ("integration",
         "Does {tool} integration exist? If not, any plans to add it? "
         "It would completely change our workflow."),
        
        ("notification",
         "Would love email or Slack notifications when {event}. "
         "I miss things because I cannot monitor the dashboard all day."),
        
        ("mobile_app",
         "Is there a mobile app or is one planned? "
         "I need to check on things when I am away from my desk."),
        
        ("api_access",
         "We would love API access to {data_type} so we can build our own integrations. "
         "Is this on the roadmap at all?"),
    ],
 
        # --- ACCOUNT ACCESS TICKETS ---
        # Tickets related to users having trouble accessing their accounts, managing team members, or other account-related issues.
        Category.ACCOUNT_ACCESS: [
        ("locked_out",
         "Completely locked out. Have not been able to log in for {days} days. "
         "Reset emails are not coming through."),
        
        ("transfer",
         "Need to move my account to a different email address. "
         "Old email is no longer accessible. What is the process?"),
        
        ("team_access",
         "How do I add a new team member with {role} access? "
         "We are on the {plan} plan and cannot figure out the settings."),
        
        ("two_fa_issue",
         "Lost my authenticator app. Cannot get past 2FA. "
         "Have backup codes but they are not being accepted. This is urgent."),
        
        ("account_merge",
         "I accidentally created two accounts. Can they be merged? "
         "I want to keep all the data from both if possible."),
        
        ("sso_issue",
         "Our SSO login is broken {time_ref}. "
         "The whole team cannot access the platform. This is completely blocking us."),
    ],
 
        # --- PERFORMANCE TICKETS ---
        # Tickets related to the product being slow, timing out, using too much memory, or other performance-related issues that impact the user experience.
         Category.PERFORMANCE: [
        ("slow_loading",
         "The {page} page takes {seconds} seconds to load {time_ref}. "
         "It was instant before. Our whole team is affected."),
        
        ("timeout",
         "Exports keep timing out. Any {data_type} report with more than {count} rows fails. "
         "I need this data urgently."),
        
        ("slow_search",
         "Search results take {seconds} seconds to appear after typing. "
         "Makes the product nearly impossible to use efficiently."),
        
        ("memory_leak",
         "The browser tab uses more and more memory over time and eventually crashes. "
         "Have to refresh every hour or so."),
        
        ("slow_load_charts",
         "Charts and graphs on the analytics page take forever to render. "
         "Sometimes they never load at all."),
    ],
 
        # --- GENERAL TICKETS ---
        # Tickets related to general questions, onboarding, compatibility, how-to's, and other non-urgent inquiries that users have about the product.
       Category.GENERAL: [
        ("general_question",
         "Quick question about {topic} — could not find a clear answer in the docs. "
         "Can someone point me in the right direction?"),
        
        ("onboarding",
         "Just signed up for {plan} and getting started. "
         "Is there an onboarding guide or someone we can talk to?"),
        
        ("compatibility",
         "Does your platform work with {tool}? "
         "Evaluating options right now and this is a key requirement."),
       
        ("how_to",
         "How do I {task}? The documentation is not very clear on this step."),
        
        ("pricing_question",
         "What is included in the {plan} plan exactly? "
         "Trying to figure out if we need to upgrade before committing to annual."),
        
        ("cancellation",
         "How do I cancel my subscription? "
         "I cannot find the option anywhere in the settings."),
    ],
}

    # Urgency rules based on category and issue type to determine how quickly the support team needs to respond to each ticket.
    URGENCY_RULES = {
    Category.BILLING: {
        "double_charge":      UrgencyLevel.HIGH,
        "refund_request":     UrgencyLevel.HIGH,
        "wrong_amount":       UrgencyLevel.NORMAL,
        "update_payment":     UrgencyLevel.NORMAL,
        "invoice_question":   UrgencyLevel.NORMAL,
        "unexpected_charge":  UrgencyLevel.HIGH,
        "downgrade_billing":  UrgencyLevel.NORMAL,
    },
    Category.BUG_REPORT: {
        "crash":              UrgencyLevel.CRITICAL,
        "feature_broken":     UrgencyLevel.HIGH,
        "data_not_saving":    UrgencyLevel.HIGH,
        "login_broken":       UrgencyLevel.CRITICAL,
        "api_error":          UrgencyLevel.CRITICAL,
        "display_bug":        UrgencyLevel.HIGH,
        "intermittent_bug":   UrgencyLevel.HIGH,
    },
    Category.FEATURE_REQUEST: {
        "dark_mode":          UrgencyLevel.NORMAL,
        "export":             UrgencyLevel.NORMAL,
        "bulk_action":        UrgencyLevel.NORMAL,
        "integration":        UrgencyLevel.NORMAL,
        "notification":       UrgencyLevel.NORMAL,
        "mobile_app":         UrgencyLevel.NORMAL,
        "api_access":         UrgencyLevel.NORMAL,
    },
    Category.ACCOUNT_ACCESS: {
        "locked_out":         UrgencyLevel.HIGH,
        "transfer":           UrgencyLevel.NORMAL,
        "team_access":        UrgencyLevel.NORMAL,
        "two_fa_issue":       UrgencyLevel.CRITICAL,
        "account_merge":      UrgencyLevel.NORMAL,
        "sso_issue":          UrgencyLevel.CRITICAL,
    },
    Category.PERFORMANCE: {
        "slow_loading":       UrgencyLevel.HIGH,
        "timeout":            UrgencyLevel.HIGH,
        "slow_search":        UrgencyLevel.HIGH,
        "memory_leak":        UrgencyLevel.HIGH,
        "slow_load_charts":   UrgencyLevel.NORMAL,
    },
    Category.GENERAL: {
        "general_question":   UrgencyLevel.NORMAL,
        "onboarding":         UrgencyLevel.NORMAL,
        "compatibility":      UrgencyLevel.NORMAL,
        "how_to":             UrgencyLevel.NORMAL,
        "pricing_question":   UrgencyLevel.NORMAL,
        "cancellation":       UrgencyLevel.NORMAL,
    },
}
    # Random values to replace the placeholders in the templates.
    FILL_VALUES = {
    "plan":          ["Pro", "Business", "Enterprise", "Starter", "Team", "Basic"],
    "amount":        ["49", "99", "149", "299", "19", "29", "79"],
    "date":          ["January 15", "February 3", "last week", "two weeks ago", "last month"],
    "platform":      ["macOS Ventura", "Windows 11", "iOS 17", "Android 14", "Ubuntu 22"],
    "action":        ["upload a file", "export a report", "save my changes", "run an analysis"],
    "feature":       ["bulk edit", "export", "search", "notifications", "the dashboard", "filters"],
    "data_type":     ["contacts", "reports", "transactions", "projects", "invoices", "records"],
    "format":        ["CSV", "Excel", "PDF", "JSON", "Google Sheets"],
    "items":         ["contacts", "records", "tasks", "transactions", "invoices"],
    "tool":          ["Slack", "HubSpot", "Salesforce", "Zapier", "Notion", "Jira", "QuickBooks"],
    "event":         ["a payment fails", "a report finishes", "a task gets assigned to me"],
    "days":          ["3", "5", "7", "14", "30"],
    "role":          ["admin", "editor", "read-only", "billing manager"],
    "page":          ["dashboard", "analytics", "reports", "settings", "billing"],
    "seconds":       ["10", "15", "20", "30", "45"],
    "count":         ["500", "1000", "5000", "10000"],
    "topic":         ["user permissions", "data export limits", "API rate limits", "billing cycles"],
    "task":          ["export historical data", "set up SSO", "bulk delete old records"],
    "error":         ["500 Internal Server Error", "Request timeout", "Authentication failed"],
    "last4":         ["4242", "1234", "5678", "9012"],
    "wrong_amount":  ["199", "299", "149", "99"],
    "correct_amount":["49", "79", "99", "29"],
    "company":       ["Acme Corp", "Tech Solutions Inc", "Bright Ventures LLC", "NovaCo"],
    "time":          ["14:30", "09:15", "18:45", "23:00"],
    "complaint":     ["absolutely unacceptable", "very frustrating", "the third time this month",
                      "causing us serious problems", "a major issue for our team"],
    "time_ref":      ["since the last update", "starting yesterday", "for the past week",
                      "after we upgraded plans", "randomly throughout the day"],
}


    def _fill_template(self, template: str) -> str:
        """
        Replace all {placeholder} tags in a template string with random values.
        ARGS = string with placeholder, Return = full string with replacements
        """
        result = template  # original template 
 
        # Loop through each placeholder and its possible values
        for key, values in self.FILL_VALUES.items():
            placeholder = "{" + key + "}" 
 
            # Replaces the placeholder that actually appears in the template
            if placeholder in result:
                # Pick a random value from the list for this placeholder
                chosen_value = random.choice(values)
                result = result.replace(placeholder, chosen_value)
 
        return result 
    
    def generate(self, n_per_category: int = 100, random_seed: int = 42) -> pd.DataFrame:
        """
        Generate synthetic tickets.
        ARGS = # of tickets per category, random seed - same seed = same tickets 
        Return = DataFrame with columns: ticket_id, subject, body, category, urgency
        
        1. Select random ticket template
        2. Fill placeholds with fake data
        3. Assign urgency level 
        4. Create a unique ticket ID
        5. Store everything in a list 
        6. Conver list into DataFrame and return

        Final output = table 

        """
        # Create randomness, store in list, and create ticket IDS
        random.seed(random_seed)
        rows = []
        ticket_counter = 1

        for category in Category:
            templates = self.TEMPLATES.get(category, [])
            if not templates:
                continue

            for i in range(n_per_category):
                # pick a random template for the category 
                template_name, template_body = random.choice(templates)
                urgency = self.URGENCY_RULES[category][template_name]
                body = self._fill_template(template_body)

                subject_map = {
                    "double charge": "Charged twice this month",
                    "refund request": "Need refund for cancelled subscription",
                    "crash": "App keeps crashing",
                    "login broken": "Cannot log into my account",
                    "slow loading": "Dashboard is extremely slow",
                    "api error": "API returning 500 errors",
                    "general question": "Question about product features",
                    "onboarding": "Need help getting started",
                    "compatibility": "Does your product work with other tools?",
                    "how to": "How do I perform a specific task?",
                    "feature broken": "A key feature is not working",
                    "data not saving": "My data is not being saved",
                    "export": "Request for data export feature",
                    "bulk action": "Request for bulk action feature",
                    "integration": "Request for integration with other tools",
                    "notification": "Request for email notifications",
                    "locked out": "Locked out of my account",
                    "transfer": "Need to transfer account to new email",
                    "team access": "Need to add team member to account",
                    "two fa issue": "Lost access to two factor authentication",
                    "account merge": "Want to merge two accounts",
                    "slow search": "Search is very slow",
                    "memory leak": "App is using too much memory",
                }
                subject = subject_map.get(template_name.replace("_", " ").title())

                rows.append({
                    "ticket_id": f"SYN-{ticket_counter:05d}",
                    "subject": subject,
                    "body": body,
                    "category": category.value,
                    "urgency": urgency.value,
                    "source": "synthetic",
                })
                ticket_counter += 1

        df = pd.DataFrame(rows)
        logger.info(f"Generated {len(df)} synthetic tickets")
        logger.info(f"Category distribution:\n{df['category'].value_counts().to_string()}")
        return df

class DatasetLoader:
    """
    Loads and prepares the full training dataset.
    Combines synthetic data with any real data available.

    - loading datasets
    - saving datasets
    - splitting train/test
    importing Kaggle data
    """
 
    def __init__(self):
        self.generator = SyntheticTicketGenerator()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
 
    def load_or_generate(self, n_synthetic_per_category: int = 150) -> pd.DataFrame:
        """
        Load dataset from disk if it exists, otherwise generate it.
        If real ticket data exists in data/raw/, mix it with synthetic data.
        """
        processed_path = PROCESSED_DIR / "full_dataset.csv"
        real_data_path = RAW_DIR / "real_tickets.csv"

        if processed_path.exists():
            logger.info(f"Loading existing dataset from {processed_path}")
            df = pd.read_csv(processed_path)
            logger.info(f"Loaded {len(df):,} tickets")
            return df

        logger.info("Generating synthetic data...")
        df_synthetic = self.generator.generate(
            n_per_category=n_synthetic_per_category
        )

        # Mix in real data if available
        if real_data_path.exists():
            logger.info(f"Mixing in real ticket data from {real_data_path}")
            df_real = pd.read_csv(real_data_path)

            # Only keep categories we support
            valid_cats = {c.value for c in Category}
            df_real = df_real[df_real["category"].isin(valid_cats)]

            df = pd.concat([df_synthetic, df_real], ignore_index=True)
            logger.info(f"Combined: {len(df_synthetic)} synthetic + {len(df_real)} real = {len(df)} total")
        else:
            df = df_synthetic
            logger.info("No real data found — using synthetic only")
            logger.info("Run scripts/download_real_data.py to add real tickets")

        df.to_csv(processed_path, index=False)
        logger.info(f"Saved dataset to {processed_path}")
        return df

    def load_kaggle_twitter_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Load and map the Kaggle 'Customer Support on Twitter' dataset.
        Download from: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
 
        The dataset has tweets/replies. We extract customer messages and
        map them to our 6 categories based on keywords.
 
        Args:
            filepath: Path to the downloaded CSV
        Returns:
            DataFrame in our format, or None if file not found
        """
        if not Path(filepath).exists():
            logger.warning(f"Kaggle dataset not found at {filepath}. Using synthetic data only.")
            return None
 
        logger.info(f"Loading Kaggle dataset from {filepath}")
        df = pd.read_csv(filepath)
 
        # The Twitter dataset has specific columns — adapt to our format
        # 'text' column contains the tweet text
        # We take customer messages (not company replies)
        if "text" in df.columns:
            # Filter to customer messages only (not @company replies)
            customer_msgs = df[~df["text"].str.startswith("@", na=True)].copy()
 
            mapped = pd.DataFrame({
                "ticket_id": [f"TWT-{i:06d}" for i in range(len(customer_msgs))],
                "subject": "",
                "body": customer_msgs["text"].fillna(""),
                "source": "twitter_kaggle",
            })
 
            # Auto-label using keyword heuristics
            # (These won't be perfect — that's expected for real data)
            mapped["category"] = mapped["body"].apply(self._heuristic_label)
            mapped["urgency"] = mapped["body"].apply(self._heuristic_urgency)
 
            # Drop rows where heuristics couldn't determine a category
            mapped = mapped[mapped["category"] != "unknown"]
 
            logger.info(f"Loaded {len(mapped):,} Twitter tickets after filtering")
            return mapped
 
        return None
 
    def _heuristic_label(self, text: str) -> str:
        """Simple keyword-based labeling for real data. Not perfect — that's okay."""
        text_lower = str(text).lower()
        if any(w in text_lower for w in ["charge", "bill", "invoice", "refund", "payment", "price"]):
            return Category.BILLING.value
        if any(w in text_lower for w in ["crash", "broken", "bug", "error", "not working", "glitch"]):
            return Category.BUG_REPORT.value
        if any(w in text_lower for w in ["feature", "add", "wish", "would love", "suggestion"]):
            return Category.FEATURE_REQUEST.value
        if any(w in text_lower for w in ["login", "password", "access", "locked", "account"]):
            return Category.ACCOUNT_ACCESS.value
        if any(w in text_lower for w in ["slow", "performance", "lag", "loading", "timeout"]):
            return Category.PERFORMANCE.value
        if any(w in text_lower for w in ["how", "can i", "question", "help", "where"]):
            return Category.GENERAL.value
        return "unknown"
 
    def _heuristic_urgency(self, text: str) -> str:
        """Keyword-based urgency labeling."""
        text_lower = str(text).lower()
        critical_words = ["down", "outage", "cannot login", "data loss", "urgent", "asap", "immediately"]
        high_words = ["broken", "not working", "charged twice", "refund", "crashed"]
        if any(w in text_lower for w in critical_words):
            return UrgencyLevel.CRITICAL.value
        if any(w in text_lower for w in high_words):
            return UrgencyLevel.HIGH.value
        return UrgencyLevel.NORMAL.value
 
    def get_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into training and test sets.
 
        WHY stratify=True?
          Without stratification, the random split might put ALL
          'performance' tickets in training and none in test (if
          there aren't many). stratify ensures each category appears
          in both splits proportionally.
 
        Args:
            df: Full dataset
            test_size: Fraction for test set (0.2 = 80/20 split)
            random_state: Seed for reproducibility
        Returns:
            (train_df, test_df) tuple
        """
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["category"],  # Maintain class proportions in both splits
        )
        logger.info(f"Train: {len(train_df):,} tickets | Test: {len(test_df):,} tickets")
        logger.info(f"Test category distribution:\n{test_df['category'].value_counts().to_string()}")
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
 

