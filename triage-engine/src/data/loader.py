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
             "I was charged twice for my {plan} subscription this month. "
             "I see two charges of ${amount} on my statement. "
             "Please issue a refund for one of them."),
 
            ("refund_request",
             "I cancelled my subscription on {date} but was still charged ${amount}. "
             "I need a full refund processed immediately."),
 
            ("wrong_amount",
             "My invoice shows ${wrong_amount} but I should only be on the {plan} plan "
             "at ${correct_amount} per month. Please correct this and refund the difference."),
 
            ("update_payment",
             "I need to update my credit card on file. "
             "The old card ending in {last4} expired. "
             "How do I update my payment method?"),
 
            ("invoice_question",
             "Can you send me an invoice for my last payment? "
             "I need it for expense reporting with company name: {company}."),
        ],

        # --- Bug REPORT TICKETS ---
        # Tickets related to software bugs, crashes, errors, and other technical issues that users are experiencing with the product.
        Category.BUG_REPORT: [
            ("crash",
             "The app crashes every time I try to {action}. "
             "I am on {platform} version {version}. "
             "This started after the last update. Error message: {error}"),
 
            ("feature_broken",
             "The {feature} feature is completely broken. "
             "When I click it nothing happens. "
             "I have tried clearing cache and logging out. Same issue on Chrome and Firefox."),
 
            ("data_not_saving",
             "My {data_type} are not being saved. "
             "I fill out the form, click save, get a success message, "
             "but when I refresh the page nothing is there."),
 
            ("login_broken",
             "Cannot log in at all. "
             "I enter my password, it says incorrect, but I know it is right. "
             "Password reset email never arrives. Tried four times now."),
 
            ("api_error",
             "Getting 500 internal server error on all API calls to the endpoint. "
             "This is blocking our entire integration. "
             "Started happening at approximately {time} UTC today."),
        ],
 
        # --- FEATURE REQUEST TICKETS ---
        # Tickets related to users requesting new features, improvements, or enhancements to the product.
        Category.FEATURE_REQUEST: [
            ("dark_mode",
             "Would love a dark mode option for the dashboard. "
             "Spending 8 hours a day in the tool and the bright white background strains my eyes."),
 
            ("export",
             "Please add the ability to export {data_type} to {format}. "
             "Currently I have to manually copy everything which takes hours each month."),
 
            ("bulk_action",
             "It would save so much time if I could {action} multiple {items} at once "
             "instead of one by one. Our team manages over 500 {items}."),
 
            ("integration",
             "Do you have an integration with {tool}? "
             "We use it heavily and switching between apps is a major workflow problem."),
 
            ("notification",
             "Please add email notifications when {event}. "
             "I keep missing important updates because I have to manually check."),
        ],
 
        # --- ACCOUNT ACCESS TICKETS ---
        # Tickets related to users having trouble accessing their accounts, managing team members, or other account-related issues.
        Category.ACCOUNT_ACCESS: [
            ("locked_out",
             "I am locked out of my account. "
             "I have not logged in for {days} days and now it will not accept my password. "
             "The reset email never comes through."),
 
            ("transfer",
             "I need to transfer my account to a new email address. "
             "My old email is no longer accessible. "
             "Can you help me update it?"),
 
            ("team_access",
             "I need to add a new team member with {role} permissions. "
             "We are on the {plan} plan. How do I do this?"),
 
            ("two_fa_issue",
             "I lost access to my authenticator app and cannot get past two factor authentication. "
             "I have my backup codes but the field will not accept them. Please help urgently."),
 
            ("account_merge",
             "I have two accounts with different emails and want to merge them into one. "
             "Is this possible? I would like to keep all data from both accounts."),
        ],
 
        # --- PERFORMANCE TICKETS ---
        # Tickets related to the product being slow, timing out, using too much memory, or other performance-related issues that impact the user experience.
        Category.PERFORMANCE: [
            ("slow_loading",
             "The {page} page is taking {seconds} seconds to load. "
             "It was instant before. This is severely impacting our whole team's productivity."),
 
            ("timeout",
             "Exports are timing out. Every time I try to export {data_type} "
             "with more than {count} records it spins for a few minutes then fails."),
 
            ("slow_search",
             "Search is extremely slow. "
             "Typing in the search box takes {seconds} seconds to show results. "
             "Makes the product nearly unusable for our daily workflow."),
 
            ("memory_leak",
             "The browser tab for your app uses several gigabytes of RAM "
             "after being open for a few hours. It slows our entire machine down."),
        ],
 
        # --- GENERAL TICKETS ---
        # Tickets related to general questions, onboarding, compatibility, how-to's, and other non-urgent inquiries that users have about the product.
        Category.GENERAL: [
            ("general_question",
             "I have a question about {topic}. "
             "Can you point me to the right resource or connect me with someone who can help?"),
 
            ("onboarding",
             "We just signed up for the {plan} plan and are getting started. "
             "Do you have onboarding resources or can we schedule a call with your team?"),
 
            ("compatibility",
             "Does your product work with {tool}? "
             "We are evaluating vendors and this is an important requirement for us."),
 
            ("how_to",
             "How do I {task}? "
             "I have looked in the documentation but cannot find a clear answer."),
        ],
    }

    # Urgency rules based on category and issue type to determine how quickly the support team needs to respond to each ticket.
    URGENCY_RULES = {
        Category.BILLING: {
            "double_charge":     UrgencyLevel.HIGH,     
            "refund_request":    UrgencyLevel.HIGH,     
            "wrong_amount":      UrgencyLevel.NORMAL,   
            "update_payment":    UrgencyLevel.NORMAL,   
            "invoice_question":  UrgencyLevel.NORMAL,   
        },
        Category.BUG_REPORT: {
            "crash":             UrgencyLevel.CRITICAL,  
            "feature_broken":    UrgencyLevel.HIGH,      
            "data_not_saving":   UrgencyLevel.HIGH,      
            "login_broken":      UrgencyLevel.CRITICAL, 
            "api_error":         UrgencyLevel.CRITICAL,  
        },
        Category.FEATURE_REQUEST: {
            "dark_mode":         UrgencyLevel.NORMAL,   
            "export":            UrgencyLevel.NORMAL,   
            "bulk_action":       UrgencyLevel.NORMAL,   
            "integration":       UrgencyLevel.NORMAL,   
            "notification":      UrgencyLevel.NORMAL,  
        },
        Category.ACCOUNT_ACCESS: {
            "locked_out":        UrgencyLevel.HIGH,      
            "transfer":          UrgencyLevel.NORMAL,   
            "team_access":       UrgencyLevel.NORMAL,    
            "two_fa_issue":      UrgencyLevel.CRITICAL,  
            "account_merge":     UrgencyLevel.NORMAL,    
        },
        Category.PERFORMANCE: {
            "slow_loading":      UrgencyLevel.HIGH,      
            "timeout":           UrgencyLevel.HIGH,      
            "slow_search":       UrgencyLevel.HIGH,      
            "memory_leak":       UrgencyLevel.HIGH,      
        },
        Category.GENERAL: {
            "general_question":  UrgencyLevel.NORMAL,
            "onboarding":        UrgencyLevel.NORMAL,
            "compatibility":     UrgencyLevel.NORMAL,
            "how_to":            UrgencyLevel.NORMAL,
        },
    }
    # Random values to replace the placeholders in the templates.
    FILL_VALUES = {
        "plan":          ["Pro", "Business", "Enterprise", "Starter", "Team"],
        "amount":        ["49", "99", "149", "299", "19", "29"],
        "date":          ["January 15", "February 3", "last week", "two weeks ago"],
        "platform":      ["macOS", "Windows 11", "iOS 17", "Android", "Ubuntu"],
        "version":       ["2.4.1", "3.0.0", "1.9.5", "latest"],
        "action":        ["upload a file", "export data", "save changes", "run a report"],
        "feature":       ["bulk edit", "export", "search", "notifications", "dashboard"],
        "data_type":     ["contacts", "reports", "projects", "invoices", "files"],
        "format":        ["CSV", "Excel", "PDF", "JSON"],
        "items":         ["contacts", "records", "tasks", "projects"],
        "tool":          ["Slack", "HubSpot", "Salesforce", "Zapier", "Notion", "Jira"],
        "event":         ["a payment fails", "a new comment is added", "a task is assigned"],
        "days":          ["30", "60", "90", "14"],
        "role":          ["admin", "editor", "viewer", "billing manager"],
        "page":          ["dashboard", "reports", "analytics", "settings", "contacts"],
        "seconds":       ["10", "15", "30", "45", "60"],
        "count":         ["1000", "5000", "10000", "500"],
        "topic":         ["pricing", "data export", "user permissions", "security", "the API"],
        "task":          ["bulk delete records", "set up SSO", "configure webhooks"],
        "error":         ["500 Internal Server Error", "Connection refused", "Timeout"],
        "last4":         ["4242", "1234", "5678", "9012"],
        "wrong_amount":  ["199", "299", "149"],
        "correct_amount":["49", "99", "79"],
        "company":       ["Acme Corp", "Tech Solutions Inc", "Global Ventures LLC"],
        "time":          ["14:30", "09:15", "18:45"],
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
        Load existing processed data if available, otherwise generate.
 
        Args:
            n_synthetic_per_category: Synthetic tickets per category
        Returns:
            Full dataset DataFrame
        """
        processed_path = PROCESSED_DIR / "full_dataset.csv"
 
        if processed_path.exists():
            logger.info(f"Loading existing dataset from {processed_path}")
            df = pd.read_csv(processed_path)
            logger.info(f"Loaded {len(df):,} tickets")
            return df
 
        logger.info("No existing dataset found. Generating synthetic data...")
        df = self.generator.generate(n_per_category=n_synthetic_per_category)
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
 

