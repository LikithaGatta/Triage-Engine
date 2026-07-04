"""
Clean raw ticket text before it is processed by ML model
-> Want TF-IDF to see identical words as identical, not completely different.
"""

import re # text cleaning
import string # string constants 
from typing import List # type annotations

import nltk # natural language processing
import pandas as pd # data manipulation
from tqdm import tqdm # progress bars

from src.utils.logger import logger 
from src.utils.schemas import RawTicket, ProcessedTicket

def download_nltk_resources():
    """Download required NLTK data files if not already present."""
    resources = ["punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger"]
    for name in resources:
        try:
            nltk.download(name, quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK resource {name}: {e}")

from nltk.corpus import stopwords # Provide list of common words that are removed since they have little meaning
from nltk.stem import WordNetLemmatizer # Reduce words to their base form
from nltk.tokenize import word_tokenize # Breaks text into individual words


# Base English stopwords — words that carry no classification signal
STOP_WORDS = set(stopwords.words("english"))

# BUT some stopwords matter for support tickets!
# "not working" and "working" mean very different things.
# We remove these from the stopword list so they are kept.
KEEP_WORDS = {
    "not", "no", "never", "cannot", "down", "broken",
    "wrong", "error", "fail", "failed", "off",
}
STOP_WORDS -= KEEP_WORDS

# Pre-compiled regex patterns (compiled once = faster than recompiling each call)
URL_PATTERN       = re.compile(r"http[s]?://\S+|www\.\S+")
EMAIL_PATTERN     = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")
HTML_PATTERN      = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")

class TicketPreprocessor:
    """
    Cleans and normalizes raw support ticket text for ML.

    Usage:
        preprocessor = TicketPreprocessor()
        df = preprocessor.process_dataframe(raw_df)
    """

    def __init__(self, min_word_count: int = 3):
        """
        Args:
            min_word_count: Tickets with fewer words after cleaning
                           are flagged as too_short = True
        """
        # WordNetLemmatizer reduces words to their root form
        # "charges" → "charge", "running" → "run", "crashed" → "crash"
        self.lemmatizer = WordNetLemmatizer()
        self.min_word_count = min_word_count
        logger.info("TicketPreprocessor ready")

    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning steps to a raw string.

        Args:
            text: Raw text from subject or body field
        Returns:
            Cleaned, lowercased, noise-removed string
        """
        # Handle empty or non-string input gracefully
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Lowercase — "URGENT" and "urgent" are the same word
        text = text.lower()

        # Step 2: Remove HTML tags — "<p>Cannot login</p>" → "Cannot login"
        text = HTML_PATTERN.sub(" ", text)

        # Step 3: Remove URLs — they add noise, not classification signal
        text = URL_PATTERN.sub(" ", text)

        # Step 4: Remove email addresses — privacy + not useful as features
        text = EMAIL_PATTERN.sub(" ", text)

        # Step 5: Remove punctuation
        # We keep apostrophes so "can't" stays as "can't" not "cant"
        punct = string.punctuation.replace("'", "")
        text = text.translate(str.maketrans(punct, " " * len(punct)))

        # Step 6: Collapse multiple spaces into one
        text = WHITESPACE_PATTERN.sub(" ", text).strip()

        return text

    def tokenize_and_filter(self, text: str) -> List[str]:
        """
        Split text into tokens, remove stopwords, lemmatize.

        Args:
            text: Already cleaned text
        Returns:
            List of meaningful root-form words
        """
        # Split text into individual word tokens
        tokens = word_tokenize(text)

        filtered = []
        for token in tokens:
            # Skip stopwords (but keep our KEEP_WORDS)
            if token in STOP_WORDS:
                continue

            # Skip single characters — usually noise
            if len(token) <= 1:
                continue

            # Skip pure numbers — "123" is not a useful feature
            if token.isdigit():
                continue

            # Lemmatize: reduce to root form
            # "charged" → "charge", "payments" → "payment"
            lemma = self.lemmatizer.lemmatize(token)
            filtered.append(lemma)

        return filtered

    def combine_subject_body(self, subject: str, body: str) -> str:
        """
        Merge subject and body into one string.

        We repeat the subject twice because it is a dense signal —
        "URGENT: payment failed" tells us a lot in very few words.
        Repeating it gives it more weight in TF-IDF.
        """
        subject = subject.strip() if subject else ""
        body    = body.strip()    if body    else ""

        if subject:
            # Subject appears twice to give it more weight
            return f"{subject} {body}"
        return body

    def process_ticket(self, ticket: RawTicket) -> ProcessedTicket:
        """
        Run the full preprocessing pipeline on one ticket.

        Args:
            ticket: RawTicket pydantic model
        Returns:
            ProcessedTicket with cleaned text
        """
        # Combine subject + body
        raw_combined = self.combine_subject_body(ticket.subject, ticket.body)

        # Clean the text
        cleaned = self.clean_text(raw_combined)

        # Tokenize, filter stopwords, lemmatize
        tokens = self.tokenize_and_filter(cleaned)

        # Rejoin tokens into a string — TF-IDF expects a string not a list
        processed_text = " ".join(tokens)

        return ProcessedTicket(
            ticket_id=ticket.ticket_id,
            raw_subject=ticket.subject,
            raw_body=ticket.body,
            text=processed_text,
            char_count=len(raw_combined),
            word_count=len(tokens),
            source=ticket.source,
            created_at=ticket.created_at,
        )

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess an entire DataFrame of tickets at once.

        Adds columns to df:
            processed_text: cleaned text for the ML model
            word_count:     words remaining after cleaning
            char_count:     original character count
            too_short:      True if fewer than min_word_count words

        Args:
            df: DataFrame with raw ticket data
        Returns:
            DataFrame with new preprocessing columns added
        """
        logger.info(f"Preprocessing {len(df):,} tickets...")

        processed_texts = []
        word_counts     = []
        char_counts     = []

        # tqdm wraps the loop and shows a progress bar in the terminal
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning tickets"):

            # Handle different possible column names for text content
            body    = str(row.get("body",    row.get("text",    "")))
            subject = str(row.get("subject", row.get("title",   "")))
            tid     = str(row.get("ticket_id", row.name))

            # Create a RawTicket object for this row
            raw = RawTicket(
                ticket_id=tid,
                subject=subject,
                body=body,
                source=str(row.get("source", "csv")),
            )

            # Run the full preprocessing pipeline
            processed = self.process_ticket(raw)

            processed_texts.append(processed.text)
            word_counts.append(processed.word_count)
            char_counts.append(processed.char_count)

        # Add new columns to a copy of the DataFrame
        df = df.copy()
        df["processed_text"] = processed_texts
        df["word_count"]     = word_counts
        df["char_count"]     = char_counts

        # Flag very short tickets — not enough text to classify reliably
        df["too_short"] = df["word_count"] < self.min_word_count

        short_count = df["too_short"].sum()
        if short_count > 0:
            logger.warning(f"{short_count} tickets flagged as too short (< {self.min_word_count} words)")

        logger.info("Preprocessing complete")
        return df