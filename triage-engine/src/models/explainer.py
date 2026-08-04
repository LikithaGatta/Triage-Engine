"""
For every ticket prediction, it produces:
1. Top tokens that drove the prediction (postive SHAP vals)
2. Top tokens that opposed the prediction (negative SHAP vals)
3. Confidence score: base rate + token contributions = final confidence
4. Readable string

SHAP -> shows what actually changed the prediction, measures actual contribution

"""
# Structure dataset and its features
from dataclasses import dataclass, field 
# Type checking 
from typing import Dict, List, Optional, Tuple 
# Data manipulation and operations
import numpy as np 
import pandas as pd 
# Explain predications across models 
import shap 
# Chain into one workflow
from sklearn.pipeline import Pipeline
# Traceable steps and debugging purposes
from src.utils.logger import logger 

# @dataclass generate the automatic standard class structure for _init, _repr_, _eq_
@dataclass
# Represent token and SHAP val
class TokenContribution: 
    token: str
    shap_value: float
    direction: str 

@dataclass
class ExplanationResult:
    ticket_id:          str
    predicted_category: str
    confidence:         float
    base_value:         float
    top_positive:       List[TokenContribution]
    top_negative:       List[TokenContribution]
    all_contributions:  List[TokenContribution]
    explanation_text:   str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization (API responses)."""
        return {
            "ticket_id":          self.ticket_id,
            "predicted_category": self.predicted_category,
            "confidence":         round(self.confidence, 4),
            "base_value":         round(self.base_value, 4),
            "top_positive": [
                {"token": t.token, "shap_value": round(t.shap_value, 4)}
                for t in self.top_positive
            ],
            "top_negative": [
                {"token": t.token, "shap_value": round(t.shap_value, 4)}
                for t in self.top_negative
            ],
            "explanation_text": self.explanation_text,
        }

def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization (API responses)."""
        return {
            "ticket_id":          self.ticket_id,
            "predicted_category": self.predicted_category,
            "confidence":         round(self.confidence, 4),
            "base_value":         round(self.base_value, 4),
            "top_positive": [
                {"token": t.token, "shap_value": round(t.shap_value, 4)}
                for t in self.top_positive
            ],
            "top_negative": [
                {"token": t.token, "shap_value": round(t.shap_value, 4)}
                for t in self.top_negative
            ],
            "explanation_text": self.explanation_text,
        }

class TicketExplainer:
    
    def __init__(
        self,
        pipeline: Pipeline,
        category_names: List[str],
        n_top_tokens: int = 5,
    ):
        self.pipeline        = pipeline
        self.category_names  = category_names
        self.n_top_tokens    = n_top_tokens
        self.shap_explainer  = None    # Initialized in .fit()
        self.background_mean = None    # Average prediction (base value)
        self.is_fitted       = False

        self.vectorizer  = pipeline.named_steps["tfidf"]
        self.classifier  = pipeline.named_steps["classifier"]

        self.feature_names = self.vectorizer.get_feature_names_out()

        logger.info(f"TicketExplainer initialized")
        logger.info(f"Vocabulary size: {len(self.feature_names):,} tokens")
        logger.info(f"Categories: {category_names}")

    def fit(self, background_texts: List[str], n_background: int = 100) -> None:
        """
        Args:
            background_texts: List of preprocessed ticket texts from training set
                              n_background of them for efficiency
            n_background:    How many background samples to use
                             More = more accurate base value, slower fitting
                             
        """
        logger.info(f"Fitting SHAP explainer on {min(n_background, len(background_texts))} background samples...")

        # Sample background texts for efficiency
        # np.random.choice picks n_background items randomly without replacement
        if len(background_texts) > n_background:
            indices    = np.random.choice(len(background_texts), n_background, replace=False)
            background = [background_texts[i] for i in indices]
        else:
            background = background_texts

        # Shape: (n_background, n_features)
        # Each row is one ticket represented as TF-IDF weights
        background_vectors = self.vectorizer.transform(background)

        #SHAP values calculation for linear models
        self.shap_explainer = shap.LinearExplainer(
            self.classifier,
            background_vectors,
            feature_perturbation="interventional",
        )

        # Compute the base value — average prediction across background
        # base_value shape: (n_classes,) — one average per category
        background_preds   = self.classifier.predict_proba(background_vectors)
        self.background_mean = background_preds.mean(axis=0)

        self.is_fitted = True
        logger.info("SHAP explainer fitted successfully")
        logger.info(f"Base values per category: { {cat: round(float(v), 3) for cat, v in zip(self.category_names, self.background_mean)} }")

    def explain(
        self,
        text: str,
        ticket_id: str = "unknown",
        predicted_category: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> ExplanationResult:
        """
        Explain the prediction for a single ticket. Call after each.
        Args:
            text:               Preprocessed ticket text (same as fed to classifier)
            ticket_id:          ID for tracking (appears in ExplanationResult)
            predicted_category: Optional — if not provided, we predict internally
            confidence:         Optional — if not provided, we compute internally
        Returns:
            ExplanationResult with top tokens and explanation text
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .explain()")
        # Get prediction
        if predicted_category is None or confidence is None:
            proba              = self.pipeline.predict_proba([text])[0]
            predicted_idx      = proba.argmax()
            predicted_category = self.category_names[predicted_idx]
            confidence         = float(proba[predicted_idx])

        # Get the index of the predicted category
        try:
            predicted_idx = self.category_names.index(predicted_category)
        except ValueError:
            predicted_idx = 0

   
        text_vector = self.vectorizer.transform([text])
        shap_values = self.shap_explainer.shap_values(text_vector)

        # Extract SHAP values for the predicted class only
        if isinstance(shap_values, list):
            class_shap_values = shap_values[predicted_idx][0]
        else:
            class_shap_values = shap_values[0, :, predicted_idx]

        
        text_array    = text_vector.toarray()[0]  # Convert sparse to dense array
        present_mask  = text_array > 0            # True where word is present

        contributions = []
        for i, (is_present, shap_val) in enumerate(zip(present_mask, class_shap_values)):
            if is_present:
                token     = self.feature_names[i]
                direction = "positive" if shap_val > 0 else "negative"
                contributions.append(TokenContribution(
                    token=token,
                    shap_value=float(shap_val),
                    direction=direction,
                ))

        # Sort by absolute SHAP value — biggest impact first
        contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)

        # Split into top positive and top negative
        top_positive = [c for c in contributions if c.shap_value > 0][:self.n_top_tokens]
        top_negative = [c for c in contributions if c.shap_value < 0][:self.n_top_tokens]

        explanation_text = self._build_explanation_text(
            predicted_category=predicted_category,
            confidence=confidence,
            top_positive=top_positive,
            top_negative=top_negative,
        )

        # Base value for predicted category 
        base_value = float(self.background_mean[predicted_idx])

        return ExplanationResult(
            ticket_id=ticket_id,
            predicted_category=predicted_category,
            confidence=confidence,
            base_value=base_value,
            top_positive=top_positive,
            top_negative=top_negative,
            all_contributions=contributions,
            explanation_text=explanation_text,
        )

    def explain_batch(
        self,
        texts: List[str],
        ticket_ids: Optional[List[str]] = None,
    ) -> List[ExplanationResult]:
        """
        Args:
            texts:      List of preprocessed ticket texts
            ticket_ids: Optional list of IDs matching texts
        Returns:
            List of ExplanationResult, one per ticket
        """
        if ticket_ids is None:
            ticket_ids = [f"ticket_{i}" for i in range(len(texts))]

        text_vectors = self.vectorizer.transform(texts)

        probas = self.pipeline.predict_proba(texts)

        
        shap_values = self.shap_explainer.shap_values(text_vectors)

        results = []
        for i, (text, tid) in enumerate(zip(texts, ticket_ids)):
            predicted_idx      = probas[i].argmax()
            predicted_category = self.category_names[predicted_idx]
            confidence         = float(probas[i][predicted_idx])

            if isinstance(shap_values, list):
                sample_shap = shap_values[predicted_idx][i]
            else:
                sample_shap = shap_values[i, :, predicted_idx]

            text_array   = text_vectors[i].toarray()[0]
            present_mask = text_array > 0

            contributions = []
            for j, (is_present, shap_val) in enumerate(zip(present_mask, sample_shap)):
                if is_present:
                    contributions.append(TokenContribution(
                        token=self.feature_names[j],
                        shap_value=float(shap_val),
                        direction="positive" if shap_val > 0 else "negative",
                    ))

            contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)
            top_positive = [c for c in contributions if c.shap_value > 0][:self.n_top_tokens]
            top_negative = [c for c in contributions if c.shap_value < 0][:self.n_top_tokens]

            explanation_text = self._build_explanation_text(
                predicted_category=predicted_category,
                confidence=confidence,
                top_positive=top_positive,
                top_negative=top_negative,
            )

            results.append(ExplanationResult(
                ticket_id=tid,
                predicted_category=predicted_category,
                confidence=confidence,
                base_value=float(self.background_mean[predicted_idx]),
                top_positive=top_positive,
                top_negative=top_negative,
                all_contributions=contributions,
                explanation_text=explanation_text,
            ))

        return results

    def _build_explanation_text(
        self,
        predicted_category: str,
        confidence: float,
        top_positive: List[TokenContribution],
        top_negative: List[TokenContribution],
    ) -> str:
        """
        Explaination for dashboard
        """
        # Format the positive tokens as a readable list
        pos_tokens = [f'"{t.token}"' for t in top_positive[:3]]
        neg_tokens = [f'"{t.token}"' for t in top_negative[:2]]

        category_display = predicted_category.replace("_", " ")

        if pos_tokens and neg_tokens:
            pos_str = ", ".join(pos_tokens)
            neg_str = ", ".join(neg_tokens)
            return (
                f"Classified as {category_display} ({confidence:.0%} confidence). "
                f"Key signals: {pos_str}. "
                f"Competing signals toward other categories: {neg_str}."
            )
        elif pos_tokens:
            pos_str = ", ".join(pos_tokens)
            return (
                f"Classified as {category_display} ({confidence:.0%} confidence). "
                f"Key signals: {pos_str}."
            )
        else:
            return (
                f"Classified as {category_display} ({confidence:.0%} confidence). "
                f"No strong single-word signals — classification based on overall pattern."
            )

    def get_category_top_features(self, n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
    
        # classifier.coef_ shape: (n_classes, n_features)
        # Each row is one class, each column is one feature (word)
        coefs = self.classifier.coef_

        result = {}
        for class_idx, category in enumerate(self.category_names):
            # Get coefficients for this class
            class_coefs = coefs[class_idx]

            # Get indices of top n features by absolute coefficient value
            top_indices = np.argsort(np.abs(class_coefs))[-n:][::-1]

            # Build list of (token, coefficient) sorted by absolute value
            top_features = [
                (self.feature_names[i], float(class_coefs[i]))
                for i in top_indices
            ]
            result[category] = top_features

        return result

    def save(self, path: str) -> None:
        """Save the fitted explainer to disk using joblib."""
        import joblib
        joblib.dump({
            "shap_explainer":  self.shap_explainer,
            "background_mean": self.background_mean,
            "category_names":  self.category_names,
            "feature_names":   self.feature_names,
            "n_top_tokens":    self.n_top_tokens,
        }, path)
        logger.info(f"Explainer saved to {path}")

    def load(self, path: str) -> None:
        """Load a previously saved explainer from disk."""
        import joblib
        data = joblib.load(path)
        self.shap_explainer  = data["shap_explainer"]
        self.background_mean = data["background_mean"]
        self.category_names  = data["category_names"]
        self.feature_names   = data["feature_names"]
        self.n_top_tokens    = data["n_top_tokens"]
        self.is_fitted       = True
        logger.info(f"Explainer loaded from {path}")

    
       