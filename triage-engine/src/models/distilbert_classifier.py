
# IMPORTS

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch

# HuggingFace Transformers: the library that gives us DistilBERT
# AutoTokenizer: loads the right tokenizer for whatever model we name
# AutoModelForSequenceClassification: loads DistilBERT + classification head
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# HuggingFace datasets — wraps pandas data into Trainer format
from datasets import Dataset

# evaluation metrics
import evaluate 

from src.utils.logger import logger
from src.utils.schemas import Category


# The pretrained model 
MODEL_NAME = "distilbert-base-uncased"

# Map category strings to integers — the model outputs numbers not strings
CATEGORY_TO_ID = {
    "billing":          0,
    "bug_report":       1,
    "feature_request":  2,
    "account_access":   3,
    "performance":      4,
    "general":          5,
}

# Reverse mapping — converts model output integers back to category strings
ID_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_ID.items()}

# Number of categories the model needs to predict
NUM_LABELS = len(CATEGORY_TO_ID)  

# Where we save the fine-tuned model
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

class DistilBertClassifier:

    def __init__(self, model_version: str = "2.0.0"):
        
        # Initialize the classifier. 

        self.model_version = model_version
        self.tokenizer     = None 
        self.model         = None   
        self.is_trained    = False

        # Detect which hardware to use for training
     
        # 1. Apple GPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple Silicon GPU (MPS) for training")
        # 2. NVIDIA GPU
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using NVIDIA GPU (CUDA) for training")
        # 3. CPU fallback 
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for training (slower — consider reducing epochs if too slow)")

    def _load_model_and_tokenizer(self):
        
        logger.info(f"Loading tokenizer from HuggingFace: {MODEL_NAME}")

        # Load the specific vocab and rules for ML model 
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Track when ML model starts loading 
        logger.info(f"Loading model from HuggingFace: {MODEL_NAME}")

        # AutoModelForSequenceClassification loads pretrained model for classification task
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
        )

        logger.info("Model loaded successfully")
        logger.info(f"Model parameters: {self.model.num_parameters():,}")

    def _tokenize_dataset(
        self,
        # Input data 
        df: pd.DataFrame,
        # Default column name
        text_col: str = "body",
        # Label column 
        label_col: str = "category",
        # Max tokens (word count)
        max_length: int = 128,
        # returns a dataset 
    ) -> Dataset:
        """
        Convert a pandas DataFrame into a HuggingFace Dataset with tokenized text.

        Args:
            df:         DataFrame with ticket text and labels
            text_col:   Column containing raw ticket text (NOT preprocessed_text)
            label_col:  Column with category strings
            max_length: Maximum token length 
        Returns:
            HuggingFace Dataset ready for the Trainer
        """

        # Convert category strings to integers
    
        # Separate copy than original
        df = df.copy()
        # Converting labels to integers 
        df["label"] = df[label_col].map(CATEGORY_TO_ID)

        # Combine subject + body 
        # If subject column exists, it pairs with the body column. Otherwise defaults back to text_col
        # Check if subject column exists 
        if "subject" in df.columns:
            # Creates new column and run function on every row -> combine subject with the body using [sep] token
            df["input_text"] = df.apply(
                lambda row: f"{row.get('subject', '')} [SEP] {row.get('body', '')}".strip(),
                axis=1,
            )
        else:
            df["input_text"] = df[text_col].fillna("")

        # Select only the columns the model needs
        dataset_dict = {
            "text":  df["input_text"].tolist(),
            "label": df["label"].tolist(),
        }

        # Convert to HuggingFace Dataset
        hf_dataset = Dataset.from_dict(dataset_dict)

        # Tokenization function — applied to every example in the dataset
        def tokenize_function(examples):
            """
            Convert raw text strings into token IDs that DistilBERT can read.

            The tokenizer returns three things:
              input_ids:      The token IDs (integers representing words/subwords)
              attention_mask: 1 for real tokens, 0 for padding tokens
                             Tells the model which positions to pay attention to
                             and which to ignore (padding added to make all
                             sequences the same length in a batch)
            """
            return self.tokenizer(
                examples["text"],
                max_length=max_length,      # Truncate at 128 tokens
                padding="max_length",       # Pad shorter sequences to 128 with zeros
                truncation=True,            # Cut sequences longer than max_length
                return_tensors=None,        # Return lists, not tensors (Trainer handles this)
            )

        # Apply tokenize_function to every row in the dataset
        # batched=True processes multiple rows at once — much faster
        hf_dataset = hf_dataset.map(tokenize_function, batched=True)

        # Remove the raw text column — the model only needs token IDs and attention masks
        hf_dataset = hf_dataset.remove_columns(["text"])

        # Tell PyTorch what format we want tensors in
        hf_dataset.set_format("torch")

        return hf_dataset

    def _compute_metrics(self, eval_pred) -> Dict:
        """
        Calculate accuracy and F1 score during training evaluation.

        This function gets called by the Trainer after each epoch.
        It receives (predictions, true_labels) and returns a metrics dict.

        HuggingFace's evaluate library provides standardized metric implementations
        so you don't have to write accuracy/F1 calculations yourself.

        Args:
            eval_pred: Tuple of (logits, labels) from the model
        Returns:
            Dict with accuracy and f1 keys
        """
        # Load metric calculators from the evaluate library
        accuracy_metric = evaluate.load("accuracy")
        f1_metric       = evaluate.load("f1")

        # eval_pred is a tuple: (logits, labels)
        # logits: raw model outputs, shape (n_samples, 6) — NOT probabilities yet
        # labels: true category integers, shape (n_samples,)
        logits, labels = eval_pred

        # argmax converts logits to predictions
        # For each sample, pick the category with the highest logit score
        # axis=-1 means "take the max across the last dimension (categories)"
        predictions = np.argmax(logits, axis=-1)

        # Compute accuracy: fraction of predictions that match true labels
        accuracy = accuracy_metric.compute(
            predictions=predictions,
            references=labels,
        )

        # Compute macro F1: average F1 across all categories
        # average="macro" treats all categories equally regardless of size
        f1 = f1_metric.compute(
            predictions=predictions,
            references=labels,
            average="macro",
        )

        # Return combined metrics dict
        return {
            "accuracy": accuracy["accuracy"],
            "f1":       f1["f1"],
        }

    def train(
        self,
        train_df: pd.DataFrame,
        test_df:  pd.DataFrame,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        experiment_name: str = "triage_distilbert",
    ) -> Dict:
        """
        Fine-tune DistilBERT on our ticket data.

        WHAT HAPPENS DURING FINE-TUNING:
          1. Model makes predictions on a batch of tickets
          2. Loss is calculated: how wrong were the predictions?
          3. Backpropagation: calculate which weights caused the error
          4. Optimizer updates weights to reduce the error
          5. Repeat for all batches → one epoch complete
          6. Repeat for num_epochs epochs

        HYPERPARAMETERS EXPLAINED:
          num_epochs=3:        3 full passes through training data
                               More epochs = better fit but risk of overfitting
                               3 is the standard starting point for fine-tuning

          batch_size=16:       Process 16 tickets at once before updating weights
                               Larger = faster but needs more memory
                               16 is safe for most laptops

          learning_rate=2e-5:  How much to adjust weights each step
                               2e-5 (0.00002) is the standard for fine-tuning
                               Too high = unstable training, too low = too slow

        Args:
            train_df:        Training set DataFrame
            test_df:         Test set DataFrame (used for evaluation each epoch)
            num_epochs:      Number of complete passes through training data
            batch_size:      Tickets processed together in each training step
            learning_rate:   Step size for weight updates
            experiment_name: MLflow experiment name
        Returns:
            Dict of final evaluation metrics
        """
        # Load the model if not already loaded
        if self.model is None:
            self._load_model_and_tokenizer()

        logger.info(f"Fine-tuning DistilBERT on {len(train_df):,} training tickets")
        logger.info(f"Epochs: {num_epochs} | Batch size: {batch_size} | LR: {learning_rate}")

        # ---- TOKENIZE DATA ----
        # Convert DataFrames to HuggingFace Datasets with token IDs
        logger.info("Tokenizing training set...")
        train_dataset = self._tokenize_dataset(train_df)

        logger.info("Tokenizing test set...")
        test_dataset  = self._tokenize_dataset(test_df)

        # ---- TRAINING ARGUMENTS ----
        # TrainingArguments configures every aspect of the training loop
        # so we don't have to write the loop ourselves
        output_dir = str(MODELS_DIR / "distilbert_checkpoints")

        training_args = TrainingArguments(
            output_dir=output_dir,              # Where to save checkpoints

            # Training schedule
            num_train_epochs=num_epochs,         # 3 full passes through data
            per_device_train_batch_size=batch_size,  # 16 tickets per step
            per_device_eval_batch_size=batch_size,   # 16 tickets per eval step

            # Learning rate with warmup
            # Warmup: start with a very small LR and increase to learning_rate
            # over the first 10% of training steps. Prevents unstable early updates.
            learning_rate=learning_rate,
            warmup_ratio=0.1,

            # Evaluation strategy: run evaluation after every epoch
            # This lets us see accuracy improving each epoch
            evaluation_strategy="epoch",
            save_strategy="epoch",

            # Keep only the best checkpoint (by eval accuracy) to save disk space
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",

            # Logging
            logging_dir=str(MODELS_DIR / "logs"),
            logging_steps=10,           # Log every 10 training steps

            # Apple Silicon GPU support
            use_mps_device=(self.device.type == "mps"),

            # Reproducibility
            seed=42,

            # Suppress excessive HuggingFace output
            report_to="none",          # Don't send to wandb/tensorboard
        )

        # ---- TRAINER ----
        # HuggingFace Trainer handles the entire training loop:
        #   forward pass → loss calculation → backprop → optimizer step
        # We provide the model, data, and config — it handles the rest
        trainer = Trainer(
            model=self.model,                    # Our DistilBERT + classification head
            args=training_args,                  # Training configuration above
            train_dataset=train_dataset,         # 720 tokenized tickets
            eval_dataset=test_dataset,           # 180 tokenized tickets
            compute_metrics=self._compute_metrics,  # Our accuracy/F1 function
            callbacks=[
                # EarlyStopping: if accuracy doesn't improve for 2 epochs, stop
                # Prevents wasted compute and overfitting
                EarlyStoppingCallback(early_stopping_patience=2)
            ],
        )

        # ---- TRAIN ----
        logger.info("Starting fine-tuning...")
        logger.info("You will see progress bars for each epoch below:")
        train_result = trainer.train()

        self.is_trained = True
        logger.info("Fine-tuning complete")

        # ---- EVALUATE ----
        logger.info("Running final evaluation on test set...")
        eval_metrics = trainer.evaluate()

        # Extract the metric values (HuggingFace prefixes them with "eval_")
        final_accuracy = eval_metrics.get("eval_accuracy", 0)
        final_f1       = eval_metrics.get("eval_f1", 0)

        logger.info(f"Test accuracy: {final_accuracy:.1%}")
        logger.info(f"Macro F1:      {final_f1:.1%}")

        # ---- SAVE MODEL ----
        # Save the fine-tuned model weights to disk
        # Saving both model and tokenizer so we can reload later without internet
        model_save_path = MODELS_DIR / "distilbert_finetuned"
        self.model.save_pretrained(str(model_save_path))
        self.tokenizer.save_pretrained(str(model_save_path))
        logger.info(f"Fine-tuned model saved to {model_save_path}")

        # ---- LOG TO MLFLOW ----
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"distilbert_v{self.model_version}"):
            mlflow.log_params({
                "model_name":     MODEL_NAME,
                "num_epochs":     num_epochs,
                "batch_size":     batch_size,
                "learning_rate":  learning_rate,
                "max_length":     128,
                "n_train":        len(train_df),
                "model_version":  self.model_version,
            })
            mlflow.log_metrics({
                "test_accuracy":  final_accuracy,
                "f1_macro":       final_f1,
            })

        # Return metrics dict (same structure as BaselineClassifier for easy comparison)
        return {
            "test_accuracy":  final_accuracy,
            "f1_macro":       final_f1,
            "n_train":        len(train_df),
            "n_test":         len(test_df),
        }

    def predict_single(self, text: str) -> Dict:
        """
        Classify a single ticket and return prediction + confidence.

        This is what runs in real-time when a ticket comes in.
        Much faster than training — just a single forward pass.

        Args:
            text: Raw ticket text (NOT preprocessed — DistilBERT handles that)
        Returns:
            Dict with predicted_category, confidence, all_probabilities, auto_routed
        """
        if self.model is None or not self.is_trained:
            # Try to load a saved model first
            saved_path = MODELS_DIR / "distilbert_finetuned"
            if saved_path.exists():
                self.load(str(saved_path))
            else:
                raise RuntimeError(
                    "Model not trained. Run train() first or ensure a saved model exists."
                )

        # Switch model to evaluation mode
        # This disables dropout (random neuron deactivation used during training)
        # Dropout is good for training (prevents overfitting) but bad for inference
        # (you want consistent, deterministic predictions)
        self.model.eval()

        # Tokenize the single input text
        # return_tensors="pt" means return PyTorch tensors
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",  # pt = PyTorch
        )

        # Move tensors to the same device as the model (GPU or CPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # torch.no_grad() tells PyTorch not to track gradients
        # Gradients are needed for training (backprop) but not for inference
        # Skipping gradient tracking makes inference faster and uses less memory
        with torch.no_grad():
            outputs = self.model(**inputs)
            # outputs.logits shape: (1, 6) — one row, six category scores
            logits = outputs.logits

        # Convert logits to probabilities using softmax
        # Softmax converts raw scores to values between 0 and 1 that sum to 1.0
        # e.g. [2.1, 0.3, -0.5, 0.1, -0.2, 0.0] → [0.82, 0.13, 0.03, ...]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Move from GPU to CPU and convert to numpy for easier handling
        proba_np = probabilities[0].cpu().numpy()

        # Build probability dict: category name → probability
        proba_dict = {
            ID_TO_CATEGORY[i]: float(proba_np[i])
            for i in range(NUM_LABELS)
        }

        # Predicted category = the one with highest probability
        predicted_id       = int(proba_np.argmax())
        predicted_category = ID_TO_CATEGORY[predicted_id]
        confidence         = float(proba_np.max())

        return {
            "predicted_category":  predicted_category,
            "confidence":          confidence,
            "all_probabilities":   proba_dict,
            "auto_routed":         confidence >= 0.75,
        }

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Classify multiple tickets efficiently.
        Processes in batches rather than one at a time for speed.

        Args:
            texts:      List of raw ticket texts
            batch_size: How many to process at once
        Returns:
            List of prediction dicts (same format as predict_single)
        """
        results = []

        # Process in chunks of batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Tokenize the whole batch at once
            inputs = self.tokenizer(
                batch,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            proba_np = probabilities.cpu().numpy()

            # Build result for each ticket in the batch
            for j in range(len(batch)):
                row = proba_np[j]
                pred_id = int(row.argmax())
                results.append({
                    "predicted_category": ID_TO_CATEGORY[pred_id],
                    "confidence":         float(row.max()),
                    "all_probabilities":  {ID_TO_CATEGORY[k]: float(row[k]) for k in range(NUM_LABELS)},
                    "auto_routed":        float(row.max()) >= 0.75,
                })

        return results

    def load(self, model_path: str) -> None:
        """
        Load a previously fine-tuned model from disk.
        Used to reload the model without retraining.

        Args:
            model_path: Path to the saved model directory
        """
        logger.info(f"Loading fine-tuned model from {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.is_trained = True

        logger.info("Model loaded successfully")