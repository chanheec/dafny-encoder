import torch
import random
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from typing import List, Tuple
import os, json

# ==========================================
# Configuration
# ==========================================

DATA_PATH = "/content/drive/MyDrive/dafny-encoder/dafny-data.json"
PRETRAINED_MODEL_PATH = "/content/drive/MyDrive/dafny-encoder/dafny_codebert_mlm"
# OUTPUT_DIR = "/content/drive/MyDrive/dafny-encoder/dafny_lemma_recommender"
OUTPUT_DIR = "/content/drive/MyDrive/dafny-encoder/dafny_lemma_recommender_bce"

USE_BCE_LOSS = True  # If False, use Mean Squared Error Loss

MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
NEGATIVE_RATIO = 3  # For every 1 correct lemma, train on 3 wrong ones

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# ==========================================
# Dataset with Negative Sampling
# ==========================================
class LemmaRecommendationDataset(Dataset):
    def __init__(self, 
                 context_lemma_pairs: List[Tuple[str, str]], 
                 all_lemmas: List[str], 
                 tokenizer: RobertaTokenizer, 
                 max_length: int,
                 negative_ratio: int = 1):
        """
        Notable Args:
            context_lemma_pairs: List of (proof_context, correct_lemma_signature)
            all_lemmas: A list of ALL available lemma signatures (pool for negative sampling)
            negative_ratio: How many negative examples to generate for each positive one
        """
        self.context_lemma_pairs = context_lemma_pairs
        self.all_lemmas = all_lemmas
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.negative_ratio = negative_ratio
        
        # Pre-generate samples with negative sampling
        self.samples = self._generate_samples()

    def _generate_samples(self):
        samples = []
        for context, correct_lemma in self.context_lemma_pairs:
            # 1. Add the Positive Sample (Label = 1)
            samples.append({
                "context": context,
                "lemma": correct_lemma,
                "label": 1.0
            })
            
            # 2. Add Negative Samples (Label = 0)
            # Pick random lemmas that are NOT the correct one
            for _ in range(self.negative_ratio):
                random_lemma = random.choice(self.all_lemmas)
                # Ensure we don't accidentally pick the correct one
                # TODO: however, in fact, there might be multiple correct lemmas 
                # i.e., different lemmas can be actually significantly relevant 
                # This might be improved by making sure the picked lemma is from different file (from the file the proof goal's is from)
                # This is somthing to think about.
                while random_lemma == correct_lemma:
                    random_lemma = random.choice(self.all_lemmas)
                
                samples.append({
                    "context": context,
                    "lemma": random_lemma,
                    "label": 0.0
                })
        
        # Shuffle so positives and negatives are mixed
        random.shuffle(samples)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Format: <s> Proof Context </s> </s> Lemma Signature </s>
        # This allows the model to "attend" from the proof to the lemma
        encoding = self.tokenizer(
            item["context"],
            item["lemma"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        if not USE_BCE_LOSS:
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(item["label"], dtype=torch.float)
            }
        else:
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(item["label"], dtype=torch.float).unsqueeze(0)
            }
    

def load_dafny_data(json_file_path):
    """
    returns 
    (1) list of (goal-req/ens, function-body) pairs
    (2) the long list of candidate lemmas 
    """
    print(json_file_path)
    examples = []
    if not os.path.exists(json_file_path):
        print(f"Data file {json_file_path} not found. Skipping.")
        return []

    # TODO: name, pre, body, post ----  different pair structures might work better
    with open(json_file_path, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
        lemma_pool = []
        # Example: A proof needing a math lemma
        # Example: A list proof
        for pair in pairs:
            try:
                method_name = pair["method_name"]
                body = pair["body"]
                body = body.strip()

                name_body = f"Method: {method_name}\nBody: {body}"
                spec = ""

                for pre in pair["preconditions"]:
                    pre = pre.strip()
                    spec = f"{spec}\nrequires {pre}"

                for postcond in pair["postconditions"]:
                    postcond = postcond.strip()
                    spec = f"{spec}\nensures {pre}"
                
                examples.append((spec, name_body))
                lemma_pool.append(name_body)


            except json.JSONDecodeError:
                print("json error")
                continue

    print(examples[0])
    return examples, lemma_pool



# ==========================================
# Main Training Pipeline
# ==========================================
def main():
    # Load Tokenizer & Model
    print("Loading previous pre-trained model...")
    tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    
    # Using `SequenceClassification`
    if USE_BCE_LOSS:
        model = RobertaForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL_PATH, 
            num_labels=1, # binary classification
            problem_type="multi_label_classification" # BCE Loss
    )
    else:
        model = RobertaForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL_PATH, 
            num_labels=1, # binary classification
            problem_type="regression" # Mean Squared Error Loss
    )
    
    model.to(device)


    # Prepare Datasets
    pairs, lemma_pool = load_dafny_data(DATA_PATH)
    
    # 90/10 Train/Val Split
    split_idx = int(0.9 * len(pairs))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    train_dataset = LemmaRecommendationDataset(
        train_pairs, lemma_pool, tokenizer, MAX_LENGTH, negative_ratio=NEGATIVE_RATIO
    )
    val_dataset = LemmaRecommendationDataset(
        val_pairs, lemma_pool, tokenizer, MAX_LENGTH, negative_ratio=NEGATIVE_RATIO
    )

    print(f"Training samples: {len(train_dataset)}")



    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # using DataCollatorWithPadding (NOT MLM collator -- DataCollatorForLanguageModeling)
        data_collator=DataCollatorWithPadding(tokenizer) 
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model
    print(f"Saving recommender model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # print("\nDone! To use this model:")
    # print("1. Encode (Context, Lemma_A), (Context, Lemma_B)...")
    # print("2. The model outputs a score for each pair.")
    # print("3. Pick the Lemma with the highest score.")

if __name__ == "__main__":
    main()