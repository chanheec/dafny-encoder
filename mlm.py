import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from typing import List, Tuple
import os, json

# ==========================================
# Configuration & Setup
# ==========================================
DATA_PATH = "/content/drive/MyDrive/dafny-encoder/dafny-data.json"

# MODEL_NAME = "microsoft/codebert-base"
MODEL_NAME = "microsoft/codebert-base-mlm"
MAX_LENGTH = 512  # CodeBERT max length
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
OUTPUT_DIR = "/content/drive/MyDrive/dafny-encoder/dafny_codebert_mlm"

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




# ==========================================
# Dataset Preparation
# ==========================================
# TODO: name, pre, body, post ----  different pair structures might work better
def load_dafny_data(json_file_path):
    print(json_file_path)
    examples = []
    if not os.path.exists(json_file_path):
        print(f"Data file {json_file_path} not found. Skipping.")
        return []

    with open(json_file_path, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
        for pair in pairs:
            try:
                method_name = pair["method_name"]
                body = pair["body"]
                body = body.strip()

                # note that we are simply using keyword `Method` instead of `function`, `lemma`, etc.
                name_body = f"Method: {method_name}\nBody: {body}"
                spec = ""

                for pre in pair["preconditions"]:
                    pre = pre.strip()
                    spec = f"{spec}\nrequires {pre}"

                for postcond in pair["postconditions"]:
                    postcond = postcond.strip()
                    spec = f"{spec}\nensures {pre}"
                
                examples.append((name_body, spec))


            except json.JSONDecodeError:
                print("json error")
                continue

    print(examples[0])
    return examples


class DafnyDataset(Dataset):
    def __init__(self, data_pairs: List[Tuple[str, str]], tokenizer: RobertaTokenizer, max_length: int):
        """
        Args:
            data_pairs: List of (specification, proof) tuples.
            tokenizer: The HuggingFace tokenizer.
            max_length: Maximum sequence length.
        """
        self.data_pairs = data_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        spec, proof = self.data_pairs[idx]

        # Format: <s> Specification </s> </s> Proof </s>
        # RoBERTa tokenizer handles the <s> and </s> automatically if we pass two sequences.
        # We allow truncation to ensure it fits within the model's limit.
        encoding = self.tokenizer(
            spec,
            proof,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # The 'input_ids' and 'attention_mask' are returned as tensors of shape [1, seq_len]
        # We need to squeeze them to [seq_len] for the Dataset
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            # Labels are not needed here; DataCollatorForLanguageModeling creates them automatically
        }



# ==========================================
# Main Training Pipeline
# ==========================================
def main():
    # Load Tokenizer and Model 
    print("Loading CodeBERT model and tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaForMaskedLM.from_pretrained(MODEL_NAME)
    model.to(device)

    # Load Your Data
    print("Loading dataset...")
    dataset_pairs =  load_dafny_data(DATA_PATH)

    # Split into train and validation (90/10 split)
    train_size = int(0.9 * len(dataset_pairs))
    train_data = dataset_pairs[:train_size]
    val_data = dataset_pairs[train_size:]

    # Create PyTorch Datasets
    train_dataset = DafnyDataset(train_data, tokenizer, MAX_LENGTH)
    eval_dataset = DafnyDataset(val_data, tokenizer, MAX_LENGTH)

    
    # Data Collator for Masked Language Modeling (MLM)
    # randomly select 15% of the tokens in the input_ids for masking/random replacement/keeping
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15
    )


    
    # Training Arguments 
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(), # Use mixed precision if GPU is available
        report_to="none" # Disable wandb/mlflow logging for this simple script
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    # Train
    print("Starting training...")
    trainer.train()


    # Save Final Model
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete.")


    # Evaluate
    eval_results = trainer.evaluate()
    print(eval_results)

if __name__ == "__main__":
    main()