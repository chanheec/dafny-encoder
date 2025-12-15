import json
import logging
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    models,
    evaluation
)
import os

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

# --- Configuration ---
DATA_PATH = "/content/drive/MyDrive/dafny-encoder/dafny-data.json"

MODEL_NAME = "huggingface/CodeBERTa-small-v1"
OUTPUT_PATH = "/content/drive/MyDrive/dafny-encoder/dafny-codeberta-contrastive"
BATCH_SIZE = 16 # Small batch size for small GPUs; MNR loss benefits from larger batches if VRAM allows
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512

# Setup Logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)


# Json structure for Dafny data
# #[derive(Debug, Serialize)]
# struct DafnyPair {
#     method_name: String,
#     preconditions: Vec<String>,
#     postconditions: Vec<String>,
#     body: String,
# }
def load_dafny_data(json_file_path):
    print(json_file_path)
    examples = []
    if not os.path.exists(json_file_path):
        logging.warning(f"Data file {json_file_path} not found. Skipping.")
        print(f"Data file {json_file_path} not found. Skipping.")
        return []
    else:
        logging.info(f"Loading data from {json_file_path}")

    with open(json_file_path, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
        for pair in pairs:
            try:
                method_name = pair["method_name"]
                body = pair["body"]
                body = body.strip()

                name_pre_body = f"Method: {method_name}\nBody: {body}"

                for pre in pair["preconditions"]:
                    pre = pre.strip()
                    name_pre_body = f"{name_pre_body}\nRequires: {pre}"

                for postcond in pair["postconditions"]:
                    postcond = postcond.strip()
                    examples.append(InputExample(texts=[name_pre_body, postcond]))


            except json.JSONDecodeError:
                print("json error")
                continue

    logging.info(f"Loaded {len(examples)} pairs from {json_file_path}")
    print(examples[0])
    return examples

def main():
    # 1. Define the Model Architecture
    # We wrap CodeBERTa in a transformer model that outputs pooled embeddings
    logging.info(f"Loading base model: {MODEL_NAME}")

    word_embedding_model = models.Transformer(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH
    )

    # Add special tokens for Dafny if they weren't added during pre-training
    # (Note: For contrastive learning, it's better if the model was already MLM pre-trained
    # with these tokens, but adding them here prevents crashes).
    dafny_tokens = ["ensures", "requires", "invariant", "ghost", "==>", "::", "forall"]
    word_embedding_model.tokenizer.add_tokens(dafny_tokens)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    # Pooling: Convert token embeddings to a single vector (Mean Pooling is standard)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # 2. Load Data
    train_examples = load_dafny_data(DATA_PATH)

    if not train_examples:
        logging.error("No training data found. Please generate 'dafny_data.json' first.")
        return

    # 3. Create DataLoader
    # Shuffle is critical for MNR loss to ensure diverse negatives in every batch
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    # 4. Define Loss Function
    # MultipleNegativesRankingLoss:
    # For a pair (a, b), 'b' is the positive for 'a', and ALL other 'b's in the batch are negatives.
    # TODO: a = (prog1, post1), b = (prog1, post2), then post2 is not really a negative for prog1.
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 5. (Optional) Define Evaluator
    # In a real scenario, you reserve some data for this to track progress.
    # evaluator = evaluation.EmbeddingSimilarityEvaluator(...)

    # 6. Train
    logging.info("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=int(len(train_dataloader) * 0.1), # 10% warmup
        optimizer_params={'lr': LEARNING_RATE},
        output_path=OUTPUT_PATH,
        show_progress_bar=True,
        save_best_model=True,
        # wandb_config={"log": False}
    )

    logging.info(f"Training finished. Model saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()