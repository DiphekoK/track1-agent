"""
Fine-tunes DistilBERT on the labeled routing data from
data/label_dataset.py. Run this once (or whenever the labeled dataset
changes) before building the docker image - the trained weights get
baked into router/model/ and copied into the container so routing at
inference time is a local forward pass, zero Fireworks tokens spent
deciding.

CPU-only, tiny dataset, this takes well under a couple of minutes even
without a GPU. On AMD Developer Cloud, same script, just runs on
whatever device torch picks up - point it at a ROCm MI300X instance if
the dataset grows and this gets slow.

Run after label_dataset.py: python router/train_router.py
"""
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

DATASET_PATH = Path(__file__).parent.parent / "data" / "labeled_dataset.jsonl"
OUTPUT_DIR = Path(__file__).parent / "model"
LABELS = ["local", "fireworks"]
EPOCHS = 4
BATCH_SIZE = 8
LR = 2e-5


class RoutingDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.encodings = tokenizer(
            [r["prompt"] for r in records],
            truncation=True, padding=True, max_length=256, return_tensors="pt",
        )
        self.labels = torch.tensor([LABELS.index(r["label"]) for r in records])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_records():
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    correct = total = tp = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = torch.argmax(model(**batch).logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for p, l in zip(preds.tolist(), labels.tolist()):
                if p == 1 and l == 1:
                    tp += 1
                elif p == 1 and l == 0:
                    fp += 1
                elif p == 0 and l == 1:
                    fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"accuracy": correct / total, "fireworks_precision": precision, "fireworks_recall": recall}


def main():
    if not DATASET_PATH.exists():
        print(f"{DATASET_PATH} doesn't exist - run data/label_dataset.py first", file=sys.stderr)
        sys.exit(1)

    records = load_records()
    if len(records) < 10:
        print(f"only {len(records)} labeled records, need more before training means anything", file=sys.stderr)
        sys.exit(1)

    device = pick_device()
    print(f"training on {device}")

    split = int(len(records) * 0.8)
    train_records, test_records = records[:split], records[split:]

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(LABELS),
    ).to(device)

    train_loader = DataLoader(RoutingDataset(train_records, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(RoutingDataset(test_records, tokenizer), batch_size=BATCH_SIZE)

    # the dataset is almost always skewed toward "local" (that's a good
    # sign, it means the local model is fine most of the time) - weight
    # the loss so the minority "fireworks" class doesn't just get ignored
    n_fireworks_train = sum(1 for r in train_records if r["label"] == "fireworks")
    n_local_train = len(train_records) - n_fireworks_train
    weight_fireworks = n_local_train / max(n_fireworks_train, 1)
    class_weights = torch.tensor([1.0, weight_fireworks]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch + 1}/{EPOCHS} - loss {total_loss / len(train_loader):.4f}")

    metrics = evaluate(model, test_loader, device)
    print(f"held-out eval: {metrics}")
    if metrics["fireworks_precision"] == 0 and metrics["fireworks_recall"] == 0:
        print("note: router never predicted 'fireworks' on the test split - expected with "
              "only a handful of hard examples, add more adversarial queries for that "
              "category if this matters for your submission", file=sys.stderr)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"saved router weights to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
