import pandas as pd
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score


df = pd.read_csv("data/cleaned_data.csv")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)


emotion_map = {
    "joy": "happy", "love": "happy", "amusement": "happy",
    "excitement": "happy", "optimism": "happy", "pride": "happy",
    "caring": "happy", "gratitude": "happy", "relief": "happy", "desire": "happy",

    "sadness": "sad", "grief": "sad", "disappointment": "sad",
    "remorse": "sad", "embarrassment": "sad",

    "anger": "angry", "annoyance": "angry", "disgust": "angry",
    "disapproval": "angry",

    "fear": "anxious", "nervousness": "anxious", "confusion": "anxious",

    "neutral": "neutral", "curiosity": "neutral",
    "realization": "neutral", "approval": "neutral",
    "admiration": "neutral", "surprise": "neutral"
}

df["label"] = df["label"].map(emotion_map)
df = df.dropna(subset=["label"])


label_list = ["happy", "sad", "angry", "anxious", "neutral"]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

df["label_id"] = df["label"].map(label_to_id)

dataset = Dataset.from_pandas(df[["text", "label_id"]])
dataset = dataset.rename_column("label_id", "labels")
dataset = dataset.train_test_split(test_size=0.2)


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize, batched=True)


model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_list)
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,   # reduce if RAM issues
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()


trainer.save_model("model")
tokenizer.save_pretrained("model")

print("\nModel saved to /model folder")