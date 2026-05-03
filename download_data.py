from datasets import load_dataset
import pandas as pd

dataset = load_dataset("dair-ai/emotion")

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])
val_df = pd.DataFrame(dataset["validation"])

df = pd.concat([train_df, test_df, val_df], ignore_index=True)

label_map = {
    0: "sad",
    1: "happy",  
    2: "love",
    3: "angry",
    4: "anxious", 
    5: "surprise"
}

df["label"] = df["label"].map(label_map)


df = df[["text", "label"]]

df.to_csv("data/emotion_dataset.csv", index=False)

print("Saved dataset!")
print(df.head())
print("\nClass distribution:")
print(df["label"].value_counts())