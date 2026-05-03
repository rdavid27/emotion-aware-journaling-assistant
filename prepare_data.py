import pandas as pd

df = pd.read_csv("data/emotion_dataset.csv")

print("Original shape:", df.shape)
print(df.head())



df = df[df["label"] != "surprise"]

df["label"] = df["label"].replace("love", "happy")


print("\nClass distribution:")
print(df["label"].value_counts())



df.to_csv("data/final_emotion_5class.csv", index=False)

print("\nSaved as final_emotion_5class.csv")