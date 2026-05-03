import pandas as pd
import re

df = pd.read_csv("data/cleaned_data.csv")

print("Dataset shape:", df.shape)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

df = df[df["text"].str.strip() != ""]


emotion_map = {
    # HAPPY
    "joy": "happy", "love": "happy", "amusement": "happy",
    "excitement": "happy", "optimism": "happy", "pride": "happy",
    "caring": "happy", "gratitude": "happy", "relief": "happy", "desire": "happy",

    # SAD
    "sadness": "sad", "grief": "sad", "disappointment": "sad",
    "remorse": "sad", "embarrassment": "sad",

    # ANGRY
    "anger": "angry", "annoyance": "angry", "disgust": "angry",
    "disapproval": "angry",

    # ANXIOUS
    "fear": "anxious", "nervousness": "anxious", "confusion": "anxious",

    # NEUTRAL
    "neutral": "neutral", "curiosity": "neutral",
    "realization": "neutral", "approval": "neutral",
    "admiration": "neutral", "surprise": "neutral"
}

df["label_5"] = df["label"].map(emotion_map)

df = df.dropna(subset=["label_5"])

print("\nDistribution BEFORE balancing:")
print(df["label_5"].value_counts())


neutral_df = df[df["label_5"] == "neutral"].sample(20000, random_state=42)
other_df = df[df["label_5"] != "neutral"]

df = pd.concat([neutral_df, other_df])

print("\nDistribution AFTER balancing:")
print(df["label_5"].value_counts())


from sklearn.model_selection import train_test_split

X = df["text"]
y = df["label_5"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=30000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=300, class_weight="balanced")
model.fit(X_train_vec, y_train)

print("\nModel trained!")


from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_vec)

print("\n=== GOEMOTIONS MODEL ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


import joblib

joblib.dump(model, "model_noisy.pkl")
joblib.dump(vectorizer, "vectorizer_noisy.pkl")

print("\nSaved model_noisy.pkl and vectorizer_noisy.pkl")


def predict(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]


print("\n=== SAMPLE TEST ===")
samples = [
    "I feel amazing today",
    "I am really stressed",
    "This is so frustrating",
    "I feel sad and empty",
    "It was just a normal day",
    "yeah great my life is amazing"
]

for s in samples:
    print(f"{s} → {predict(s)}")