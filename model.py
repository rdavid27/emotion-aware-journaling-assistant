import pandas as pd
import re

df = pd.read_csv("data/cleaned_data.csv")

print("Dataset shape:", df.shape)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

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

print("\n5-class distribution BEFORE balancing:")
print(df["label_5"].value_counts())

neutral_df = df[df["label_5"] == "neutral"].sample(20000, random_state=42)
other_df = df[df["label_5"] != "neutral"]

df = pd.concat([neutral_df, other_df])

print("\n5-class distribution AFTER balancing:")
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
    max_df=0.85
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

from sklearn.svm import LinearSVC

model = LinearSVC(class_weight="balanced")
model.fit(X_train_vec, y_train)

print("\nModel training complete!")

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_vec)

print("\n=== 5-CLASS EVALUATION ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.title("Confusion Matrix (5-Class)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


def predict_emotion(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    return {
        "input": text,
        "predicted_emotion": pred
    }


print("\n=== SAMPLE PREDICTIONS ===")

print(predict_emotion("I am so happy today"))
print(predict_emotion("I feel really sad and tired"))
print(predict_emotion("This is so frustrating"))
print(predict_emotion("I am nervous about tomorrow"))
print(predict_emotion("It was just an average day"))