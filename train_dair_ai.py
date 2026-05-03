import pandas as pd

df = pd.read_csv("data/final_emotion_5class.csv")

df = df.dropna()
df["text"] = df["text"].astype(str)

print("Dataset:", df.shape)
print(df["label"].value_counts())


from sklearn.model_selection import train_test_split

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

print("\nModel trained!")


from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_vec)

print("\n=== CLEAN MODEL ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


import joblib

joblib.dump(model, "model_clean.pkl")
joblib.dump(vectorizer, "vectorizer_clean.pkl")

print("\nSaved model_clean.pkl and vectorizer_clean.pkl")


def predict(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]


print("\n=== SAMPLE TEST ===")

samples = [
    "I feel amazing today",
    "I am really stressed",
    "This is so frustrating",
    "I feel sad and empty",
    "It was just a normal day"
]

for s in samples:
    print(f"{s} → {predict(s)}")