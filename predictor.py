import joblib


model_clean = joblib.load("model_clean.pkl")
vectorizer_clean = joblib.load("vectorizer_clean.pkl")

model_noisy = joblib.load("model_noisy.pkl")
vectorizer_noisy = joblib.load("vectorizer_noisy.pkl")


def is_messy(text):
    text = text.lower()

    if len(text.split()) < 6:
        return True

    if "..." in text or "??" in text or "!!" in text:
        return True

    return False


def predict_emotion(text):
    text = str(text)

    vec_clean = vectorizer_clean.transform([text])
    vec_noisy = vectorizer_noisy.transform([text])

    pred_clean = model_clean.predict(vec_clean)[0]
    pred_noisy = model_noisy.predict(vec_noisy)[0]

    
    if pred_clean == pred_noisy:
        return pred_clean, "agreement"

    if is_messy(text):
        return pred_noisy, "noisy_model"
    else:
        return pred_clean, "clean_model"