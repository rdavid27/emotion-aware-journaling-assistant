from predictor import predict_emotion

samples = [
    "I feel amazing today",
    "I am really stressed about exams",
    "This is so frustrating",
    "I feel sad and empty",
    "It was just an average day",

    "yeah great my life is amazing",
    "idk man just feeling weird",
    "this is fine..."
]

for s in samples:
    pred, method = predict_emotion(s)
    print(f"\nInput: {s}")
    print("Prediction:", pred)
    print("Used:", method)