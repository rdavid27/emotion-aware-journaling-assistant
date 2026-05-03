import pandas as pd
from datetime import datetime

def save_entry(text, emotion):
    new_entry = pd.DataFrame([{
        "date": datetime.now(),
        "text": text,
        "emotion": emotion
    }])

    try:
        df = pd.read_csv("journal_log.csv")
        df = pd.concat([df, new_entry], ignore_index=True)
    except:
        df = new_entry

    df.to_csv("journal_log.csv", index=False)