import pandas as pd

def load_data():
    df1= pd.read_csv("data/goemotions_1.csv", sep=",")
    df2= pd.read_csv("data/goemotions_2.csv", sep=",")
    df3= pd.read_csv("data/goemotions_3.csv", sep=",") 
    df= pd.concat([df1, df2, df3], ignore_index= True)
    df.columns= df.columns.str.strip().str.lower()
    return df

def clean_data(df):
    df= df.dropna()
    emotion_cols=["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
    df["emotion_count"]= df[emotion_cols].sum(axis=1)
    df= df[df["emotion_count"]==1]
    df["label"] = df[emotion_cols].idxmax(axis=1) 
    return df[["text", "label"]]

def main():
    df= load_data()
    print("Loaded: ", df.shape)
    df= clean_data(df)
    print("After cleaning:", df.shape)
    print(df.head())

if __name__ == "__main__":
    main()