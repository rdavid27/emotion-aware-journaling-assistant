import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
from predictor import predict_emotion
from responses import get_response
from storage import save_entry
from report import generate_pdf

st.set_page_config(page_title="Emotion Assistant", page_icon="💬", layout="wide")

page = st.sidebar.radio("Navigate", ["Chat", "Dashboard"])

emoji_map = {
    "happy": "😊",
    "sad": "😔",
    "angry": "😠",
    "anxious": "😟",
    "neutral": "😐"
}

def type_text(text):
    placeholder = st.empty()
    displayed = ""
    for char in text:
        displayed += char
        time.sleep(0.01)
        placeholder.markdown(displayed)

if page == "Chat":

    st.title("💬 Emotion-Aware Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "stage" not in st.session_state:
        st.session_state.stage = "initial"

    if "last_emotion" not in st.session_state:
        st.session_state.last_emotion = None

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Type your message...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        if st.session_state.stage == "initial":
            emotion, method = predict_emotion(user_input)
            st.session_state.last_emotion = emotion
            save_entry(user_input, emotion)
            reply = get_response(emotion)
            st.session_state.stage = "followup"

        elif st.session_state.stage == "followup":
            emotion = st.session_state.last_emotion

            if emotion == "sad":
                reply = "Has this been happening often, or was today different?"
            elif emotion == "happy":
                reply = "Do you think this is something you can repeat tomorrow?"
            elif emotion == "angry":
                reply = "Did you get a chance to respond, or are you holding it in?"
            elif emotion == "anxious":
                reply = "Is this something coming up soon, or ongoing?"
            else:
                reply = "Was there anything that stood out today?"

            st.session_state.stage = "reflection"

        else:
            emotion = st.session_state.last_emotion

            if emotion == "sad":
                reply = "Thanks for sharing that. Taking things step by step can help."
            elif emotion == "happy":
                reply = "That’s great. Moments like this are worth remembering."
            elif emotion == "angry":
                reply = "It’s good you noticed that. Stepping back sometimes helps."
            elif emotion == "anxious":
                reply = "You're aware of it—that’s a good first step."
            else:
                reply = "Thanks for reflecting on that."

            st.session_state.stage = "initial"

        reply = f"{emoji_map.get(st.session_state.last_emotion, '')} {reply}"

        st.session_state.messages.append({"role": "assistant", "content": reply})

        with st.chat_message("assistant"):
            type_text(reply)

    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.stage = "initial"
        st.session_state.last_emotion = None


elif page == "Dashboard":

    st.title("📊 Emotion Dashboard")

    try:
        df = pd.read_csv("journal_log.csv")
    except:
        st.warning("No data yet")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])

    st.subheader("Recent Entries")
    st.dataframe(df.tail(10))

    st.subheader("Emotion Distribution")

    emotion_counts = df["emotion"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%')
    st.pyplot(fig1)

    st.subheader("Emotion Over Time")

    df_sorted = df.sort_values("date")

    fig2, ax2 = plt.subplots()
    ax2.plot(df_sorted["date"], df_sorted["emotion"].astype("category").cat.codes)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Emotion")
    st.pyplot(fig2)

    st.subheader("Insights")

    most_common = emotion_counts.idxmax()

    st.write(f"Most frequent emotion: **{most_common}**")
    st.write(f"Total entries: **{len(df)}**")

    if st.button("Generate PDF"):
        generate_pdf()

        with open("emotion_report.pdf", "rb") as f:
            st.download_button(
                "Download Report",
                f,
                file_name="emotion_report.pdf",
                mime="application/pdf"
            )