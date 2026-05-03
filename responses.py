import random

responses = {
    "happy": [
        "That sounds like a positive moment. What made it feel that way?",
        "Nice! What contributed to this feeling today?",
    ],
    "sad": [
        "That seems like a tough moment. Want to talk about what caused it?",
        "I hear some heaviness there. What happened?",
    ],
    "angry": [
        "That sounds frustrating. What triggered this feeling?",
        "Something seems to have upset you. Want to unpack it?",
    ],
    "anxious": [
        "It sounds like something is worrying you. What’s on your mind?",
        "That feels tense. Is there something specific causing this?",
    ],
    "neutral": [
        "Sounds like a balanced day. Anything notable you want to reflect on?",
        "Nothing too strong here—anything that stood out today?",
    ]
}


def get_response(emotion):
    return random.choice(responses.get(emotion, ["Tell me more."]))