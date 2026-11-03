import re
import uuid

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/chat"

# UI strings for each supported language (English is the default)
STRINGS = {
    "en": {
        "title": "Student Dropout Assistant",
        "caption": "Ask about dropout factors, request a prediction for a student, or explore dataset statistics.",
        "placeholder": "Type your question...",
        "spinner": "Thinking...",
        "error": "Could not reach the server: ",
        "new_chat": "New conversation",
    },
    "el": {
        "title": "Βοηθός Πρόβλεψης Εγκατάλειψης Σπουδών",
        "caption": "Ρώτησε για τους παράγοντες εγκατάλειψης, ζήτησε πρόβλεψη για έναν φοιτητή ή δες στατιστικά του dataset.",
        "placeholder": "Γράψε την ερώτησή σου...",
        "spinner": "Σκέφτομαι...",
        "error": "Αδυναμία επικοινωνίας με τον server: ",
        "new_chat": "Νέα συνομιλία",
    },
}

GREEK_CHARS = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")


def detect_language(text: str) -> str:
    """Return 'el' if the text contains Greek characters, otherwise 'en'."""
    return "el" if GREEK_CHARS.search(text) else "en"


st.set_page_config(page_title="Student Dropout Assistant", layout="centered")

# Styling on top of the theme in .streamlit/config.toml
st.markdown(
    """
    <style>
    h1 { color: #0B4F9E; }
    [data-testid="stChatMessage"] {
        background-color: #E6F1FC;
        border: 1px solid #B9D6F2;
        border-radius: 12px;
    }
    [data-testid="stSidebar"] { background-color: #DCEBFA; }
    .stButton > button {
        background-color: #1E6FD9;
        color: white;
        border: none;
        border-radius: 8px;
    }
    .stButton > button:hover { background-color: #0B4F9E; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Per-browser-session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lang" not in st.session_state:
    st.session_state.lang = "en"

t = STRINGS[st.session_state.lang]

st.title(t["title"])
st.caption(t["caption"])

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle a new message
if prompt := st.chat_input(t["placeholder"]):
    st.session_state.lang = detect_language(prompt)
    t = STRINGS[st.session_state.lang]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(t["spinner"]):
            try:
                r = requests.post(
                    API_URL,
                    json={"message": prompt, "session_id": st.session_state.session_id},
                    timeout=120,
                )
                r.raise_for_status()
                answer = r.json()["response"]
            except Exception as e:
                answer = t["error"] + str(e)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    if st.button(t["new_chat"]):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()