import streamlit as st
import pickle

# -------------------------------
# Load the model and vectorizer
# -------------------------------
MODEL_PATH = "Bestmodel.pkl"
VECTORIZER_PATH = "tfidf_vector.pkl"

# Load safely with error handling
try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_PATH, "rb") as vector_file:
        vectorizer = pickle.load(vector_file)
    model_loaded = True
except FileNotFoundError:
    st.error("❌ Model or vectorizer file not found! Please ensure both .pkl files are in the same directory.")
    model_loaded = False

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="Text Classification App", page_icon="🧠", layout="wide")

st.title("🧠 Text Classification with TF-IDF + ML Model")
st.write("This app predicts categories for given texts using your trained model and TF-IDF vectorizer.")

st.markdown("---")

# Example inputs
default_texts = [
    "Stock market rises.",
    "Indian women Teams win World Cup",
    "Car prices down across the board The cost of buying both new and second hand cars fell sharply over the past five years, a new survey has found",
    "Apple launches new iPhone with advanced features"
]

user_input = st.text_area(
    "✍️ Enter one text per line for prediction:",
    value="\n".join(default_texts),
    height=200
)

if st.button("🔍 Predict") and model_loaded:
    texts = [t.strip() for t in user_input.split("\n") if t.strip()]
    if len(texts) == 0:
        st.warning("Please enter at least one text.")
    else:
        # Transform texts
        text_tfidf = vectorizer.transform(texts)

        # Predict
        predictions = model.predict(text_tfidf)

        # Map your label numbers to readable names (customize below)
        label_map = {
            0: "World News 🌍",
            1: "Sports 🏏",
            2: "Business 💼",
            3: "Technology 💻"
        }

        st.markdown("---")
        st.subheader("🧾 Predictions:")
        for text, pred in zip(texts, predictions):
            label = label_map.get(pred, str(pred))
            st.success(f"**Text:** {text}\n\n→ Predicted Label: **{label}**")
            st.markdown("---")

st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    This app uses:
    - Trained ML model (e.g., XGBoost, RandomForest, Naive Bayes)
    - TF-IDF Vectorizer
    - Streamlit for UI
    """
)