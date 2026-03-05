import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (only once)
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("spam_classifier.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

ps = PorterStemmer()

# Text preprocessing function (SAME as training)
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    words = [ps.stem(word) for word in words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧")

st.title("📧 Email Spam Classifier")
st.write("Enter an email message below to check whether it is **Spam** or **Not Spam**.")

# Input text
email_text = st.text_area("✉️ Email Content", height=200)

if st.button("🔍 Predict"):
    if email_text.strip() == "":
        st.warning("Please enter an email message.")
    else:
        processed_text = preprocess_text(email_text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 1:
            st.error("🚫 This email is **SPAM**")
        else:
            st.success("✅ This email is **NOT SPAM (HAM)**")

