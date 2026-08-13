import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
from sklearn.metrics.pairwise import cosine_similarity

# ---------- FONT ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Fake News Detection",
    layout="wide"
)

# ---------- LOAD MODEL ----------
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# ---------- NEWS API FUNCTION ----------
def get_latest_news(query):
    api_key = " b53502f34fa046f3ae2e05f550967292"   # 🔴 PUT YOUR API KEY HERE

    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"

    response = requests.get(url)
    data = response.json()

    articles = []

    if data["status"] == "ok":
        for article in data["articles"][:3]:
            articles.append({
                "title": article["title"],
                "url": article["url"]
            })

    return articles

# ---------- SIDEBAR ----------
st.sidebar.title("Fake News AI Dashboard")

st.sidebar.info(
"""
This system uses **Machine Learning and NLP**
to classify news articles as Fake or Real.
"""
)

st.sidebar.markdown("---")
st.sidebar.write("Developed for AI Project")

# ---------- MAIN TITLE ----------
st.title("Fake News Detection System")

st.write(
"""
Paste a news article below and the AI model will analyze it and determine
whether the news is **Fake or Real**.
"""
)

# ---------- INPUT ----------
news = st.text_area("Enter News Article", height=200)

if st.button("Analyze News"):

    if news.strip() == "":
        st.warning("Please enter news text")
    else:

        vector = vectorizer.transform([news])

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        col1, col2 = st.columns(2)

        # ---------- RESULT ----------
        with col1:

            st.subheader("Prediction Result")

            if prediction == 0:
                st.error("⚠️ Fake News Detected")
                confidence = probability[0]

                # ---------- LIVE NEWS + SIMILARITY ----------
                st.subheader("Related Real News & Similarity Check")

                keywords = " ".join(news.split()[:5])
                articles = get_latest_news(keywords)

                for article in articles:

                    real_text = article["title"]

                    vec1 = vectorizer.transform([news])
                    vec2 = vectorizer.transform([real_text])

                    similarity = cosine_similarity(vec1, vec2)[0][0]

                    st.write(f"🔗 {article['title']}")
                    st.write(article["url"])

                    st.write("Similarity Score:", round(similarity * 100, 2), "%")

                    if similarity > 0.5:
                        st.success("High similarity with real news")
                    else:
                        st.warning("Low similarity — possible misinformation")

                    st.markdown("---")

            else:
                st.success("✅ Real News")
                confidence = probability[1]

            st.write("Confidence:", round(confidence * 100, 2), "%")
            st.progress(int(confidence * 100))

        # ---------- CHART ----------
        with col2:

            st.subheader("Prediction Probability")

            labels = ["Fake", "Real"]
            values = probability

            fig, ax = plt.subplots()
            ax.bar(labels, values)
            ax.set_ylabel("Probability")
            ax.set_title("Fake vs Real Prediction")

            st.pyplot(fig)

        # ---------- WORD CLOUD ----------
        st.subheader("News Text Word Cloud")

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(news)

        fig2, ax2 = plt.subplots()
        ax2.imshow(wordcloud)
        ax2.axis("off")

        st.pyplot(fig2)

        # ---------- TEXT ANALYSIS ----------
        st.subheader("Text Analysis")

        words = news.split()

        col3, col4, col5 = st.columns(3)

        col3.metric("Total Words", len(words))
        col4.metric("Characters", len(news))
        col5.metric("Average Word Length", round(len(news)/len(words),2))

# ---------- FOOTER ----------
st.markdown("---")
st.warning("Note: This prediction is based on trained data and may not always reflect real-world truth.")
st.markdown("Fake News Detection System using Machine Learning")