import os
import json
import logging
import requests
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# =============================
# Setup
# =============================
load_dotenv()

# Logging setup
logging.basicConfig(
    filename="dashboard.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Environment variables
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAALW%2F3gEAAAAACbuBHpkCKh5FNKW1xXLPdBZAmk4%3DogmOnHyhqONUWNhrEitUZgXpFYUliPZgmEUcmi8jv99FlV0A1u"
HF_API_KEY = os.getenv("HF_API_KEY")

# =============================
# Twitter API Function
# =============================
def fetch_tweets(query, max_results=5):
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    params = {
        "query": query,
        "max_results": max_results,
        "tweet.fields": "created_at,text,author_id"
    }
    try:
        response = requests.get(url, headers=headers, params=params)

        logging.info(f"Twitter API request URL: {response.url}")
        logging.info(f"Twitter API status: {response.status_code}")
        logging.info(f"Twitter API response text: {response.text}")

        if response.status_code != 200:
            return None, f"Error fetching tweets: {response.status_code} {response.text}"

        data = response.json()
        logging.info(f"Twitter API JSON response: {json.dumps(data, indent=2)}")

        return data.get("data", []), None
    except Exception as e:
        logging.error(f"Exception fetching tweets: {str(e)}")
        return None, str(e)

# =============================
# Hugging Face API Function
# =============================
def analyze_texts(texts):
    url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    results = []

    for text in texts:
        try:
            payload = {
                "inputs": text,
                "parameters": {"candidate_labels": ["finance", "crypto", "stocks", "commodities", "general"]}
            }
            response = requests.post(url, headers=headers, json=payload)

            logging.info(f"Hugging Face API request payload: {json.dumps(payload, indent=2)}")
            logging.info(f"Hugging Face API status: {response.status_code}")
            logging.info(f"Hugging Face API response text: {response.text}")

            if response.status_code != 200:
                results.append({"error": f"Error: {response.status_code} {response.text}"})
                continue

            data = response.json()
            logging.info(f"Hugging Face API JSON response: {json.dumps(data, indent=2)}")
            results.append(data)
        except Exception as e:
            logging.error(f"Exception analyzing text: {str(e)}")
            results.append({"error": str(e)})
    return results

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="SocialPulse", layout="wide")
st.title("📊 SocialPulse - MVP Dashboard")

# Sidebar controls
st.sidebar.header("Controls")
query = st.sidebar.text_input("Search Query", "finance")
max_results = st.sidebar.slider("Number of Tweets", min_value=1, max_value=10, value=3)

if st.sidebar.button("Fetch Tweets"):
    st.write("🔍 Fetching tweets...")
    tweets, error = fetch_tweets(query, max_results)

    if error:
        st.error(error)
    elif tweets:
        st.success(f"Fetched {len(tweets)} tweets.")
        st.json(tweets)  # Display raw JSON in Streamlit for transparency

        texts = [t["text"] for t in tweets]
        analyses = analyze_texts(texts)

        st.subheader("Tweet Analysis")
        for i, (tweet, analysis) in enumerate(zip(tweets, analyses), start=1):
            with st.expander(f"Tweet {i}: {tweet['text'][:50]}..."):
                st.write(f"**Tweet:** {tweet['text']}")
                st.json(analysis)
    else:
        st.warning("No tweets found.")

