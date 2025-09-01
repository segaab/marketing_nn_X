# streamlit_dashboard_three_requests.py
import streamlit as st
import os
from dotenv import load_dotenv
import requests
from requests_oauthlib import OAuth1Session
from huggingface_hub import InferenceClient
import pandas as pd
import logging

# -------------------------------
# Logging Setup
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.FileHandler("engageflow.log"), logging.StreamHandler()],
)

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

# -------------------------------
# Twitter Credentials
# -------------------------------
bearer_token = "AAAAAAAAAAAAAAAAAAAAALW%2F3gEAAAAACbuBHpkCKh5FNKW1xXLPdBZAmk4%3DogmOnHyhqONUWNhrEitUZgXpFYUliPZgmEUcmi8jv99FlV0A1u"
consumer_key = "1760306826262794242-TpU2JlTTm095iz0E5uzGffjmBt60yk"
consumer_secret = "etApPK2m0rhbTaVwiFcf1XPkjV4oFbgY8TmHf3zwvAzol"
oauth_token = "1760306826262794242-rGqTedDjHT8Qr7D7UTKxPSL0OoKNVU"
oauth_token_secret = "dKqswOOJMEkj0RZ1mhlV4K4iVUAp42oiQRQHvUaDhAB4S"

# -------------------------------
# Helper Functions
# -------------------------------
def log_response(response):
    logging.info(f"Status: {response.status_code}")
    logging.info(f"Headers: {response.headers}")
    logging.info(f"Body: {response.text[:400]}")  # preview first 400 chars

def fetch_tweets_format1(query, max_results):
    """Format 1: Standard params dict"""
    url = "https://api.x.com/2/tweets/search/recent"
    params = {"query": query, "max_results": max_results, "tweet.fields": "author_id,created_at,text"}
    headers = {"Authorization": f"Bearer {bearer_token}"}
    response = requests.get(url, headers=headers, params=params)
    log_response(response)
    return response.json().get("data", [])

def fetch_tweets_format2(query, max_results):
    """Format 2: Inline URL params"""
    url = f"https://api.x.com/2/tweets/search/recent?query={query}&max_results={max_results}&tweet.fields=author_id,created_at,text"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    response = requests.get(url, headers=headers)
    log_response(response)
    return response.json().get("data", [])

def fetch_tweets_format3(query, max_results):
    """Format 3: Minimal request, only query + max_results"""
    url = "https://api.x.com/2/tweets/search/recent"
    params = {"query": query, "max_results": max_results}
    headers = {"Authorization": f"Bearer {bearer_token}"}
    response = requests.get(url, headers=headers, params=params)
    log_response(response)
    return response.json().get("data", [])

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("EngageFlow: Fetch Tweets Test (Three Formats)")

query_input = st.text_input("Search Query or @username", "@TwitterDev -is:retweet")
max_results = st.slider("Max Results", 10, 100, 10)  # Free tier requires min 10

# Session state for storing results
if "tweets" not in st.session_state:
    st.session_state["tweets"] = []

# --- Request Format 1 ---
if st.button("Fetch Tweets (Format 1 - Standard)"):
    st.session_state["tweets"] = fetch_tweets_format1(query_input, max_results)
    st.success(f"Fetched {len(st.session_state['tweets'])} tweets (Format 1)")

# --- Request Format 2 ---
if st.button("Fetch Tweets (Format 2 - Inline URL)"):
    st.session_state["tweets"] = fetch_tweets_format2(query_input, max_results)
    st.success(f"Fetched {len(st.session_state['tweets'])} tweets (Format 2)")

# --- Request Format 3 ---
if st.button("Fetch Tweets (Format 3 - Minimal)"):
    st.session_state["tweets"] = fetch_tweets_format3(query_input, max_results)
    st.success(f"Fetched {len(st.session_state['tweets'])} tweets (Format 3)")

# Display tweets if available
tweets = st.session_state.get("tweets", [])
if tweets:
    st.subheader("Fetched Tweets")
    df = pd.DataFrame([{"Tweet ID": t["id"], "Text": t["text"]} for t in tweets])
    st.dataframe(df)
