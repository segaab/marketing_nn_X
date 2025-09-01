# streamlit_dashboard.py
import streamlit as st
import os
from dotenv import load_dotenv
import requests
from requests_oauthlib import OAuth1Session
from huggingface_hub import InferenceClient
import pandas as pd

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("HF_TOKEN not found in .env file!")

# -------------------------------
# Twitter API Credentials (Hardcoded for MVP)
# -------------------------------
bearer_token = "AAAAAAAAAAAAAAAAAAAAALW%2F3gEAAAAACbuBHpkCKh5FNKW1xXLPdBZAmk4%3DogmOnHyhqONUWNhrEitUZgXpFYUliPZgmEUcmi8jv99FlV0A1u"
consumer_key = "1760306826262794242-TpU2JlTTm095iz0E5uzGffjmBt60yk"
consumer_secret = "etApPK2m0rhbTaVwiFcf1XPkjV4oFbgY8TmHf3zwvAzol"
oauth_token = "1760306826262794242-rGqTedDjHT8Qr7D7UTKxPSL0OoKNVU"
oauth_token_secret = "dKqswOOJMEkj0RZ1mhlV4K4iVUAp42oiQRQHvUaDhAB4S"

# -------------------------------
# Hugging Face Client
# -------------------------------
hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
nlp_model = "HuggingFaceTB/SmolLM3-3B"

# -------------------------------
# Twitter Functions
# -------------------------------
def fetch_tweets(query, max_results=10):
    """
    Fetch recent tweets using the updated X API (api.x.com).
    """
    search_url = "https://api.x.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {
        "query": query,
        "max_results": max_results,
        "tweet.fields": "id,text,author_id,created_at,public_metrics"
    }

    response = requests.get(search_url, headers=headers, params=params)

    if response.status_code != 200:
        st.error(f"❌ Error fetching tweets: {response.status_code} {response.text}")
        return []
    
    return response.json().get("data", [])


def post_reply(text, tweet_id):
    """
    Reply to a tweet using OAuth1Session.
    """
    oauth = OAuth1Session(
        client_key=consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=oauth_token,
        resource_owner_secret=oauth_token_secret
    )
    payload = {"text": text, "in_reply_to_tweet_id": tweet_id}
    response = oauth.post("https://api.x.com/2/tweets", json=payload)
    return response.status_code, response.json()

# -------------------------------
# NLP Functions
# -------------------------------
def detect_topic(tweet_text):
    prompt = f"Classify this tweet into topics: investment, crypto, commodities, forex, market analysis. Tweet: {tweet_text}"
    completion = hf_client.chat.completions.create(
        model=nlp_model,
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message


def generate_comment(tweet_text, topic):
    prompt = f"Generate 3 professional comment options for this tweet based on topic '{topic}': {tweet_text}"
    completion = hf_client.chat.completions.create(
        model=nlp_model,
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message
