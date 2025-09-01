# streamlit_dashboard_debug.py
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
    format='%(asctime)s [%(levelname)s]: %(message)s',
    handlers=[logging.FileHandler("engageflow.log"), logging.StreamHandler()]
)

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

# -------------------------------
# Twitter Credentials (Hardcoded for MVP)
# -------------------------------
bearer_token = "AAAAAAAAAAAAAAAAAAAAALW%2F3gEAAAAACbuBHpkCKh5FNKW1xXLPdBZAmk4%3DogmOnHyhqONUWNhrEitUZgXpFYUliPZgmEUcmi8jv99FlV0A1u"
consumer_key = "1760306826262794242-TpU2JlTTm095iz0E5uzGffjmBt60yk"
consumer_secret = "etApPK2m0rhbTaVwiFcf1XPkjV4oFbgY8TmHf3zwvAzol"
oauth_token = "1760306826262794242-rGqTedDjHT8Qr7D7UTKxPSL0OoKNVU"
oauth_token_secret = "dKqswOOJMEkj0RZ1mhlV4K4iVUAp42oiQRQHvUaDhAB4S"

# -------------------------------
# NLP Model & Concepts
# -------------------------------
nlp_model = "HuggingFaceTB/SmolLM3-3B"
concepts = ["investment", "crypto", "commodities", "forex", "market analysis"]

# -------------------------------
# Helper Functions
# -------------------------------
def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

def fetch_tweets(query, max_results=10):
    logging.info(f"Fetching tweets for query: '{query}'")
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": query,
        "max_results": max_results,
        "tweet.fields": "author_id,created_at,public_metrics"
    }
    response = requests.get(url, auth=bearer_oauth, params=params)
    logging.info(f"Twitter API status: {response.status_code}")
    logging.info(f"Response content: {response.text[:300]}")  # truncated for log

    if response.status_code != 200:
        logging.error(f"Error fetching tweets: {response.status_code} {response.text}")
        st.error(f"Error fetching tweets: {response.status_code}")
        return []

    tweets = response.json().get("data", [])
    logging.info(f"Fetched {len(tweets)} tweets")
    return tweets

def detect_topic(tweet_text):
    logging.info(f"Detecting topic for tweet: {tweet_text[:50]}...")
    prompt = f"Classify this tweet into topics: {concepts}. Tweet: {tweet_text}"
    completion = hf_client.chat.completions.create(
        model=nlp_model,
        messages=[{"role": "user", "content": prompt}]
    )
    topic = completion.choices[0].message
    logging.info(f"Detected topic: {topic}")
    return topic

def generate_comment(tweet_text, topic):
    logging.info(f"Generating comment for topic '{topic}'")
    prompt = f"Generate 3 professional comment options for this tweet based on topic '{topic}': {tweet_text}"
    completion = hf_client.chat.completions.create(
        model=nlp_model,
        messages=[{"role": "user", "content": prompt}]
    )
    comments = completion.choices[0].message
    logging.info(f"Generated comments: {comments[:50]}...")
    return comments

def post_reply(text, tweet_id):
    logging.info(f"Posting reply to tweet_id={tweet_id}")
    oauth = OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=oauth_token,
        resource_owner_secret=oauth_token_secret
    )
    payload = {"text": text, "in_reply_to_tweet_id": tweet_id}
    response = oauth.post("https://api.twitter.com/2/tweets", json=payload)
    if response.status_code == 201:
        logging.info(f"Reply posted successfully to tweet_id={tweet_id}")
    else:
        logging.error(f"Failed to post reply: {response.status_code} {response.text}")
    return response.status_code, response.json()

def fetch_metrics(tweet_id):
    logging.info(f"Fetching metrics for tweet_id={tweet_id}")
    url = f"https://api.twitter.com/2/tweets/{tweet_id}?tweet.fields=public_metrics"
    response = requests.get(url, auth=bearer_oauth)
    if response.status_code != 200:
        logging.error(f"Error fetching metrics for tweet_id={tweet_id}")
        return None
    metrics = response.json().get("data", {}).get("public_metrics", {})
    logging.info(f"Metrics: {metrics}")
    return metrics

def generate_followup(tweet_text, engagement_data):
    logging.info(f"Generating follow-up for tweet based on engagement {engagement_data}")
    prompt = f"Based on engagement {engagement_data}, generate a follow-up reply for the tweet: {tweet_text}"
    completion = hf_client.chat.completions.create(
        model=nlp_model,
        messages=[{"role": "user", "content": prompt}]
    )
    followup = completion.choices[0].message
    logging.info(f"Follow-up: {followup[:50]}...")
    return followup

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("EngageFlow: Social Media Networking Dashboard")

query_input = st.text_input("Search Query or @username", "@twitterdev")
max_results = st.slider("Max Results", 5, 20, 10)

# --- 1️⃣ Fetch Tweets ---
if st.button("Fetch Tweets"):
    fetched = fetch_tweets(query_input, max_results)
    st.session_state['tweets'] = fetched
    logging.info("Fetch Tweets button clicked")
    st.write("Raw fetched tweets:", fetched)

tweets = st.session_state.get('tweets', [])

if tweets:
    st.subheader("Fetched Tweets")
    df = pd.DataFrame([{"Tweet ID": t['id'], "Text": t['text']} for t in tweets])
    st.dataframe(df)

    # --- 2️⃣ Detect Topic ---
    if st.button("Detect Topic"):
        for t in tweets:
            t['topic'] = detect_topic(t['text'])
        st.session_state['tweets'] = tweets
        logging.info("Detect Topic button clicked")
        st.success("Topics detected for all tweets")
        st.write("Tweets with topics:", tweets)

    # --- 3️⃣ Generate Comment ---
    selected_tweet_id = st.selectbox("Select Tweet to Comment", [t['id'] for t in tweets])
    selected_tweet = next((t for t in tweets if t['id']==selected_tweet_id), None)

    if selected_tweet and st.button("Generate Comment"):
        selected_tweet['suggested_comments'] = generate_comment(
            selected_tweet['text'], selected_tweet.get('topic', 'General')
        )
        st.session_state['tweets'] = tweets
        logging.info(f"Generate Comment button clicked for tweet_id={selected_tweet_id}")
        st.text_area("Suggested Comments", value=selected_tweet['suggested_comments'], height=100)

    # --- 4️⃣ Post Reply ---
    reply_text = st.text_area("Edit / Write Your Comment Here", height=100)
    if st.button("Post Reply"):
        status, response = post_reply(reply_text, selected_tweet_id)
        if status == 201:
            st.success("Reply posted successfully!")
        else:
            st.error(f"Error posting reply: {status} {response}")

    # --- 5️⃣ Refresh Metrics ---
    if st.button("Refresh Metrics"):
        metrics = fetch_metrics(selected_tweet_id)
        st.json(metrics)
        logging.info(f"Refresh Metrics button clicked for tweet_id={selected_tweet_id}")

    # --- 6️⃣ Generate Follow-Up ---
    if st.button("Generate Follow-Up"):
        engagement_data = fetch_metrics(selected_tweet_id)
        followup_text = generate_followup(reply_text, engagement_data)
        st.text_area("Suggested Follow-Up", value=followup_text, height=100)
        logging.info(f"Generate Follow-Up button clicked for tweet_id={selected_tweet_id}")

else:
    st.info("No tweets fetched yet. Press 'Fetch Tweets' to begin.")
