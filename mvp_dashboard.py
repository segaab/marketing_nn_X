# streamlit_dashboard_buttons.py
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
# Load environment variables (HF Token only)
# -------------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

# -------------------------------
# Twitter Credentials (Updated Hardcoded)
# -------------------------------
bearer_token = "AAAAAAAAAAAAAAAAAAAALW%2F3gEAAAAAeA2XC2Sr0T08tk0Y5ZcXfwedwK8%3D9fgXU3xWrQGTJGZAAzzUalV1ePOFFAa4JWllkZl3T5eYA6bKCS"
consumer_key = "CKkhTfjQEmP20GYFlF3gVwe98"
consumer_secret = "8TPzOOxB2eEui28pO6u1pXthlUBNd4wpspQsSAEvHgEQkb7tsR"
oauth_token = "1760306826262794242-Id0C3xho2pl4h9k5aJg1knb3wt3kkh"
oauth_token_secret = "d2dPP8LzBj3ar6bYH7abDfocwmr7zA9a57lDXZuoveXE1"

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

def fetch_tweets(query):
    max_results = 15  # Fixed number of posts
    logging.info(f"Fetching {max_results} tweets for query: '{query}'")
    url = "https://api.x.com/2/tweets/search/recent"
    params = {
        "query": query,
        "max_results": max_results,
        "tweet.fields": "author_id,created_at,public_metrics,text"
    }
    response = requests.get(url, headers={"Authorization": f"Bearer {bearer_token}"}, params=params)
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
    completion = hf_client.chat.completions.create(model=nlp_model, messages=[{"role": "user", "content": prompt}])
    topic = completion.choices[0].message
    logging.info(f"Detected topic: {topic}")
    return topic

def generate_comment(tweet_text, topic):
    logging.info(f"Generating comment for topic '{topic}'")
    prompt = f"Generate 3 professional comment options for this tweet based on topic '{topic}': {tweet_text}"
    completion = hf_client.chat.completions.create(model=nlp_model, messages=[{"role": "user", "content": prompt}])
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
    url = f"https://api.x.com/2/tweets/{tweet_id}?tweet.fields=public_metrics"
    response = requests.get(url, headers={"Authorization": f"Bearer {bearer_token}"})
    if response.status_code != 200:
        logging.error(f"Error fetching metrics for tweet_id={tweet_id}: {response.status_code}")
        return None
    metrics = response.json().get("data", {}).get("public_metrics", {})
    logging.info(f"Metrics: {metrics}")
    return metrics

def generate_followup(tweet_text, engagement_data):
    logging.info(f"Generating follow-up for tweet based on engagement {engagement_data}")
    prompt = f"Based on engagement {engagement_data}, generate a follow-up reply for the tweet: {tweet_text}"
    completion = hf_client.chat.completions.create(model=nlp_model, messages=[{"role": "user", "content": prompt}])
    followup = completion.choices[0].message
    logging.info(f"Follow-up: {followup[:50]}...")
    return followup

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("EngageFlow: Social Media Networking Dashboard")

query_input = st.text_input("Search Query or @username", "@twitterdev")

# --- 1️⃣ Fetch Tweets ---
if st.button("Fetch Tweets"):
    st.session_state['tweets'] = fetch_tweets(query_input)
    logging.info("Fetch Tweets button clicked")

tweets = st.session_state.get('tweets', [])

if tweets:
    st.subheader("Fetched Tweets")
    df = pd.DataFrame([{"Tweet ID": t['id'], "Text": t['text']} for t in tweets])
    st.dataframe(df)

    # --- 2️⃣ Detect Topic ---
    if st.button("Detect Topic"):
        for t in tweets:
            t['topic'] = detect_topic(t['text'])
        logging.info("Detect Topic button clicked")
        st.success("Topics detected")

    # --- 3️⃣ Generate Comment ---
    selected_tweet_id = st.selectbox("Select Tweet to Comment", [t['id'] for t in tweets])
    selected_tweet = next((t for t in tweets if t['id']==selected_tweet_id), None)

    if selected_tweet and st.button("Generate Comment"):
        selected_tweet['suggested_comments'] = generate_comment(selected_tweet['text'], selected_tweet.get('topic', 'General'))
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
