# streamlit_dashboard.py
import streamlit as st
import os
import requests
from requests_oauthlib import OAuth1Session
from huggingface_hub import InferenceClient
import pandas as pd
import time

# -------------------------------
# Environment / API Setup
# -------------------------------
# Twitter API credentials
bearer_token = os.environ.get("BEARER_TOKEN")
consumer_key = os.environ.get("CONSUMER_KEY")
consumer_secret = os.environ.get("CONSUMER_SECRET")
oauth_token = os.environ.get("OAUTH_TOKEN")           # after OAuth flow
oauth_token_secret = os.environ.get("OAUTH_TOKEN_SECRET") # after OAuth flow

# Hugging Face Inference API
hf_client = InferenceClient(provider="hf-inference", api_key=os.environ.get("HF_TOKEN"))
nlp_model = "HuggingFaceTB/SmolLM3-3B"

# Predefined topics/concepts
concepts = ["investment", "crypto", "commodities", "forex", "market analysis"]

# -------------------------------
# Twitter Functions
# -------------------------------
def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

def fetch_tweets(query, max_results=10):
    search_url = "https://api.twitter.com/2/tweets/search/recent"
    query_params = {
        'query': query,
        'max_results': max_results,
        'tweet.fields': 'author_id,created_at,public_metrics'
    }
    response = requests.get(search_url, auth=bearer_oauth, params=query_params)
    if response.status_code != 200:
        st.error(f"Error fetching tweets: {response.status_code}")
        return []
    return response.json().get("data", [])

def post_reply(text, tweet_id):
    oauth = OAuth1Session(
        consumer_key,
        client_secret=consumer_secret,
        resource_owner_key=oauth_token,
        resource_owner_secret=oauth_token_secret
    )
    payload = {"text": text, "in_reply_to_tweet_id": tweet_id}
    response = oauth.post("https://api.twitter.com/2/tweets", json=payload)
    return response.status_code, response.json()

def fetch_metrics(tweet_id):
    url = f"https://api.twitter.com/2/tweets/{tweet_id}?tweet.fields=public_metrics"
    response = requests.get(url, auth=bearer_oauth)
    if response.status_code != 200:
        return None
    return response.json().get("data", {}).get("public_metrics", {})

# -------------------------------
# NLP Functions
# -------------------------------
def detect_topic(tweet_text):
    prompt = f"Classify this tweet into the following topics: {concepts}. Tweet: {tweet_text}"
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

def generate_followup(tweet_text, engagement_data):
    prompt = f"Based on engagement {engagement_data}, generate a follow-up reply or DM for the tweet: {tweet_text}"
    completion = hf_client.chat.completions.create(
        model=nlp_model,
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Social Media Networking Dashboard", layout="wide")
st.title("Social Media Networking Dashboard")

# --- Sidebar ---
st.sidebar.header("Fetch Tweets")
query_input = st.sidebar.text_input("Search Query or @username", "@twitterdev")
max_results = st.sidebar.slider("Max Results", 5, 20, 10)
fetch_button = st.sidebar.button("Fetch Tweets")

# --- Main Workflow ---
if fetch_button:
    st.info("Fetching tweets...")
    tweets = fetch_tweets(query_input, max_results)
    
    if tweets:
        tweet_data = []
        for tweet in tweets:
            tweet_id = tweet['id']
            text = tweet['text']
            metrics = tweet.get('public_metrics', {})
            topic = detect_topic(text)
            suggested_comments = generate_comment(text, topic)
            
            tweet_data.append({
                "Tweet ID": tweet_id,
                "Text": text,
                "Topic": topic,
                "Suggested Comments": suggested_comments,
                "Likes": metrics.get("like_count", 0),
                "Retweets": metrics.get("retweet_count", 0),
                "Replies": metrics.get("reply_count", 0)
            })
        
        df = pd.DataFrame(tweet_data)
        st.subheader("Fetched Tweets & NLP Analysis")
        st.dataframe(df)
        
        # --- Select tweet to reply ---
        selected_tweet_id = st.selectbox("Select Tweet to Reply", df["Tweet ID"])
        selected_comments = df[df["Tweet ID"]==selected_tweet_id]["Suggested Comments"].values[0]
        st.text_area("Suggested Comments", value=selected_comments, height=100)
        reply_text = st.text_area("Edit / Write Your Comment Here", height=100)
        
        if st.button("Post Reply"):
            status, response = post_reply(reply_text, selected_tweet_id)
            if status == 201:
                st.success("Reply posted successfully!")
            else:
                st.error(f"Error posting reply: {status}")
        
        # --- Monitor Engagement ---
        st.subheader("Monitor Engagement")
        if st.button("Refresh Metrics"):
            updated_metrics = fetch_metrics(selected_tweet_id)
            st.json(updated_metrics)
        
        # --- Generate Follow-Up ---
        if st.button("Generate Follow-Up"):
            engagement_data = fetch_metrics(selected_tweet_id)
            followup_text = generate_followup(reply_text, engagement_data)
            st.text_area("Suggested Follow-Up", value=followup_text, height=100)
