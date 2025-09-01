# streamlit_dashboard_buttons.py
import streamlit as st
import os
from dotenv import load_dotenv
import requests
import json
import pandas as pd
import logging
from huggingface_hub import InferenceClient
from requests_oauthlib import OAuth2Session
import time

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
CLIENT_ID = os.getenv("TWITTER_OAUTH2_CLIENT_ID")
CLIENT_SECRET = os.getenv("TWITTER_OAUTH2_CLIENT_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_OAUTH2_ACCESS_TOKEN")
REFRESH_TOKEN = os.getenv("TWITTER_OAUTH2_REFRESH_TOKEN")
CALLBACK_URI = os.getenv("TWITTER_CALLBACK_URI")

# -------------------------------
# Initialize HuggingFace client
# -------------------------------
hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

# -------------------------------
# NLP Model & Concepts
# -------------------------------
nlp_model = "HuggingFaceTB/SmolLM3-3B"
concepts = ["investment", "crypto", "commodities", "forex", "market analysis"]

# -------------------------------
# OAuth2 Token Management
# -------------------------------
def refresh_oauth2_token(client_id, client_secret, refresh_token):
    """Refresh OAuth2 access token using refresh token"""
    logging.info("Refreshing OAuth2 token")
    token_url = "https://api.twitter.com/2/oauth2/token"
    
    data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token
    }
    
    response = requests.post(token_url, data=data)
    
    if response.status_code == 200:
        tokens = response.json()
        logging.info("OAuth2 token refreshed successfully")
        
        # Update environment variables
        os.environ["TWITTER_OAUTH2_ACCESS_TOKEN"] = tokens['access_token']
        if 'refresh_token' in tokens:
            os.environ["TWITTER_OAUTH2_REFRESH_TOKEN"] = tokens['refresh_token']
            
        return tokens['access_token'], tokens.get('refresh_token', refresh_token)
    else:
        logging.error(f"Failed to refresh token: {response.status_code} {response.text}")
        st.error("Failed to refresh authentication token. Please check your credentials.")
        return None, None

def get_auth_session():
    """Get an authenticated OAuth2 session"""
    access_token = os.environ.get("TWITTER_OAUTH2_ACCESS_TOKEN")
    refresh_token = os.environ.get("TWITTER_OAUTH2_REFRESH_TOKEN")
    
    if not access_token or not refresh_token:
        st.error("OAuth2 tokens not found. Please set up authentication first.")
        return None
    
    # Check if token needs refreshing by making a test request
    auth_headers = {"Authorization": f"Bearer {access_token}"}
    test_response = requests.get("https://api.twitter.com/2/users/me", headers=auth_headers)
    
    if test_response.status_code == 401:
        # Token expired, refresh it
        new_access_token, new_refresh_token = refresh_oauth2_token(CLIENT_ID, CLIENT_SECRET, refresh_token)
        if not new_access_token:
            return None
        access_token = new_access_token
        
    return access_token

# -------------------------------
# Helper Functions
# -------------------------------
def fetch_tweets(query):
    """Fetch tweets based on search query"""
    max_results = 15
    logging.info(f"Fetching {max_results} tweets for query: '{query}'")
    
    access_token = get_auth_session()
    if not access_token:
        return []
    
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {
        "query": query,
        "max_results": max_results,
        "tweet.fields": "author_id,created_at,public_metrics,text"
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        logging.error(f"Error fetching tweets: {response.status_code} {response.text}")
        st.error(f"Error fetching tweets: {response.status_code}")
        return []
    
    tweets = response.json().get("data", [])
    logging.info(f"Fetched {len(tweets)} tweets")
    return tweets

def detect_topic(tweet_text):
    """Detect the topic of a tweet using the NLP model"""
    logging.info(f"Detecting topic for tweet: {tweet_text[:50]}...")
    prompt = f"Classify this tweet into ONE of these topics: {concepts}. Tweet: {tweet_text}"
    completion = hf_client.chat.completions.create(
        model=nlp_model, 
        messages=[{"role": "user", "content": prompt}]
    )
    topic = completion.choices[0].message.content.strip()
    logging.info(f"Detected topic: {topic}")
    return topic

def generate_comment(tweet_text, topic):
    """Generate comment suggestions for a tweet"""
    logging.info(f"Generating comment for topic '{topic}'")
    prompt = f"Generate 3 professional comment options for this tweet based on topic '{topic}'. Each comment should be concise (max 280 chars) and engaging: {tweet_text}"
    completion = hf_client.chat.completions.create(
        model=nlp_model, 
        messages=[{"role": "user", "content": prompt}]
    )
    comments = completion.choices[0].message.content
    logging.info(f"Generated comments: {comments[:50]}...")
    return comments

def post_reply(text, tweet_id):
    """Post a reply to a tweet"""
    logging.info(f"Posting reply to tweet_id={tweet_id}")
    
    access_token = get_auth_session()
    if not access_token:
        return 401, {"error": "Authentication failed"}
    
    url = "https://api.twitter.com/2/tweets"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "text": text,
        "reply": {"in_reply_to_tweet_id": tweet_id}
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 201:
        logging.info(f"Reply posted successfully to tweet_id={tweet_id}")
        return response.status_code, response.json()
    else:
        logging.error(f"Failed to post reply: {response.status_code} {response.text}")
        return response.status_code, response.json() if response.text else {"error": "Unknown error"}

def fetch_metrics(tweet_id):
    """Fetch the metrics for a tweet"""
    logging.info(f"Fetching metrics for tweet_id={tweet_id}")
    
    access_token = get_auth_session()
    if not access_token:
        return None
    
    url = f"https://api.twitter.com/2/tweets/{tweet_id}?tweet.fields=public_metrics"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        logging.error(f"Error fetching metrics for tweet_id={tweet_id}: {response.status_code}")
        return None
    
    metrics = response.json().get("data", {}).get("public_metrics", {})
    logging.info(f"Metrics: {metrics}")
    return metrics

def generate_followup(tweet_text, engagement_data):
    """Generate a follow-up reply based on engagement data"""
    logging.info(f"Generating follow-up for tweet based on engagement {engagement_data}")
    
    # Create a user-friendly description of the engagement
    engagement_desc = "This tweet has "
    if engagement_data:
        if "retweet_count" in engagement_data:
            engagement_desc += f"{engagement_data['retweet_count']} retweets, "
        if "reply_count" in engagement_data:
            engagement_desc += f"{engagement_data['reply_count']} replies, "
        if "like_count" in engagement_data:
            engagement_desc += f"{engagement_data['like_count']} likes, "
        if "impression_count" in engagement_data:
            engagement_desc += f"{engagement_data['impression_count']} impressions, "
        engagement_desc = engagement_desc.rstrip(", ")
    else:
        engagement_desc += "no engagement yet."
    
    prompt = f"Based on engagement metrics ({engagement_desc}), generate a follow-up reply for the tweet: {tweet_text}. Keep it under 280 characters and make it engaging."
    
    completion = hf_client.chat.completions.create(
        model=nlp_model, 
        messages=[{"role": "user", "content": prompt}]
    )
    followup = completion.choices[0].message.content
    logging.info(f"Follow-up: {followup[:50]}...")
    return followup

# -------------------------------
# Authentication Setup UI
# -------------------------------
def setup_auth_ui():
    """UI for setting up OAuth2 authentication"""
    st.subheader("Authentication Setup")
    
    with st.form("oauth_setup"):
        client_id = st.text_input("Client ID", value=os.environ.get("TWITTER_OAUTH2_CLIENT_ID", ""))
        client_secret = st.text_input("Client Secret", type="password", value=os.environ.get("TWITTER_OAUTH2_CLIENT_SECRET", ""))
        access_token = st.text_input("Access Token", value=os.environ.get("TWITTER_OAUTH2_ACCESS_TOKEN", ""))
        refresh_token = st.text_input("Refresh Token", value=os.environ.get("TWITTER_OAUTH2_REFRESH_TOKEN", ""))
        callback_uri = st.text_input("Callback URI", value=os.environ.get("TWITTER_CALLBACK_URI", ""))
        
        submitted = st.form_submit_button("Save Credentials")
        
        if submitted:
            os.environ["TWITTER_OAUTH2_CLIENT_ID"] = client_id
            os.environ["TWITTER_OAUTH2_CLIENT_SECRET"] = client_secret
            os.environ["TWITTER_OAUTH2_ACCESS_TOKEN"] = access_token
            os.environ["TWITTER_OAUTH2_REFRESH_TOKEN"] = refresh_token
            os.environ["TWITTER_CALLBACK_URI"] = callback_uri
            
            st.success("Credentials saved!")
            
            # Test the credentials
            test_access_token = get_auth_session()
            if test_access_token:
                st.success("Authentication successful!")
            else:
                st.error("Authentication failed. Please check your credentials.")

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("EngageFlow: Social Media Networking Dashboard")

# Add a sidebar with tabs
tab_options = ["Tweet Engagement", "Authentication Setup"]
selected_tab = st.sidebar.radio("Navigation", tab_options)

if selected_tab == "Authentication Setup":
    setup_auth_ui()
else:
    # Main tweet engagement workflow
    query_input = st.text_input("Search Query or @username", "@twitterdev")

    # Initialize session state variables if they don't exist
    if 'tweets' not in st.session_state:
        st.session_state['tweets'] = []
    if 'selected_tweet_id' not in st.session_state:
        st.session_state['selected_tweet_id'] = None

    # --- 1️⃣ Fetch Tweets ---
    if st.button("Fetch Tweets"):
        with st.spinner("Fetching tweets..."):
            st.session_state['tweets'] = fetch_tweets(query_input)
            logging.info("Fetch Tweets button clicked")

    tweets = st.session_state.get('tweets', [])

    if tweets:
        st.subheader("Fetched Tweets")
        
        # Create a more visually appealing dataframe
        df = pd.DataFrame([{
            "Tweet ID": t['id'], 
            "Text": t['text'], 
            "Topic": t.get('topic', 'Not analyzed')
        } for t in tweets])
        
        st.dataframe(df)

        # --- 2️⃣ Detect Topic ---
        if st.button("Detect Topics for All Tweets"):
            progress_bar = st.progress(0)
            for i, t in enumerate(tweets):
                with st.spinner(f"Analyzing tweet {i+1}/{len(tweets)}..."):
                    t['topic'] = detect_topic(t['text'])
                    progress_bar.progress((i + 1) / len(tweets))
            
            logging.info("Detect Topic button clicked")
            st.success("Topics detected for all tweets")
            
            # Update the dataframe
            df = pd.DataFrame([{
                "Tweet ID": t['id'], 
                "Text": t['text'], 
                "Topic": t.get('topic', 'Not analyzed')
            } for t in tweets])
            
            st.dataframe(df)

        # --- 3️⃣ Generate Comment ---
        st.subheader("Generate and Post Comments")
        
        tweet_ids = [t['id'] for t in tweets]
        selected_tweet_id = st.selectbox(
            "Select Tweet to Comment", 
            tweet_ids,
            format_func=lambda x: next((t['text'][:50] + "..." for t in tweets if t['id'] == x), x)
        )
        
        st.session_state['selected_tweet_id'] = selected_tweet_id
        selected_tweet = next((t for t in tweets if t['id'] == selected_tweet_id), None)

        if selected_tweet:
            st.markdown(f"**Selected Tweet:**  \n{selected_tweet['text']}")
            
            # Show the topic if available
            if 'topic' in selected_tweet:
                st.markdown(f"**Detected Topic:** {selected_tweet['topic']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Generate Comment Suggestions"):
                    with st.spinner("Generating comments..."):
                        topic = selected_tweet.get('topic', 'General')
                        selected_tweet['suggested_comments'] = generate_comment(selected_tweet['text'], topic)
                        logging.info(f"Generate Comment button clicked for tweet_id={selected_tweet_id}")
            
            with col2:
                if st.button("Refresh Metrics"):
                    with st.spinner("Fetching metrics..."):
                        metrics = fetch_metrics(selected_tweet_id)
                        selected_tweet['metrics'] = metrics
                        logging.info(f"Refresh Metrics button clicked for tweet_id={selected_tweet_id}")
            
            # Display suggested comments if available
            if 'suggested_comments' in selected_tweet:
                st.subheader("Suggested Comments")
                st.markdown(selected_tweet['suggested_comments'])
            
            # Display metrics if available
            if 'metrics' in selected_tweet and selected_tweet['metrics']:
                st.subheader("Tweet Metrics")
                metrics = selected_tweet['metrics']
                cols = st.columns(len(metrics))
                for i, (metric, value) in enumerate(metrics.items()):
                    cols[i].metric(metric.replace('_', ' ').title(), value)
            
            # --- 4️⃣ Post Reply ---
            st.subheader("Post Your Reply")
            reply_text = st.text_area(
                "Edit / Write Your Comment Here", 
                height=100,
                value=selected_tweet.get('selected_comment', '')
            )
            
            if reply_text:
                char_count = len(reply_text)
                st.caption(f"Character count: {char_count}/280")
                
                if char_count > 280:
                    st.warning("Your reply exceeds the 280 character limit!")
            
            if st.button("Post Reply", disabled=not reply_text or len(reply_text) > 280):
                with st.spinner("Posting reply..."):
                    status, response = post_reply(reply_text, selected_tweet_id)
                    if status == 201:
                        st.success("Reply posted successfully!")
                        st.json(response)
                    else:
                        st.error(f"Error posting reply: {status}")
                        st.json(response)

            # --- 6️⃣ Generate Follow-Up ---
            if 'metrics' in selected_tweet and selected_tweet['metrics']:
                if st.button("Generate Follow-Up Comment"):
                    with st.spinner("Generating follow-up..."):
                        followup_text = generate_followup(selected_tweet['text'], selected_tweet['metrics'])
                        selected_tweet['followup_text'] = followup_text
                        logging.info(f"Generate Follow-Up button clicked for tweet_id={selected_tweet_id}")
                
                if 'followup_text' in selected_tweet:
                    st.subheader("Suggested Follow-Up")
                    st.markdown(selected_tweet['followup_text'])
                    
                    if st.button("Use This Follow-Up"):
                        selected_tweet['selected_comment'] = selected_tweet['followup_text']
                        st.experimental_rerun()
