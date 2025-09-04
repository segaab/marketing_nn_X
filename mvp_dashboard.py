# streamlit_dashboard_buttons.py
import os
import json
import time
import logging
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from requests_oauthlib import OAuth1Session
from huggingface_hub import InferenceClient

# ==============================
# Logging
# ==============================
LOG_FILE = "engageflow.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)

# ==============================
# Load Environment
# ==============================
load_dotenv()

ENV_MAP = {
    "X_BEARER_TOKEN": os.getenv("X_BEARER_TOKEN"),
    "X_API_KEY": os.getenv("X_API_KEY"),
    "X_API_SECRET": os.getenv("X_API_SECRET"),
    "X_ACCESS_TOKEN": os.getenv("X_ACCESS_TOKEN"),
    "X_ACCESS_SECRET": os.getenv("X_ACCESS_SECRET"),
    "X_CLIENT_ID": os.getenv("X_CLIENT_ID"),
    "X_CLIENT_SECRET": os.getenv("X_CLIENT_SECRET"),
}

HF_TOKEN = os.getenv("HF_TOKEN")
hf_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

NLP_MODEL = "HuggingFaceTB/SmolLM3-3B"
CONCEPTS = ["investment", "crypto", "commodities", "forex", "market analysis"]

# ==============================
# API Usage Tracking
# ==============================
class APIUsageTracker:
    def __init__(self):
        self.read_count = 0
        self.post_count = 0
        self.free_tier_read_limit = 100  # Free tier: 100 reads per month
        self.free_tier_post_limit = 500  # Free tier: 500 posts per month
        
    def log_read(self):
        self.read_count += 1
        logging.info(f"API READ: {self.read_count}/{self.free_tier_read_limit} for this month")
        
    def log_post(self):
        self.post_count += 1
        logging.info(f"API POST: {self.post_count}/{self.free_tier_post_limit} for this month")
        
    def can_read(self):
        return self.read_count < self.free_tier_read_limit
    
    def can_post(self):
        return self.post_count < self.free_tier_post_limit

# Initialize tracker
api_tracker = APIUsageTracker()

# ==============================
# Session State
# ==============================
def _seed_session_from_env():
    for k, v in ENV_MAP.items():
        st.session_state.setdefault(k.lower(), v)
    st.session_state.setdefault("tweets", [])
    st.session_state.setdefault("selected_tweet_id", None)
    st.session_state.setdefault("cached_metrics", {})
    st.session_state.setdefault("api_read_count", 0)
    st.session_state.setdefault("api_post_count", 0)

_seed_session_from_env()

# ==============================
# Auth Helpers
# ==============================
def bearer_headers():
    token = st.session_state.get("x_bearer_token")
    if not token:
        st.error("Missing Bearer Token in .env")
        return None
    return {"Authorization": f"Bearer {token}"}

def oauth1_session():
    api_key = st.session_state.get("x_api_key")
    api_secret = st.session_state.get("x_api_secret")
    access_token = st.session_state.get("x_access_token")
    access_secret = st.session_state.get("x_access_secret")

    if not all([api_key, api_secret, access_token, access_secret]):
        st.error("Missing OAuth1.0a credentials in .env")
        return None

    return OAuth1Session(
        client_key=api_key,
        client_secret=api_secret,
        resource_owner_key=access_token,
        resource_owner_secret=access_secret,
    )

# ==============================
# Rate Limit Handling
# ==============================
def handle_rate_limits(resp):
    """Handle rate limiting and log remaining limits."""
    remaining = resp.headers.get("x-rate-limit-remaining")
    reset = resp.headers.get("x-rate-limit-reset")
    
    if remaining and reset:
        remaining = int(remaining)
        reset_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(reset)))
        logging.info(f"Rate limits → Remaining: {remaining}, Reset: {reset_time}")
        
        if remaining < 5:
            st.warning(f"⚠️ API rate limit almost reached! Resets at {reset_time}")
            
    return resp

def backoff_and_retry(func):
    """Decorator to implement exponential backoff for rate limiting."""
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_count = 0
        base_wait = 5
        
        while retry_count < max_retries:
            try:
                resp = func(*args, **kwargs)
                if resp.status_code == 429:  # Rate limited
                    retry_count += 1
                    wait_time = base_wait * (2 ** retry_count)
                    logging.warning(f"Rate limited. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return resp
            except Exception as e:
                logging.error(f"API error: {str(e)}")
                raise
                
        logging.error("Max retries reached")
        return None
    return wrapper

# ==============================
# API Functions (Modified for Free Tier)
# ==============================
def fetch_latest_tweet_ids(username: str, count=3):
    """
    Fetch the latest tweet IDs for a username using user timeline endpoint.
    Note: Free tier has very limited search capabilities, using simplified approach.
    """
    if not api_tracker.can_read():
        st.error("Monthly API read limit reached (100/month). Consider upgrading to Basic tier.")
        return []
        
    url = f"https://api.x.com/2/users/by/username/{username}/tweets"
    headers = bearer_headers()
    if headers is None:
        return []

    # Using smaller max_results to conserve API calls
    params = {"max_results": min(count, 5), "tweet.fields": "id"}
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp = handle_rate_limits(resp)
        api_tracker.log_read()
        
        if resp.status_code != 200:
            logging.error(f"Fetch IDs error {resp.status_code}: {resp.text}")
            st.error(f"Error fetching tweet IDs: {resp.status_code}")
            return []
            
        data = resp.json().get("data", [])
        return [t["id"] for t in data]
    except Exception as e:
        logging.error(f"Exception fetching tweet IDs: {str(e)}")
        st.error("Failed to fetch tweets. Check connection and API credentials.")
        return []

def lookup_tweets_by_ids(tweet_ids):
    """
    Lookup tweets with details using Tweet Lookup endpoint.
    Modified to work within Free tier constraints.
    """
    if not tweet_ids:
        return []
        
    if not api_tracker.can_read():
        st.error("Monthly API read limit reached (100/month). Consider upgrading to Basic tier.")
        return []

    url = "https://api.x.com/2/tweets"
    headers = bearer_headers()
    
    # Limit to maximum 10 IDs per request to conserve API calls
    # Free tier is extremely limited
    ids_param = ",".join(tweet_ids[:min(len(tweet_ids), 10)])

    params = {
        "ids": ids_param,
        "tweet.fields": "author_id,created_at,public_metrics,text",
        "expansions": "author_id",
        "user.fields": "username,name",
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp = handle_rate_limits(resp)
        api_tracker.log_read()
        
        if resp.status_code != 200:
            logging.error(f"Lookup error {resp.status_code}: {resp.text}")
            st.error(f"Error fetching tweets: {resp.status_code}")
            return []
            
        return resp.json().get("data", [])
    except Exception as e:
        logging.error(f"Exception looking up tweets: {str(e)}")
        st.error("Failed to fetch tweet details. Check connection and API credentials.")
        return []

# ==============================
# NLP Helpers
# ==============================
def detect_topic(tweet_text: str) -> str:
    logging.info(f"Detecting topic for tweet: {tweet_text[:50]}...")
    prompt = f"Classify this tweet into ONE of these topics: {CONCEPTS}. Tweet: {tweet_text}"
    try:
        completion = hf_client.chat.completions.create(
            model=NLP_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        topic = completion.choices[0].message.content.strip()
        logging.info(f"Detected topic: {topic}")
        return topic
    except Exception as e:
        logging.error(f"Error in topic detection: {str(e)}")
        return "General"

def generate_comment(tweet_text: str, topic: str) -> str:
    logging.info(f"Generating comment for topic '{topic}'")
    prompt = (
        f"Generate 3 professional comment options for this tweet on topic '{topic}'. "
        f"Each must be under 280 characters:\n{tweet_text}"
    )
    try:
        completion = hf_client.chat.completions.create(
            model=NLP_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        comments = completion.choices[0].message.content
        logging.info(f"Generated comments: {comments[:50]}...")
        return comments
    except Exception as e:
        logging.error(f"Error generating comments: {str(e)}")
        return "Error generating comments. Please try again."

def generate_followup(tweet_text: str, engagement_data: dict) -> str:
    desc = "This tweet has "
    if engagement_data:
        parts = [
            f"{engagement_data.get('retweet_count', 0)} retweets",
            f"{engagement_data.get('reply_count', 0)} replies",
            f"{engagement_data.get('like_count', 0)} likes",
            f"{engagement_data.get('impression_count', 0)} impressions",
        ]
        desc += ", ".join(parts)
    else:
        desc += "no engagement yet"

    prompt = (
        f"Based on engagement metrics ({desc}), generate a follow-up reply for the tweet: {tweet_text}. "
        f"Keep it under 280 characters."
    )
    try:
        completion = hf_client.chat.completions.create(
            model=NLP_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating follow-up: {str(e)}")
        return "Error generating follow-up. Please try again."

# ==============================
# Post & Metrics Functions
# ==============================
def post_reply(text: str, tweet_id: str):
    if not api_tracker.can_post():
        st.error("Monthly API posting limit reached (500/month). Consider upgrading to Basic tier.")
        return 429, {"error": "Monthly posting limit reached"}
        
    sess = oauth1_session()
    if not sess:
        return 401, {"error": "OAuth1 session not available"}

    url = "https://api.x.com/2/tweets"
    payload = {"text": text, "reply": {"in_reply_to_tweet_id": tweet_id}}
    
    try:
        resp = sess.post(url, json=payload, timeout=30)
        
        if resp.status_code == 201:
            logging.info("Reply posted successfully")
            api_tracker.log_post()
        else:
            logging.error(f"Post reply failed {resp.status_code}: {resp.text}")
            
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, {"raw": resp.text}
    except Exception as e:
        logging.error(f"Exception posting reply: {str(e)}")
        return 500, {"error": str(e)}

def fetch_metrics(tweet_id: str, use_cache=True):
    """Fetch metrics with caching to reduce API calls."""
    # Check cache first if enabled
    if use_cache and tweet_id in st.session_state["cached_metrics"]:
        logging.info(f"Using cached metrics for tweet {tweet_id}")
        return st.session_state["cached_metrics"][tweet_id]
        
    if not api_tracker.can_read():
        st.error("Monthly API read limit reached (100/month). Consider upgrading to Basic tier.")
        return None
        
    url = f"https://api.x.com/2/tweets/{tweet_id}"
    headers = bearer_headers()
    if headers is None:
        return None

    params = {"tweet.fields": "public_metrics"}
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp = handle_rate_limits(resp)
        api_tracker.log_read()
        
        if resp.status_code != 200:
            logging.error(f"Metrics fetch failed {resp.status_code}: {resp.text}")
            return None
            
        metrics = resp.json().get("data", {}).get("public_metrics", {})
        
        # Update cache
        st.session_state["cached_metrics"][tweet_id] = metrics
        return metrics
    except Exception as e:
        logging.error(f"Exception fetching metrics: {str(e)}")
        return None

# ==============================
# Streamlit UI
# ==============================
st.title("EngageFlow: Social Media Networking Dashboard")
st.caption("Free Tier X API Edition - Limited to 100 reads & 500 posts per month")

# Display API usage
st.sidebar.header("API Usage Tracking")
st.sidebar.metric("Reads Used", f"{api_tracker.read_count}/{api_tracker.free_tier_read_limit}")
st.sidebar.metric("Posts Used", f"{api_tracker.post_count}/{api_tracker.free_tier_post_limit}")
st.sidebar.markdown("---")
st.sidebar.info("⚠️ Free tier is highly limited. Each tweet fetch and interaction counts against your monthly limits.")

username_input = st.text_input("Enter Username", "twitterdev")

# --- 1️⃣ Fetch Tweets ---
fetch_col1, fetch_col2 = st.columns([3, 1])
with fetch_col1:
    fetch_count = st.slider("Number of tweets to fetch", 1, 5, 3, 
                        help="Higher values use more API reads. Free tier is limited to 100 reads per month.")
with fetch_col2:
    if st.button("Fetch Tweets", help="This will use 1-2 API reads from your monthly quota"):
        if api_tracker.read_count >= api_tracker.free_tier_read_limit:
            st.error("Monthly read limit reached. Please upgrade to continue.")
        else:
            with st.spinner("Fetching tweets..."):
                tweet_ids = fetch_latest_tweet_ids(username_input, count=fetch_count)
                if tweet_ids:
                    st.session_state["tweets"] = lookup_tweets_by_ids(tweet_ids)
                    logging.info(f"Fetched {len(st.session_state['tweets'])} tweets")
                else:
                    st.warning("No tweets found or API limit reached.")

tweets = st.session_state.get("tweets", [])

if tweets:
    st.subheader("Fetched Tweets")
    df = pd.DataFrame(
        [
            {"Tweet ID": t["id"], "Text": t["text"][:100] + "...", "Topic": t.get("topic", "Not analyzed")}
            for t in tweets
        ]
    )
    st.dataframe(df)

    # --- 2️⃣ Detect Topics ---
    if st.button("Detect Topics for All", help="Uses HuggingFace ML model, not X API quota"):
        for t in tweets:
            t["topic"] = detect_topic(t["text"])
        st.success("Topics detected")
        logging.info("Topic detection complete")
        df = pd.DataFrame(
            [
                {"Tweet ID": t["id"], "Text": t["text"][:100] + "...", "Topic": t.get("topic", "Not analyzed")}
                for t in tweets
            ]
        )
        st.dataframe(df)

    # --- 3️⃣ Select Tweet ---
    tweet_ids = [t["id"] for t in tweets]
    sel_id = st.selectbox(
        "Select Tweet to work on", tweet_ids,
        format_func=lambda x: next((t["text"][:50] + "..." for t in tweets if t["id"] == x), x),
    )
    st.session_state["selected_tweet_id"] = sel_id
    selected = next((t for t in tweets if t["id"] == sel_id), None)

    if selected:
        st.markdown(f"**Tweet:** {selected['text']}")
        if "topic" in selected:
            st.markdown(f"**Detected Topic:** {selected['topic']}")

        # --- 4️⃣ Generate Comments ---
        if st.button("Generate Comment Suggestions", help="Uses HuggingFace ML model, not X API quota"):
            with st.spinner("Generating..."):
                selected["suggested_comments"] = generate_comment(
                    selected["text"], selected.get("topic", "General")
                )
        if "suggested_comments" in selected:
            st.subheader("Suggested Comments")
            st.markdown(selected["suggested_comments"])

        # --- 5️⃣ Metrics ---
        metrics_col1, metrics_col2 = st.columns([3, 1])
        with metrics_col2:
            use_cached = st.checkbox("Use cached metrics", value=True, 
                                help="Save API reads by using previously fetched metrics")
        with metrics_col1:
            if st.button("Refresh Metrics", help="Uses 1 API read from your monthly quota"):
                if api_tracker.read_count >= api_tracker.free_tier_read_limit:
                    st.error("Monthly read limit reached. Please upgrade to continue.")
                else:
                    with st.spinner("Fetching metrics..."):
                        selected["metrics"] = fetch_metrics(sel_id, use_cache=use_cached)
        
        if "metrics" in selected and selected["metrics"]:
            st.subheader("Tweet Metrics")
            cols = st.columns(len(selected["metrics"]))
            for i, (k, v) in enumerate(selected["metrics"].items()):
                cols[i].metric(k.replace("_", " ").title(), v)

        # --- 6️⃣ Post Reply ---
        st.subheader("Reply to Tweet")
        reply_text = st.text_area("Write Your Reply", "", height=100)
        
        post_col1, post_col2 = st.columns([1, 3])
        with post_col1:
            can_post = api_tracker.read_count < api_tracker.free_tier_post_limit
            if reply_text and len(reply_text) <= 280 and can_post:
                if st.button("Post Reply", help="Uses 1 API post from your monthly quota"):
                    with st.spinner("Posting..."):
                        status, resp = post_reply(reply_text, sel_id)
                        if status == 201:
                            st.success("Reply posted successfully!")
                            st.json(resp)
                        else:
                            st.error(f"Error {status}")
                            st.json(resp)
            elif not can_post:
                st.error("Monthly posting limit reached")
        
        with post_col2:
            char_count = len(reply_text)
            st.caption(f"Character count: {char_count}/280" + 
                      (" ✓" if 0 < char_count <= 280 else " ❌ Too long" if char_count > 280 else ""))

        # --- 7️⃣ Follow-Up ---
        if "metrics" in selected and selected["metrics"]:
            if st.button("Generate Follow-Up", help="Uses HuggingFace ML model, not X API quota"):
                with st.spinner("Generating follow-up..."):
                    selected["followup"] = generate_followup(selected["text"], selected["metrics"])
            if "followup" in selected:
                st.subheader("Suggested Follow-Up")
                st.markdown(selected["followup"])

st.markdown("---")
st.caption("Note: X API Free tier is limited to 100 reads and 500 posts per month. Use this dashboard sparingly or upgrade to Basic tier.")
