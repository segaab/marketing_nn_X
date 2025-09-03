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
# Session State
# ==============================
def _seed_session_from_env():
    for k, v in ENV_MAP.items():
        st.session_state.setdefault(k.lower(), v)
    st.session_state.setdefault("tweets", [])
    st.session_state.setdefault("selected_tweet_id", None)

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
# API Functions
# ==============================
def fetch_latest_tweet_ids(username: str, count=5):
    """
    Fetch the latest tweet IDs for a username.
    """
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = bearer_headers()
    if headers is None:
        return []

    params = {"query": f"from:{username}", "max_results": count, "tweet.fields": "id"}
    resp = requests.get(url, headers=headers, params=params, timeout=30)

    logging.info(
        f"Rate limits (ID fetch) → Remaining: {resp.headers.get('x-rate-limit-remaining')} "
        f"Reset: {resp.headers.get('x-rate-limit-reset')}"
    )

    if resp.status_code != 200:
        logging.error(f"Fetch IDs error {resp.status_code}: {resp.text}")
        st.error(f"Error fetching tweet IDs: {resp.status_code}")
        return []

    data = resp.json().get("data", [])
    return [t["id"] for t in data]

def lookup_tweets_by_ids(tweet_ids):
    """
    Lookup tweets with details using Tweet Lookup endpoint.
    """
    if not tweet_ids:
        return []

    url = "https://api.twitter.com/2/tweets"
    headers = bearer_headers()
    ids_param = ",".join(tweet_ids)

    params = {
        "ids": ids_param,
        "tweet.fields": "author_id,created_at,public_metrics,text",
        "expansions": "author_id",
        "user.fields": "created_at,username,name",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)

    logging.info(
        f"Rate limits (lookup) → Remaining: {resp.headers.get('x-rate-limit-remaining')} "
        f"Reset: {resp.headers.get('x-rate-limit-reset')}"
    )

    if resp.status_code != 200:
        logging.error(f"Lookup error {resp.status_code}: {resp.text}")
        st.error(f"Error fetching tweets: {resp.status_code}")
        return []

    return resp.json().get("data", [])

# ==============================
# Streamlit UI
# ==============================
st.title("EngageFlow: Social Media Networking Dashboard")

username_input = st.text_input("Enter Username", "twitterdev")

# --- 1️⃣ Fetch Tweets ---
if st.button("Fetch Tweets"):
    with st.spinner("Fetching tweets..."):
        tweet_ids = fetch_latest_tweet_ids(username_input, count=5)
        st.session_state["tweets"] = lookup_tweets_by_ids(tweet_ids)
        logging.info(f"Fetched {len(st.session_state['tweets'])} tweets")

tweets = st.session_state.get("tweets", [])

if tweets:
    st.subheader("Fetched Tweets")
    df = pd.DataFrame(
        [
            {"Tweet ID": t["id"], "Text": t["text"], "Topic": t.get("topic", "Not analyzed")}
            for t in tweets
        ]
    )
    st.dataframe(df)

    # --- 2️⃣ Detect Topics ---
    if st.button("Detect Topics for All"):
        for t in tweets:
            t["topic"] = detect_topic(t["text"])
        st.success("Topics detected")
        logging.info("Topic detection complete")
        df = pd.DataFrame(
            [
                {"Tweet ID": t["id"], "Text": t["text"], "Topic": t.get("topic", "Not analyzed")}
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
        if st.button("Generate Comment Suggestions"):
            with st.spinner("Generating..."):
                selected["suggested_comments"] = generate_comment(
                    selected["text"], selected.get("topic", "General")
                )
        if "suggested_comments" in selected:
            st.subheader("Suggested Comments")
            st.markdown(selected["suggested_comments"])

        # --- 5️⃣ Metrics ---
        if st.button("Refresh Metrics"):
            with st.spinner("Fetching metrics..."):
                selected["metrics"] = fetch_metrics(sel_id)
        if "metrics" in selected and selected["metrics"]:
            st.subheader("Tweet Metrics")
            cols = st.columns(len(selected["metrics"]))
            for i, (k, v) in enumerate(selected["metrics"].items()):
                cols[i].metric(k.replace("_", " ").title(), v)

        # --- 6️⃣ Post Reply ---
        reply_text = st.text_area("Write Your Reply", "", height=100)
        if reply_text and len(reply_text) <= 280:
            if st.button("Post Reply"):
                with st.spinner("Posting..."):
                    status, resp = post_reply(reply_text, sel_id)
                    if status == 201:
                        st.success("Reply posted successfully!")
                        st.json(resp)
                    else:
                        st.error(f"Error {status}")
                        st.json(resp)

        # --- 7️⃣ Follow-Up ---
        if "metrics" in selected and selected["metrics"]:
            if st.button("Generate Follow-Up"):
                with st.spinner("Generating follow-up..."):
                    selected["followup"] = generate_followup(selected["text"], selected["metrics"])
            if "followup" in selected:
                st.subheader("Suggested Follow-Up")
                st.markdown(selected["followup"])
