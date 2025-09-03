# Chunk 1: Imports, Logging, Environment, Session State, Auth Helpers, API Functions, Post & Metrics Functions

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
def fetch_tweets(query: str):
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = bearer_headers()
    if headers is None:
        return []

    params = {
        "query": query,
        "max_results": 15,  # fixed to 15
        "tweet.fields": "author_id,created_at,public_metrics,text",
    }

    logging.info(f"Fetching tweets for query='{query}'")
    logging.info(f"Request: GET {url} with params={json.dumps(params)} headers={headers}")
    resp = requests.get(url, headers=headers, params=params, timeout=30)

    logging.info(f"Response: status={resp.status_code}, reason={resp.reason}")
    logging.info(f"Rate limits → Remaining: {resp.headers.get('x-rate-limit-remaining')} Reset: {resp.headers.get('x-rate-limit-reset')}")

    if resp.status_code == 200:
        logging.info(f"Response body summary: {len(resp.json().get('data', []))} tweets fetched")
    else:
        logging.error(f"Response body: {resp.text}")

    if resp.status_code != 200:
        st.error(f"Error fetching tweets: {resp.status_code}")
        return []

    data = resp.json().get("data", [])
    logging.info(f"Fetched {len(data)} tweets")
    return data

# ==============================
# Post & Metrics Functions
# ==============================
def post_reply(text: str, tweet_id: str):
    sess = oauth1_session()
    if not sess:
        return 401, {"error": "OAuth1 session not available"}

    url = "https://api.twitter.com/2/tweets"
    payload = {"text": text, "reply": {"in_reply_to_tweet_id": tweet_id}}

    logging.info(f"Posting reply to {url} with payload={json.dumps(payload)}")
    resp = sess.post(url, json=payload, timeout=30)

    logging.info(f"Response: status={resp.status_code}, reason={resp.reason}")
    if resp.status_code != 201:
        logging.error(f"Post reply failed: {resp.text}")
    else:
        logging.info(f"Response body: {resp.text}")
        logging.info("Reply posted successfully")

    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, {"raw": resp.text}

def fetch_metrics(tweet_id: str):
    url = f"https://api.twitter.com/2/tweets/{tweet_id}"
    headers = bearer_headers()
    if headers is None:
        return None

    params = {"tweet.fields": "public_metrics"}

    logging.info(f"Fetching metrics for tweet_id={tweet_id}")
    logging.info(f"Request: GET {url} with params={json.dumps(params)} headers={headers}")
    resp = requests.get(url, headers=headers, params=params, timeout=30)

    logging.info(f"Response: status={resp.status_code}, reason={resp.reason}")
    logging.info(f"Rate limits (metrics) → Remaining: {resp.headers.get('x-rate-limit-remaining')} Reset: {resp.headers.get('x-rate-limit-reset')}")

    if resp.status_code == 200:
        logging.info(f"Response body: {json.dumps(resp.json(), indent=2)}")
    else:
        logging.error(f"Response body: {resp.text}")

    if resp.status_code != 200:
        logging.error(f"Metrics fetch failed {resp.status_code}: {resp.text}")
        return None
    return resp.json().get("data", {}).get("public_metrics", {})
if __name__ == "__main__":
         # Any startup logic here, e.g., logging a message
         logging.info("Starting EngageFlow dashboard...")
         # The rest of your Streamlit UI code is already top-level, so it will run automatically
