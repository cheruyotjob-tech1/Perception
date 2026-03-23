import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from textblob import TextBlob
import re
import time
import os
import tweepy
from datetime import datetime

# ─── Setup ────────────────────────────────────────────────────────────────
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

extra_stopwords = ["the", "it", "in", "wh"]  # lowercase

st.set_page_config(
    page_title="Live X Sentiment Analysis",
    layout="wide",
    page_icon=":bird:"  # emoji fallback; replace with your .png if desired
)

# ─── Credentials from secrets.toml ────────────────────────────────────────
# In .streamlit/secrets.toml or Streamlit Cloud Secrets:
# [twitter]
# bearer_token = "AAAAAAAAAAAAAAAAAAAAA...your_bearer_token_here"
try:
    bearer_token = st.secrets["twitter"]["bearer_token"]
except Exception:
    st.error("Twitter Bearer Token not found in secrets.toml → [connections.twitter.bearer_token] or similar.")
    st.info("Get free Bearer Token at: https://developer.twitter.com → Projects & Apps → Keys & Tokens")
    st.stop()

client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# ─── Class / Logic ────────────────────────────────────────────────────────
class SentimentAnalyzer:

    def __init__(self):
        st.title("Live X (Twitter) Sentiment Analysis")
        st.markdown("Analyze recent tweets from any public X account — get sentiment, subjectivity, word cloud & more.")

    def get_tweets(self, username, max_tweets):
        tweets_list = []
        user_info = None

        try:
            # First, get user ID from username
            user = client.get_user(username=username.strip("@"))
            if not user.data:
                raise ValueError("User not found or protected account.")

            user_id = user.data.id
            user_info = {
                "name": user.data.name,
                "screen_name": user.data.username,
                "description": user.data.description or "No bio",
                "profile_image_url": user.data.profile_image_url
            }

            # Fetch tweets (paginated)
            tweets = []
            pagination_token = None
            remaining = max_tweets

            while remaining > 0:
                response = client.get_users_tweets(
                    id=user_id,
                    max_results=min(100, remaining),  # max per page = 100
                    pagination_token=pagination_token,
                    tweet_fields=["created_at", "text"],
                    exclude=["retweets", "replies"]  # optional: remove if you want replies
                )

                if not response.data:
                    break

                tweets.extend(response.data)
                remaining -= len(response.data)

                pagination_token = response.meta.get("next_token")
                if not pagination_token:
                    break

                time.sleep(1)  # gentle rate limit breathing

            for tweet in tweets:
                tweets_list.append({
                    "date_created": tweet.created_at,
                    "tweet_id": tweet.id,
                    "tweet": tweet.text
                })

        except tweepy.TweepyException as e:
            st.error(f"Twitter API error: {str(e)} (Check token, rate limits, or if account is protected/private.)")
            return [], None, None, None, None
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return [], None, None, None, None

        return tweets_list, user_info["profile_image_url"], user_info["name"], user_info["screen_name"], user_info["description"]

    def clean_tweet(self, tweet):
        tweet = re.sub(r"https?://\S+", "", tweet)           # remove URLs
        tweet = re.sub(r"#[A-Za-z0-9]+", " ", tweet)         # remove hashtags but keep words
        tweet = re.sub(r"#", "", tweet)
        tweet = re.sub(r"\n", " ", tweet)
        tweet = re.sub(r"@[A-Za-z0-9]+", "", tweet)          # remove mentions
        tweet = re.sub(r"\bRT\b", "", tweet)
        tweet = re.sub(r"\w*\d\w*", "", tweet)               # remove words with digits
        tweet = re.sub(r"[^A-Za-z\s]", "", tweet)            # keep only letters & space
        tweet = tweet.lower().strip()

        # Remove extra stopwords & short words
        words = [w for w in tweet.split() if len(w) > 2 and w not in extra_stopwords]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words if w not in stopwords.words("english")]

        return " ".join(words)

    def generate_wordcloud(self, clean_tweets):
        if not clean_tweets:
            return None

        text = " ".join(clean_tweets)
        if len(text.strip()) < 10:
            st.warning("Not enough clean text for word cloud.")
            return None

        wordcloud = WordCloud(
            width=800, height=400,
            background_color="black",
            min_font_size=10,
            random_state=42
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        return fig

    def get_polarity(self, text):
        return TextBlob(text).sentiment.polarity

    def get_analysis(self, polarity):
        if polarity < 0:
            return "Negative"
        elif polarity == 0:
            return "Neutral"
        else:
            return "Positive"

    def get_subjectivity(self, text):
        return TextBlob(text).sentiment.subjectivity

    def get_sub_analysis(self, subjectivity):
        return "Objective" if subjectivity <= 0.5 else "Subjective"

    def plot_sentiments(self, df):
        if df.empty:
            return None
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        fig = go.Figure(go.Bar(
            x=counts["sentiment"],
            y=counts["count"],
            marker_color=["red", "grey", "green"]
        ))
        fig.update_layout(
            title="Sentiment Distribution",
            xaxis_title="Sentiment",
            yaxis_title="Count",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        return fig

    def plot_subjectivity(self, df):
        if df.empty:
            return None
        counts = df["sub_obj"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.4,
            marker_colors=["#636EFA", "#00CC96"]
        )])
        fig.update_layout(title="Subjective vs Objective Tweets")
        return fig

    def run(self):
        st.sidebar.header("Settings")

        username = st.sidebar.text_input("X Username (without @)", placeholder="example: elonmusk")
        tweet_count = st.sidebar.slider("Number of recent tweets", 5, 200, 50, step=5)

        st.sidebar.markdown("---")
        st.sidebar.info("Note: Free Twitter API limits → max ~3,200 recent tweets per user. Protected accounts inaccessible.")

        if username and tweet_count > 0:
            with st.spinner("Fetching tweets and analyzing..."):
                tweets_list, img_url, name, screen_name, desc = self.get_tweets(username, tweet_count)

            if not tweets_list:
                st.stop()

            df = pd.DataFrame(tweets_list)

            # Sidebar user info
            st.sidebar.success("Account Info:")
            st.sidebar.markdown(f"**Name:** {name}")
            st.sidebar.markdown(f"**@Handle:** @{screen_name}")
            st.sidebar.markdown(f"**Bio:** {desc}")
            if img_url:
                st.sidebar.image(img_url, width=100)

            # Process
            df["clean_tweet"] = df["tweet"].apply(self.clean_tweet)
            df["polarity"] = df["clean_tweet"].apply(self.get_polarity)
            df["sentiment"] = df["polarity"].apply(self.get_analysis)
            df["subjectivity"] = df["clean_tweet"].apply(self.get_subjectivity)
            df["sub_obj"] = df["subjectivity"].apply(self.get_sub_analysis)

            # ─── Visuals ──────────────────────────────────────────────────────
            st.subheader(f"Analysis for @{screen_name} — last {len(df)} tweets")

            col1, col2 = st.columns([3, 2])

            with col1:
                fig_sent = self.plot_sentiments(df)
                if fig_sent:
                    st.plotly_chart(fig_sent, use_container_width=True)

            with col2:
                fig_sub = self.plot_subjectivity(df)
                if fig_sub:
                    st.plotly_chart(fig_sub, use_container_width=True)

            # Word Cloud
            fig_wc = self.generate_wordcloud(df["clean_tweet"])
            if fig_wc:
                st.subheader("Word Cloud")
                st.pyplot(fig_wc)

            # Show tweets
            st.subheader("Recent Tweets with Sentiment")
            for _, row in df.head(15).iterrows():  # limit display
                color = "green" if row["sentiment"] == "Positive" else "red" if row["sentiment"] == "Negative" else "grey"
                date_str = row["date_created"].strftime("%b %d, %Y  %H:%M")
                st.markdown(f"**{row['sentiment']}** ({date_str})")
                st.write(row["tweet"])
                st.caption(f"Polarity: {row['polarity']:.3f} | Subjectivity: {row['subjectivity']:.3f}")
                st.markdown("---")

        else:
            st.info("Enter a valid X username and tweet count in the sidebar → press Enter.")


if __name__ == "__main__":
    app = SentimentAnalyzer()
    app.run()
