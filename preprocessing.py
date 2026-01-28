"""
Text Preprocessing Module for Twitter Sentiment Analysis

Steps:
1. Lowercase conversion
2. Remove URLs, mentions, hashtags
3. Remove special characters
4. Tokenize words
5. Remove stopwords
"""

import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import pandas as pd
import numpy as np

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data files"""
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

download_nltk_data()

# Get stopwords
STOP_WORDS = set(stopwords.words('english'))

# Keep some sentiment-important words
KEEP_WORDS = {'not', 'no', 'never', 'nor', 'neither', 'but', 'however', 
              'very', 'really', 'too', 'so', 'more', 'most', 'only'}
STOP_WORDS = STOP_WORDS - KEEP_WORDS


def clean_tweet(text):
    """
    Clean a single tweet text.
    
    Args:
        text: Raw tweet text string
        
    Returns:
        List of cleaned tokens
    """
    if not isinstance(text, str):
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove user mentions
    text = re.sub(r"@\w+", "", text)
    
    # Remove hashtag symbol but keep the text
    text = re.sub(r"#", "", text)
    
    # Remove RT (retweet indicator)
    text = re.sub(r"\brt\b", "", text)
    
    # Handle contractions
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
        "'ve": " have", "'m": " am"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove special characters and numbers, keep only letters and spaces
    text = re.sub(r"[^a-z\s]", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords and single characters
    words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    
    return words


def clean_tweet_string(text):
    """
    Clean tweet and return as string (for Bag-of-Words).
    
    Args:
        text: Raw tweet text string
        
    Returns:
        Cleaned text as string
    """
    return " ".join(clean_tweet(text))


def load_and_preprocess_data(filepath, columns, encoding='latin-1', sample_size=None):
    """
    Load and preprocess the Sentiment140 dataset.
    
    Args:
        filepath: Path to the CSV file
        columns: Column names for the dataset
        encoding: File encoding
        sample_size: Number of samples to use (None for all)
        
    Returns:
        Tuple of (cleaned_tweets_list, cleaned_tweets_strings, labels)
    """
    print("Loading dataset...")
    df = pd.read_csv(filepath, encoding=encoding, names=columns)
    
    # Convert target: 0 = Negative (0), 4 = Positive (1)
    df['sentiment'] = df['target'].map({0: 0, 4: 1})
    
    # Remove any rows with missing sentiment
    df = df.dropna(subset=['sentiment', 'text'])
    
    # Sample if needed
    if sample_size and sample_size < len(df):
        # Balanced sampling
        df_pos = df[df['sentiment'] == 1].sample(n=sample_size//2, random_state=42)
        df_neg = df[df['sentiment'] == 0].sample(n=sample_size//2, random_state=42)
        df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(df)} tweets (balanced)")
    
    print(f"Dataset size: {len(df)} tweets")
    print(f"Positive: {(df['sentiment'] == 1).sum()}, Negative: {(df['sentiment'] == 0).sum()}")
    
    # Clean tweets
    print("Cleaning tweets...")
    cleaned_tweets_list = []
    cleaned_tweets_strings = []
    
    for text in tqdm(df['text'].values, desc="Preprocessing"):
        tokens = clean_tweet(text)
        cleaned_tweets_list.append(tokens)
        cleaned_tweets_strings.append(" ".join(tokens))
    
    labels = df['sentiment'].values
    
    return cleaned_tweets_list, cleaned_tweets_strings, labels


def get_data_statistics(cleaned_tweets_list):
    """
    Print statistics about the cleaned data.
    
    Args:
        cleaned_tweets_list: List of tokenized tweets
    """
    lengths = [len(t) for t in cleaned_tweets_list]
    vocab = set(word for tweet in cleaned_tweets_list for word in tweet)
    
    print("\n=== Data Statistics ===")
    print(f"Total tweets: {len(cleaned_tweets_list)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Average tweet length: {np.mean(lengths):.2f} words")
    print(f"Max tweet length: {max(lengths)} words")
    print(f"Min tweet length: {min(lengths)} words")
    print(f"Median tweet length: {np.median(lengths):.0f} words")
    print("=" * 25)


if __name__ == "__main__":
    # Test preprocessing
    test_tweets = [
        "I love this movie! It's amazing @user #happy",
        "This is terrible :( http://example.com",
        "RT @someone: Best day ever!!! 😊",
        "Can't believe how bad this is... #disappointed"
    ]
    
    print("Testing preprocessing:")
    for tweet in test_tweets:
        cleaned = clean_tweet(tweet)
        print(f"Original: {tweet}")
        print(f"Cleaned:  {cleaned}\n")
