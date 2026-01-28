"""
Configuration settings for Sentiment Analysis Project
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Dataset settings
DATASET_PATH = os.path.join(DATA_DIR, "training.1600000.processed.noemoticon.csv")
DATASET_COLUMNS = ['target', 'id', 'date', 'flag', 'user', 'text']
DATASET_ENCODING = 'latin-1'

# Preprocessing settings
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000

# Word2Vec settings
W2V_VECTOR_SIZE = 100  # Can increase to 200 for better accuracy
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_WORKERS = 4

# N-gram Bag-of-Words settings
NGRAM_RANGE = (1, 2)  # Unigrams + Bigrams
BOW_MAX_FEATURES = 20000

# LSTM Model settings
EMBEDDING_DIM = 100  # Should match W2V_VECTOR_SIZE
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DROPOUT_RATE = 0.3
USE_BIDIRECTIONAL = True  # Set True for better accuracy

# Training settings
BATCH_SIZE = 128
EPOCHS = 5
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# Sample size (set to None to use full dataset)
# Use smaller value for testing: e.g., 100000
SAMPLE_SIZE = 20000  # Using sample for faster demo

# Random seed for reproducibility
RANDOM_SEED = 42
