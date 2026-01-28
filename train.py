"""
Main Training Script for Sentiment Analysis

This script orchestrates the complete training pipeline:
1. Load and preprocess data
2. Create N-gram Bag-of-Words features
3. Train Word2Vec embeddings
4. Build and train LSTM model
5. Evaluate and save results
"""

import os
import sys
import numpy as np
import pickle
from datetime import datetime

# TensorFlow configuration (must be before importing TensorFlow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Local imports
from config import (
    DATASET_PATH, DATASET_COLUMNS, DATASET_ENCODING,
    MAX_SEQUENCE_LENGTH, MAX_VOCAB_SIZE, SAMPLE_SIZE,
    BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, RANDOM_SEED,
    MODEL_DIR, USE_BIDIRECTIONAL
)
from preprocessing import (
    load_and_preprocess_data, get_data_statistics, clean_tweet
)
from ngram_bow import NgramBagOfWords, analyze_ngrams
from word2vec_model import Word2VecEmbedding
from lstm_model import (
    create_lstm_model, create_advanced_lstm_model,
    get_callbacks, print_model_summary
)
from evaluate import (
    evaluate_model, print_evaluation_report,
    plot_confusion_matrix, plot_roc_curve,
    plot_training_history, plot_prediction_distribution
)


def check_gpu():
    """Check if GPU is available."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[OK] GPU available: {gpus}")
        # Enable memory growth to avoid OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[INFO] No GPU detected. Training will use CPU.")
    return len(gpus) > 0


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("  TWITTER SENTIMENT ANALYSIS - WORD2VEC + LSTM")
    print("="*70)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # ========================================
    # Step 1: Load and Preprocess Data
    # ========================================
    print("\n[STEP 1] Loading and Preprocessing Data")
    print("-" * 50)
    
    if not os.path.exists(DATASET_PATH):
        print(f"\n[ERROR] Dataset not found at {DATASET_PATH}")
        print("\n[INFO] Please download the Sentiment140 dataset:")
        print("   https://www.kaggle.com/datasets/kazanova/sentiment140")
        print(f"\n   Place 'training.1600000.processed.noemoticon.csv' in:")
        print(f"   {os.path.dirname(DATASET_PATH)}")
        sys.exit(1)
    
    cleaned_tweets_list, cleaned_tweets_strings, labels = load_and_preprocess_data(
        filepath=DATASET_PATH,
        columns=DATASET_COLUMNS,
        encoding=DATASET_ENCODING,
        sample_size=SAMPLE_SIZE
    )
    
    get_data_statistics(cleaned_tweets_list)
    
    # ========================================
    # Step 2: N-gram Bag-of-Words Analysis
    # ========================================
    print("\n[STEP 2] Creating N-gram Bag-of-Words Features")
    print("-" * 50)
    
    ngram_bow = NgramBagOfWords(ngram_range=(1, 2), max_features=MAX_VOCAB_SIZE)
    X_bow = ngram_bow.fit_transform(cleaned_tweets_strings)
    
    print(f"BoW Feature matrix shape: {X_bow.shape}")
    
    # Analyze top n-grams
    analyze_ngrams(X_bow, ngram_bow.vectorizer, labels, top_n=15)
    
    # Save vectorizer
    ngram_bow.save()
    
    # ========================================
    # Step 3: Train Word2Vec
    # ========================================
    print("\n[STEP 3] Training Word2Vec Embeddings")
    print("-" * 50)
    
    w2v = Word2VecEmbedding()
    w2v.train(cleaned_tweets_list, epochs=10)
    
    # Test Word2Vec
    test_words = ['good', 'bad', 'love', 'hate', 'happy', 'sad']
    print("\nWord2Vec similarity examples:")
    for word in test_words:
        similar = w2v.get_similar_words(word, topn=3)
        if similar:
            similar_str = ", ".join([f"{w}({s:.2f})" for w, s in similar])
            print(f"  '{word}' -> {similar_str}")
    
    # Save Word2Vec model
    w2v.save()
    
    # ========================================
    # Step 4: Tokenization and Padding
    # ========================================
    print("\n[STEP 4] Tokenizing and Padding Sequences")
    print("-" * 50)
    
    # Create tokenizer
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(cleaned_tweets_strings)
    
    word_index = tokenizer.word_index
    print(f"Total unique tokens: {len(word_index)}")
    print(f"Max vocabulary size: {MAX_VOCAB_SIZE}")
    
    # Convert to sequences
    sequences = tokenizer.texts_to_sequences(cleaned_tweets_strings)
    X_seq = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    print(f"Padded sequences shape: {X_seq.shape}")
    
    # Save tokenizer
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # ========================================
    # Step 5: Create Embedding Matrix
    # ========================================
    print("\n[STEP 5] Creating Embedding Matrix")
    print("-" * 50)
    
    embedding_matrix = w2v.create_embedding_matrix(word_index)
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    
    # Save embedding matrix
    emb_path = os.path.join(MODEL_DIR, "embedding_matrix.npy")
    np.save(emb_path, embedding_matrix)
    print(f"Embedding matrix saved to {emb_path}")
    
    # ========================================
    # Step 6: Split Data
    # ========================================
    print("\n[STEP 6] Splitting Data into Train/Test Sets")
    print("-" * 50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Training positive ratio: {np.mean(y_train):.2%}")
    print(f"Test positive ratio: {np.mean(y_test):.2%}")
    
    # ========================================
    # Step 7: Build and Train LSTM Model
    # ========================================
    print("\n[STEP 7] Building and Training LSTM Model")
    print("-" * 50)
    
    vocab_size = min(MAX_VOCAB_SIZE, len(word_index)) + 1
    
    model = create_lstm_model(
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix[:vocab_size],
        use_bidirectional=USE_BIDIRECTIONAL
    )
    
    print_model_summary(model)
    
    # Get callbacks
    callbacks = get_callbacks()
    
    # Train model
    print("\n>>> Starting training...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Validation split: {VALIDATION_SPLIT}")
    print()
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )
    
    # ========================================
    # Step 8: Evaluate Model
    # ========================================
    print("\n[STEP 8] Evaluating Model")
    print("-" * 50)
    
    # Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test)
    print_evaluation_report(metrics, y_test)
    
    # ========================================
    # Step 9: Visualizations
    # ========================================
    print("\n[STEP 9] Generating Visualizations")
    print("-" * 50)
    
    # Create plots directory
    plots_dir = os.path.join(MODEL_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot training history
    plot_training_history(
        history, 
        save_path=os.path.join(plots_dir, "training_history.png")
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_test, 
        metrics['predictions'],
        save_path=os.path.join(plots_dir, "confusion_matrix.png")
    )
    
    # Plot ROC curve
    roc_auc = plot_roc_curve(
        y_test, 
        metrics['probabilities'],
        save_path=os.path.join(plots_dir, "roc_curve.png")
    )
    
    # Plot prediction distribution
    plot_prediction_distribution(
        metrics['probabilities'],
        y_test,
        save_path=os.path.join(plots_dir, "prediction_distribution.png")
    )
    
    # ========================================
    # Step 10: Save Final Model
    # ========================================
    print("\n[STEP 10] Saving Final Model")
    print("-" * 50)
    
    model_path = os.path.join(MODEL_DIR, "lstm_sentiment_final.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # ========================================
    # Final Summary
    # ========================================
    print("\n" + "="*70)
    print("  TRAINING COMPLETE!")
    print("="*70)
    print(f"\n  Final Results:")
    print(f"     Test Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"     Test F1 Score:  {metrics['f1_score']*100:.2f}%")
    print(f"     ROC AUC:        {roc_auc:.4f}")
    print(f"\n  Saved Files:")
    print(f"     Model:          {model_path}")
    print(f"     Tokenizer:      {tokenizer_path}")
    print(f"     Word2Vec:       {os.path.join(MODEL_DIR, 'word2vec.model')}")
    print(f"     Plots:          {plots_dir}")
    print(f"\n  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    return model, tokenizer, metrics


if __name__ == "__main__":
    main()
