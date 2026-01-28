"""
Prediction Script for Sentiment Analysis

Load trained model and make predictions on new text.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import MODEL_DIR, MAX_SEQUENCE_LENGTH
from preprocessing import clean_tweet


class SentimentPredictor:
    """
    Sentiment prediction class for making predictions on new text.
    """
    
    def __init__(self, model_path=None, tokenizer_path=None):
        """
        Initialize the predictor with trained model and tokenizer.
        
        Args:
            model_path: Path to saved Keras model
            tokenizer_path: Path to saved tokenizer
        """
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "lstm_sentiment_final.keras")
        if tokenizer_path is None:
            tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        
        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}...")
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        print("✓ Model and tokenizer loaded successfully!")
    
    def preprocess(self, text):
        """
        Preprocess text for prediction.
        
        Args:
            text: Input text string
            
        Returns:
            Padded sequence ready for prediction
        """
        # Clean text
        cleaned = clean_tweet(text)
        cleaned_text = " ".join(cleaned)
        
        # Tokenize
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        
        # Pad
        padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, 
                              padding='post', truncating='post')
        
        return padded
    
    def predict(self, text):
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with sentiment and confidence
        """
        padded = self.preprocess(text)
        probability = self.model.predict(padded, verbose=0)[0][0]
        
        sentiment = "Positive" if probability >= 0.5 else "Negative"
        confidence = probability if probability >= 0.5 else 1 - probability
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': float(confidence),
            'probability': float(probability)
        }
    
    def predict_batch(self, texts):
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of prediction dictionaries
        """
        # Preprocess all texts
        padded_sequences = []
        for text in texts:
            padded = self.preprocess(text)
            padded_sequences.append(padded[0])
        
        padded_sequences = np.array(padded_sequences)
        
        # Predict
        probabilities = self.model.predict(padded_sequences, verbose=0).flatten()
        
        # Format results
        results = []
        for text, prob in zip(texts, probabilities):
            sentiment = "Positive" if prob >= 0.5 else "Negative"
            confidence = prob if prob >= 0.5 else 1 - prob
            
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': float(confidence),
                'probability': float(prob)
            })
        
        return results


def interactive_mode(predictor):
    """
    Run interactive prediction mode.
    
    Args:
        predictor: SentimentPredictor instance
    """
    print("\n" + "="*60)
    print("  INTERACTIVE SENTIMENT PREDICTION")
    print("="*60)
    print("Enter text to analyze sentiment. Type 'quit' to exit.\n")
    
    while True:
        try:
            text = input("📝 Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not text:
                continue
            
            result = predictor.predict(text)
            
            emoji = "[+]" if result['sentiment'] == "Positive" else "[-]"
            print(f"\n{emoji} Sentiment: {result['sentiment']}")
            print(f"   Confidence: {result['confidence']*100:.1f}%")
            print(f"   Raw probability: {result['probability']:.4f}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


def demo_predictions(predictor):
    """
    Run demo predictions on sample texts.
    
    Args:
        predictor: SentimentPredictor instance
    """
    sample_texts = [
        "I love this product! It's absolutely amazing!",
        "This is the worst experience ever. Never buying again.",
        "The movie was okay, nothing special but not bad either.",
        "Best day of my life! Everything is perfect!",
        "I can't believe how terrible this service is.",
        "Really happy with my purchase, highly recommend!",
        "Not sure how I feel about this, mixed emotions.",
        "Completely disappointed and frustrated with the quality.",
        "This made my day! So happy and grateful!",
        "What a waste of money, total garbage."
    ]
    
    print("\n" + "="*60)
    print("  DEMO PREDICTIONS")
    print("="*60 + "\n")
    
    results = predictor.predict_batch(sample_texts)
    
    for result in results:
        emoji = "[+]" if result['sentiment'] == "Positive" else "[-]"
        conf = result['confidence'] * 100
        print(f"{emoji} [{result['sentiment']:8s}] ({conf:5.1f}%) | {result['text'][:50]}...")
    
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sentiment Prediction')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--demo', action='store_true', help='Run demo predictions')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = SentimentPredictor()
    
    if args.text:
        # Single prediction
        result = predictor.predict(args.text)
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
    
    elif args.demo:
        # Demo mode
        demo_predictions(predictor)
    
    else:
        # Interactive mode (default)
        demo_predictions(predictor)
        interactive_mode(predictor)
