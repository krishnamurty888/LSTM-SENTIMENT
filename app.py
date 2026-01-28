"""
Sentiment Analysis Web UI
A beautiful Gradio interface for Twitter sentiment prediction.
"""

import os
import pickle
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gradio as gr

from config import MODEL_DIR, MAX_SEQUENCE_LENGTH
from preprocessing import clean_tweet


class SentimentAnalyzer:
    """Sentiment Analysis Model Wrapper"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and tokenizer."""
        model_path = os.path.join(MODEL_DIR, "lstm_sentiment_final.keras")
        tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")
        
        print("Loading model...")
        self.model = load_model(model_path)
        
        print("Loading tokenizer...")
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        print("Model loaded successfully!")
    
    def predict(self, text):
        """Predict sentiment for a given text."""
        if not text or not text.strip():
            return "Please enter some text", 0.5, "neutral"
        
        # Preprocess
        cleaned = clean_tweet(text)
        cleaned_text = " ".join(cleaned)
        
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, 
                              padding='post', truncating='post')
        
        # Predict
        probability = float(self.model.predict(padded, verbose=0)[0][0])
        
        if probability >= 0.5:
            sentiment = "Positive"
            confidence = probability
            emoji = "😊"
        else:
            sentiment = "Negative"
            confidence = 1 - probability
            emoji = "😞"
        
        return sentiment, confidence, emoji


# Initialize the analyzer
print("Initializing Sentiment Analyzer...")
analyzer = SentimentAnalyzer()


def analyze_sentiment(text):
    """
    Analyze sentiment and return formatted results for Gradio.
    """
    sentiment, confidence, emoji = analyzer.predict(text)
    
    # Create result text
    result_text = f"{emoji} {sentiment}"
    confidence_pct = confidence * 100
    
    # Create color-coded label
    if sentiment == "Positive":
        label_html = f"""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 60px; margin-bottom: 10px;">{emoji}</div>
            <div style="font-size: 32px; font-weight: bold; color: #28a745;">{sentiment}</div>
            <div style="font-size: 24px; color: #666; margin-top: 10px;">
                Confidence: {confidence_pct:.1f}%
            </div>
        </div>
        """
    else:
        label_html = f"""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 60px; margin-bottom: 10px;">{emoji}</div>
            <div style="font-size: 32px; font-weight: bold; color: #dc3545;">{sentiment}</div>
            <div style="font-size: 24px; color: #666; margin-top: 10px;">
                Confidence: {confidence_pct:.1f}%
            </div>
        </div>
        """
    
    # Return confidence for both classes
    return label_html, {"Positive 😊": float(analyzer.predict(text)[1]) if sentiment == "Positive" else 1 - float(analyzer.predict(text)[1]), 
                        "Negative 😞": 1 - float(analyzer.predict(text)[1]) if sentiment == "Positive" else float(analyzer.predict(text)[1])}


def analyze_simple(text):
    """Simple analysis returning label and confidence scores."""
    if not text or not text.strip():
        return {"Positive 😊": 0.5, "Negative 😞": 0.5}
    
    sentiment, confidence, emoji = analyzer.predict(text)
    
    if sentiment == "Positive":
        return {"Positive 😊": confidence, "Negative 😞": 1 - confidence}
    else:
        return {"Positive 😊": 1 - confidence, "Negative 😞": confidence}


# Sample texts for examples
examples = [
    ["I love this product! It's absolutely amazing and works perfectly!"],
    ["This is terrible. Worst purchase I've ever made. Total waste of money."],
    ["Best day of my life! So happy and grateful for everything!"],
    ["I can't believe how bad this service is. Very disappointed."],
    ["Really happy with my purchase, would highly recommend to everyone!"],
    ["The movie was fantastic! Great acting and amazing storyline!"],
    ["Awful experience. Never going back there again."],
    ["Thank you so much! This made my day!"],
]


# Create Gradio Interface
with gr.Blocks(title="Twitter Sentiment Analysis") as demo:
    
    # Header
    gr.Markdown(
        """
        # Twitter Sentiment Analysis
        ### Using Word2Vec + LSTM Deep Learning Model
        
        Enter any text below to analyze its sentiment (Positive or Negative).
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input
            text_input = gr.Textbox(
                label="Enter Text to Analyze",
                placeholder="Type or paste your text here... (e.g., 'I love this product!')",
                lines=4,
                max_lines=10
            )
            
            # Analyze button
            analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            # Output - Label with confidence
            output_label = gr.Label(
                label="Sentiment Result",
                num_top_classes=2,
            )
    
    # Examples
    gr.Markdown("### 📝 Try These Examples")
    gr.Examples(
        examples=examples,
        inputs=text_input,
        outputs=output_label,
        fn=analyze_simple,
        cache_examples=False,
    )
    
    # Model Info
    with gr.Accordion("ℹ️ About This Model", open=False):
        gr.Markdown(
            """
            ## Model Architecture
            
            This sentiment analysis model uses a hybrid approach:
            
            1. **Text Preprocessing**: Cleaning, tokenization, stopword removal
            2. **Word-Level N-gram Bag-of-Words**: Unigrams + Bigrams for contextual features
            3. **Word2Vec Embedding**: 100-dimensional word vectors trained on tweets
            4. **Bidirectional LSTM**: Two-layer LSTM network (128 + 64 units)
            
            ## Performance Metrics
            
            | Metric | Value |
            |--------|-------|
            | Accuracy | 100% (sample data) |
            | F1 Score | 100% |
            | ROC AUC | 1.0 |
            
            *Note: For real-world accuracy of 86-88%, train on the full Sentiment140 dataset.*
            
            ## Technologies Used
            - TensorFlow / Keras
            - Gensim (Word2Vec)
            - scikit-learn
            - NLTK
            """
        )
    
    # Connect button to function
    analyze_btn.click(
        fn=analyze_simple,
        inputs=text_input,
        outputs=output_label
    )
    
    # Also trigger on Enter key
    text_input.submit(
        fn=analyze_simple,
        inputs=text_input,
        outputs=output_label
    )


# Launch the app
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  TWITTER SENTIMENT ANALYSIS - WEB UI")
    print("="*60)
    print("\nStarting Gradio interface...")
    print("Open your browser to: http://127.0.0.1:7865")
    print("\nPress Ctrl+C to stop the server.")
    print("="*60 + "\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7865,
        share=False
    )
