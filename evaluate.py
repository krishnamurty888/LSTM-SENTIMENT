"""
Evaluation Module for Sentiment Analysis Model

Provides comprehensive evaluation metrics and visualizations.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import os

from config import MODEL_DIR


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate model and return comprehensive metrics.
    
    Args:
        model: Trained Keras model
        X_test: Test sequences
        y_test: True labels
        threshold: Classification threshold
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    return metrics


def print_evaluation_report(metrics, y_test):
    """
    Print detailed evaluation report.
    
    Args:
        metrics: Dictionary from evaluate_model
        y_test: True labels
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION REPORT")
    print("="*60)
    
    print(f"\n=== Overall Metrics ===")
    print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   Precision: {metrics['precision']*100:.2f}%")
    print(f"   Recall:    {metrics['recall']*100:.2f}%")
    print(f"   F1 Score:  {metrics['f1_score']*100:.2f}%")
    
    print(f"\n=== Classification Report ===")
    print(classification_report(
        y_test, 
        metrics['predictions'],
        target_names=['Negative', 'Positive']
    ))
    
    print("="*60 + "\n")


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_roc_curve(y_test, y_pred_proba, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.close()
    
    return roc_auc


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Keras training history object
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def plot_prediction_distribution(y_pred_proba, y_test, save_path=None):
    """
    Plot distribution of prediction probabilities.
    
    Args:
        y_pred_proba: Predicted probabilities
        y_test: True labels
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    # Separate by true class
    proba_neg = y_pred_proba[y_test == 0]
    proba_pos = y_pred_proba[y_test == 1]
    
    plt.hist(proba_neg, bins=50, alpha=0.6, label='Negative', color='red')
    plt.hist(proba_pos, bins=50, alpha=0.6, label='Positive', color='green')
    
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Probabilities', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
    
    plt.close()


def predict_sentiment(model, text, tokenizer, preprocessor, max_length=100):
    """
    Predict sentiment for a single text.
    
    Args:
        model: Trained model
        text: Input text string
        tokenizer: Fitted Keras Tokenizer
        preprocessor: Preprocessing function
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (sentiment_label, probability)
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Preprocess
    cleaned = preprocessor(text)
    cleaned_text = " ".join(cleaned) if isinstance(cleaned, list) else cleaned
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=max_length)
    
    # Predict
    probability = model.predict(padded, verbose=0)[0][0]
    sentiment = "Positive" if probability >= 0.5 else "Negative"
    
    return sentiment, probability


def batch_predict(model, texts, tokenizer, preprocessor, max_length=100):
    """
    Predict sentiment for multiple texts.
    
    Args:
        model: Trained model
        texts: List of input texts
        tokenizer: Fitted Keras Tokenizer
        preprocessor: Preprocessing function
        max_length: Maximum sequence length
        
    Returns:
        List of (text, sentiment, probability) tuples
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    results = []
    
    # Preprocess all texts
    cleaned_texts = []
    for text in texts:
        cleaned = preprocessor(text)
        cleaned_text = " ".join(cleaned) if isinstance(cleaned, list) else cleaned
        cleaned_texts.append(cleaned_text)
    
    # Tokenize and pad
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(sequences, maxlen=max_length)
    
    # Predict
    probabilities = model.predict(padded, verbose=0).flatten()
    
    for text, prob in zip(texts, probabilities):
        sentiment = "Positive" if prob >= 0.5 else "Negative"
        results.append((text, sentiment, prob))
    
    return results


if __name__ == "__main__":
    # Test plotting functions with dummy data
    import numpy as np
    
    # Dummy data
    y_test = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.6, 0.3, 0.4, 0.8, 0.9, 0.4, 0.7, 0.85])
    
    # Test confusion matrix
    print("Testing confusion matrix plot...")
    plot_confusion_matrix(y_test, y_pred)
    
    # Test ROC curve
    print("\nTesting ROC curve plot...")
    plot_roc_curve(y_test, y_pred_proba)
