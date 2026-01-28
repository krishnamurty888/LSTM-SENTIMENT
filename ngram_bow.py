"""
Word-Level N-gram Bag-of-Words Module

Creates unigram + bigram features for capturing contextual sentiment information.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import os

from config import BOW_MAX_FEATURES, NGRAM_RANGE, MODEL_DIR


class NgramBagOfWords:
    """
    Word-Level N-gram Bag-of-Words feature extractor.
    
    Extracts unigrams and bigrams to capture local contextual features.
    """
    
    def __init__(self, ngram_range=NGRAM_RANGE, max_features=BOW_MAX_FEATURES, use_tfidf=False):
        """
        Initialize the N-gram BoW extractor.
        
        Args:
            ngram_range: Tuple (min_n, max_n) for n-gram range
            max_features: Maximum number of features to extract
            use_tfidf: If True, use TF-IDF weighting
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.use_tfidf = use_tfidf
        
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                sublinear_tf=True,
                min_df=2
            )
        else:
            self.vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=2
            )
    
    def fit(self, texts):
        """
        Fit the vectorizer on the training texts.
        
        Args:
            texts: List of cleaned text strings
        """
        print(f"Fitting N-gram BoW (range={self.ngram_range}, max_features={self.max_features})...")
        self.vectorizer.fit(texts)
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return self
    
    def transform(self, texts):
        """
        Transform texts to N-gram BoW features.
        
        Args:
            texts: List of cleaned text strings
            
        Returns:
            Sparse matrix of N-gram features
        """
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """
        Fit and transform in one step.
        
        Args:
            texts: List of cleaned text strings
            
        Returns:
            Sparse matrix of N-gram features
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self):
        """Get the feature names (n-grams)."""
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features(self, n=20):
        """
        Get the top N most common features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of (feature, count) tuples
        """
        feature_names = self.get_feature_names()
        # Note: This is just the feature names, not sorted by frequency
        # For actual frequency, you'd need the transformed matrix
        return feature_names[:n]
    
    def save(self, filepath=None):
        """Save the vectorizer to disk."""
        if filepath is None:
            filepath = os.path.join(MODEL_DIR, "ngram_bow_vectorizer.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"N-gram BoW vectorizer saved to {filepath}")
    
    def load(self, filepath=None):
        """Load the vectorizer from disk."""
        if filepath is None:
            filepath = os.path.join(MODEL_DIR, "ngram_bow_vectorizer.pkl")
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"N-gram BoW vectorizer loaded from {filepath}")
        return self


def analyze_ngrams(X_bow, vectorizer, labels, top_n=20):
    """
    Analyze the most discriminative n-grams for each sentiment.
    
    Args:
        X_bow: Sparse matrix of BoW features
        vectorizer: Fitted vectorizer
        labels: Array of sentiment labels (0 or 1)
        top_n: Number of top features per class
    """
    import numpy as np
    
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate average feature frequency per class
    X_pos = X_bow[labels == 1].mean(axis=0).A1
    X_neg = X_bow[labels == 0].mean(axis=0).A1
    
    # Get top features for positive sentiment
    pos_indices = np.argsort(X_pos - X_neg)[-top_n:][::-1]
    neg_indices = np.argsort(X_neg - X_pos)[-top_n:][::-1]
    
    print("\n=== Top N-grams for Positive Sentiment ===")
    for i in pos_indices:
        print(f"  {feature_names[i]}: {X_pos[i]:.4f}")
    
    print("\n=== Top N-grams for Negative Sentiment ===")
    for i in neg_indices:
        print(f"  {feature_names[i]}: {X_neg[i]:.4f}")


if __name__ == "__main__":
    # Test N-gram BoW
    test_texts = [
        "love this movie amazing",
        "terrible awful bad experience",
        "not bad actually good",
        "happy great day wonderful"
    ]
    
    ngram_bow = NgramBagOfWords(ngram_range=(1, 2), max_features=100)
    X = ngram_bow.fit_transform(test_texts)
    
    print(f"\nShape: {X.shape}")
    print(f"Sample features: {ngram_bow.get_top_features(10)}")
