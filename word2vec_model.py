"""
Word2Vec Embedding Module

Trains Word2Vec on tweet data and creates embedding matrix for LSTM.
"""

import numpy as np
from gensim.models import Word2Vec
import os
from tqdm import tqdm

from config import (
    W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT, W2V_WORKERS,
    MODEL_DIR, EMBEDDING_DIM
)


class Word2VecEmbedding:
    """
    Word2Vec embedding trainer and embedding matrix creator.
    """
    
    def __init__(self, vector_size=W2V_VECTOR_SIZE, window=W2V_WINDOW, 
                 min_count=W2V_MIN_COUNT, workers=W2V_WORKERS):
        """
        Initialize Word2Vec parameters.
        
        Args:
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Minimum word frequency
            workers: Number of worker threads
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
    
    def train(self, sentences, epochs=10):
        """
        Train Word2Vec model on tokenized sentences.
        
        Args:
            sentences: List of tokenized sentences (list of word lists)
            epochs: Number of training epochs
            
        Returns:
            Trained Word2Vec model
        """
        print(f"Training Word2Vec (dim={self.vector_size}, window={self.window})...")
        
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=epochs,
            seed=42
        )
        
        print(f"Word2Vec vocabulary size: {len(self.model.wv)}")
        return self.model
    
    def create_embedding_matrix(self, word_index):
        """
        Create embedding matrix for Keras Embedding layer.
        
        Args:
            word_index: Dictionary mapping words to indices (from Tokenizer)
            
        Returns:
            Numpy array of shape (vocab_size + 1, embedding_dim)
        """
        if self.model is None:
            raise ValueError("Word2Vec model not trained. Call train() first.")
        
        vocab_size = len(word_index) + 1
        embedding_matrix = np.zeros((vocab_size, self.vector_size))
        
        words_found = 0
        words_not_found = 0
        
        print("Creating embedding matrix...")
        for word, i in tqdm(word_index.items(), desc="Building embeddings"):
            if word in self.model.wv:
                embedding_matrix[i] = self.model.wv[word]
                words_found += 1
            else:
                # Initialize with random small values for unknown words
                embedding_matrix[i] = np.random.uniform(-0.05, 0.05, self.vector_size)
                words_not_found += 1
        
        print(f"Words found in Word2Vec: {words_found}")
        print(f"Words not found (random init): {words_not_found}")
        print(f"Coverage: {words_found / (words_found + words_not_found) * 100:.2f}%")
        
        return embedding_matrix
    
    def get_similar_words(self, word, topn=10):
        """
        Get most similar words to a given word.
        
        Args:
            word: Query word
            topn: Number of similar words to return
            
        Returns:
            List of (word, similarity) tuples
        """
        if self.model is None:
            raise ValueError("Word2Vec model not trained.")
        
        if word in self.model.wv:
            return self.model.wv.most_similar(word, topn=topn)
        else:
            return []
    
    def get_word_vector(self, word):
        """Get the vector for a word."""
        if self.model is None:
            raise ValueError("Word2Vec model not trained.")
        
        if word in self.model.wv:
            return self.model.wv[word]
        return None
    
    def save(self, filepath=None):
        """Save Word2Vec model to disk."""
        if filepath is None:
            filepath = os.path.join(MODEL_DIR, "word2vec.model")
        
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        print(f"Word2Vec model saved to {filepath}")
    
    def load(self, filepath=None):
        """Load Word2Vec model from disk."""
        if filepath is None:
            filepath = os.path.join(MODEL_DIR, "word2vec.model")
        
        self.model = Word2Vec.load(filepath)
        self.vector_size = self.model.vector_size
        print(f"Word2Vec model loaded from {filepath}")
        return self


def visualize_embeddings(w2v_model, words, save_path=None):
    """
    Visualize word embeddings using t-SNE.
    
    Args:
        w2v_model: Trained Word2Vec model
        words: List of words to visualize
        save_path: Path to save the plot
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    # Get vectors for words that exist in vocabulary
    valid_words = [w for w in words if w in w2v_model.wv]
    vectors = np.array([w2v_model.wv[w] for w in valid_words])
    
    if len(valid_words) < 2:
        print("Not enough valid words to visualize.")
        return
    
    # Reduce dimensions
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(valid_words)-1))
    vectors_2d = tsne.fit_transform(vectors)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6)
    
    for i, word in enumerate(valid_words):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))
    
    plt.title("Word2Vec Embeddings Visualization")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test Word2Vec
    test_sentences = [
        ["love", "this", "movie", "amazing"],
        ["terrible", "awful", "bad", "experience"],
        ["not", "bad", "actually", "good"],
        ["happy", "great", "day", "wonderful"],
        ["love", "great", "amazing", "wonderful"],
        ["bad", "terrible", "awful", "horrible"]
    ]
    
    w2v = Word2VecEmbedding(vector_size=50, window=3, min_count=1)
    w2v.train(test_sentences, epochs=100)
    
    print("\nSimilar to 'love':", w2v.get_similar_words('love', topn=3))
    print("Similar to 'bad':", w2v.get_similar_words('bad', topn=3))
