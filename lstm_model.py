"""
LSTM Model Module for Sentiment Classification

Implements LSTM and Bidirectional LSTM architectures with Word2Vec embeddings.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Bidirectional, Dense, Dropout, 
    SpatialDropout1D, GlobalMaxPooling1D, Input, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from config import (
    EMBEDDING_DIM, LSTM_UNITS_1, LSTM_UNITS_2, DROPOUT_RATE,
    MAX_SEQUENCE_LENGTH, LEARNING_RATE, MODEL_DIR, USE_BIDIRECTIONAL
)


def create_lstm_model(vocab_size, embedding_matrix, max_length=MAX_SEQUENCE_LENGTH,
                      embedding_dim=EMBEDDING_DIM, lstm_units_1=LSTM_UNITS_1,
                      lstm_units_2=LSTM_UNITS_2, dropout_rate=DROPOUT_RATE,
                      use_bidirectional=USE_BIDIRECTIONAL, trainable_embedding=False):
    """
    Create LSTM model for sentiment classification.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_matrix: Pre-trained embedding matrix from Word2Vec
        max_length: Maximum sequence length
        embedding_dim: Embedding dimension
        lstm_units_1: Units in first LSTM layer
        lstm_units_2: Units in second LSTM layer
        dropout_rate: Dropout rate
        use_bidirectional: If True, use Bidirectional LSTM
        trainable_embedding: If True, allow embedding weights to be updated
        
    Returns:
        Compiled Keras model
    """
    model = Sequential(name="Sentiment_LSTM")
    
    # Embedding layer with pre-trained Word2Vec weights
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=trainable_embedding,
        name="word2vec_embedding"
    ))
    
    # Spatial dropout for regularization
    model.add(SpatialDropout1D(dropout_rate, name="spatial_dropout"))
    
    # First LSTM layer
    if use_bidirectional:
        model.add(Bidirectional(
            LSTM(lstm_units_1, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            name="bidirectional_lstm_1"
        ))
    else:
        model.add(LSTM(
            lstm_units_1, 
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            name="lstm_1"
        ))
    
    model.add(Dropout(dropout_rate, name="dropout_1"))
    
    # Second LSTM layer
    if use_bidirectional:
        model.add(Bidirectional(
            LSTM(lstm_units_2, dropout=0.2, recurrent_dropout=0.2),
            name="bidirectional_lstm_2"
        ))
    else:
        model.add(LSTM(
            lstm_units_2,
            dropout=0.2,
            recurrent_dropout=0.2,
            name="lstm_2"
        ))
    
    model.add(Dropout(dropout_rate, name="dropout_2"))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid', name="output"))
    
    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


def create_advanced_lstm_model(vocab_size, embedding_matrix, max_length=MAX_SEQUENCE_LENGTH,
                               embedding_dim=EMBEDDING_DIM):
    """
    Create advanced LSTM model with attention-like mechanism.
    
    This model uses GlobalMaxPooling to capture the most important features
    and concatenates with the final LSTM output.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_matrix: Pre-trained embedding matrix
        max_length: Maximum sequence length
        embedding_dim: Embedding dimension
        
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=(max_length,), name="input")
    
    # Embedding layer
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=False,
        name="embedding"
    )(inputs)
    
    # Spatial dropout
    x = SpatialDropout1D(0.3)(x)
    
    # Bidirectional LSTM
    lstm_out = Bidirectional(LSTM(128, return_sequences=True))(x)
    
    # Global max pooling to capture important features
    max_pool = GlobalMaxPooling1D()(lstm_out)
    
    # Second LSTM layer
    lstm_final = Bidirectional(LSTM(64))(lstm_out)
    
    # Concatenate max pool and lstm output
    concat = Concatenate()([max_pool, lstm_final])
    
    # Dense layers
    x = Dropout(0.4)(concat)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output
    outputs = Dense(1, activation='sigmoid', name="output")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Advanced_LSTM")
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(model_name="lstm_sentiment"):
    """
    Get training callbacks for model training.
    
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        # Save best model
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"{model_name}_best.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Reduce learning rate when stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    return callbacks


def print_model_summary(model):
    """Print model summary and architecture details."""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test model creation
    vocab_size = 20000
    embedding_dim = 50
    max_length = 50
    
    # Create dummy embedding matrix
    embedding_matrix = np.random.randn(vocab_size, embedding_dim)
    
    # Create standard LSTM model
    print("Creating Standard LSTM Model:")
    model = create_lstm_model(
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        use_bidirectional=False
    )
    print_model_summary(model)
    
    # Create Bidirectional LSTM model
    print("\nCreating Bidirectional LSTM Model:")
    model_bi = create_lstm_model(
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        use_bidirectional=True
    )
    print_model_summary(model_bi)
    
    # Create Advanced model
    print("\nCreating Advanced LSTM Model:")
    model_adv = create_advanced_lstm_model(
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix
    )
    print_model_summary(model_adv)
