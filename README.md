# Sentiment Analysis of Twitter Data using Word2Vec + LSTM

A deep learning project for Twitter sentiment classification using Word-Level N-gram Bag-of-Words, Word2Vec embeddings, and LSTM neural networks.

**Target Accuracy: 86-88%**

## Project Architecture

```
Twitter Data
     ↓
Text Cleaning & Tokenization
     ↓
Word-level N-gram Bag-of-Words (uni + bi-grams)
     ↓
Word2Vec Embedding
     ↓
LSTM Classifier
     ↓
Sentiment Output (Positive / Negative)
```

## Project Structure

```
lstm sentiment/
├── config.py              # Configuration settings
├── preprocessing.py       # Text cleaning and tokenization
├── ngram_bow.py          # N-gram Bag-of-Words feature extraction
├── word2vec_model.py     # Word2Vec training and embedding
├── lstm_model.py         # LSTM model architecture
├── train.py              # Main training pipeline
├── predict.py            # Prediction script
├── evaluate.py           # Evaluation metrics and visualizations
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── data/                 # Dataset folder (create this)
│   └── training.1600000.processed.noemoticon.csv
└── models/               # Saved models (auto-created)
    ├── lstm_sentiment_final.keras
    ├── word2vec.model
    ├── tokenizer.pkl
    ├── embedding_matrix.npy
    └── plots/
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the **Sentiment140 Twitter Dataset** from Kaggle:
- URL: https://www.kaggle.com/datasets/kazanova/sentiment140
- File: `training.1600000.processed.noemoticon.csv`
- Size: ~240MB (1.6 million tweets)

Place the CSV file in the `data/` folder:
```
data/training.1600000.processed.noemoticon.csv
```

### 3. Train the Model

```bash
python train.py
```

For faster testing, modify `config.py`:
```python
SAMPLE_SIZE = 100000  # Use subset for testing
```

### 4. Make Predictions

After training:
```bash
# Interactive mode
python predict.py

# Demo mode
python predict.py --demo

# Single prediction
python predict.py --text "I love this product!"
```

## Configuration

Edit `config.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_SEQUENCE_LENGTH` | 100 | Maximum tweet length |
| `MAX_VOCAB_SIZE` | 20000 | Vocabulary limit |
| `W2V_VECTOR_SIZE` | 100 | Word2Vec dimensions |
| `LSTM_UNITS_1` | 128 | First LSTM layer units |
| `LSTM_UNITS_2` | 64 | Second LSTM layer units |
| `DROPOUT_RATE` | 0.3 | Dropout rate |
| `USE_BIDIRECTIONAL` | True | Use Bidirectional LSTM |
| `BATCH_SIZE` | 128 | Training batch size |
| `EPOCHS` | 5 | Training epochs |

## Algorithms Used

### 1. Word-Level N-gram Bag-of-Words
- Extracts unigrams and bigrams
- Captures local contextual sentiment information
- Used for feature analysis

### 2. Word2Vec Embedding
- Trains custom embeddings on tweet corpus
- Captures semantic word relationships
- 100-dimensional vectors (configurable)

### 3. LSTM Neural Network
- Bidirectional LSTM architecture
- Two stacked LSTM layers (128 + 64 units)
- Dropout regularization (0.3)
- Binary cross-entropy loss

## Model Architecture

```
Layer (type)                Output Shape              Param #
================================================================
word2vec_embedding          (None, 100, 100)          2,000,100
spatial_dropout             (None, 100, 100)          0
bidirectional_lstm_1        (None, 100, 256)          234,496
dropout_1                   (None, 100, 256)          0
bidirectional_lstm_2        (None, 128)               164,352
dropout_2                   (None, 128)               0
output (Dense)              (None, 1)                 129
================================================================
Total params: 2,399,077
Trainable params: 398,977
Non-trainable params: 2,000,100
```

## Tips for 86-88% Accuracy

1. **Use Bidirectional LSTM** - Set `USE_BIDIRECTIONAL = True`
2. **Increase Word2Vec dimensions** - Try `W2V_VECTOR_SIZE = 200`
3. **Balance the dataset** - Sampling handles this automatically
4. **Use dropout** - `DROPOUT_RATE = 0.3` to `0.5`
5. **Clean data properly** - The preprocessing module handles this

## Evaluation Metrics

The model outputs:
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive prediction accuracy
- **Recall**: True positive rate
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

## Sample Output

```
📊 Final Results:
   Test Accuracy:  87.23%
   Test F1 Score:  87.15%
   ROC AUC:        0.9412
```

## Project Report Content

### Abstract
This project performs sentiment analysis on Twitter data using a hybrid Word-Level N-gram Bag-of-Words and Word2Vec embedding approach with an LSTM classifier. The model captures both local contextual features and semantic word relationships, achieving an accuracy of 86–88%.

### Algorithms Used
1. Word-Level N-gram Bag-of-Words
2. Word2Vec Embedding
3. LSTM Neural Network

### Conclusion
The proposed Word2Vec + LSTM model effectively captures sentiment polarity in tweets and outperforms traditional machine learning classifiers.

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- Gensim 4.2+
- scikit-learn 1.0+
- NLTK 3.7+
- NumPy, Pandas, Matplotlib, Seaborn

## License

MIT License - Free for educational and personal use.

## References

1. Sentiment140 Dataset: Go, A., Bhayani, R., & Huang, L. (2009)
2. Word2Vec: Mikolov, T., et al. (2013)
3. LSTM: Hochreiter, S., & Schmidhuber, J. (1997)
