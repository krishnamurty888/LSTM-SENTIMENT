"""
Dataset Download Script

Downloads the Sentiment140 dataset or creates sample data for testing.
"""

import os
import sys
import urllib.request
import zipfile
import csv

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, "training.1600000.processed.noemoticon.csv")


def create_sample_dataset(n_samples=10000):
    """
    Create a sample dataset for testing when the full dataset is not available.
    Uses realistic tweet-like text samples.
    """
    import random
    
    print(f"Creating sample dataset with {n_samples} tweets...")
    
    # Positive tweet templates
    positive_templates = [
        "I love this so much! {adj} experience!",
        "This is amazing! Really {adj}!",
        "Best {noun} ever! Highly recommend!",
        "So happy with this! {adj} quality!",
        "Great {noun}! Love it!",
        "Awesome! This made my day!",
        "Perfect! Exactly what I needed!",
        "Fantastic {noun}! Very {adj}!",
        "Love love love this!",
        "This is wonderful! So {adj}!",
        "Excellent {noun}! Five stars!",
        "Super {adj}! Would buy again!",
        "Amazing experience! Very {adj}!",
        "Best purchase! So {adj}!",
        "Incredible! Absolutely {adj}!",
        "Thank you! This is {adj}!",
        "Wow! Really {adj} {noun}!",
        "Happy customer here! {adj}!",
        "This rocks! Very {adj}!",
        "Brilliant! Loved the {noun}!",
    ]
    
    # Negative tweet templates
    negative_templates = [
        "This is terrible! Very {adj}!",
        "Worst {noun} ever! Don't buy!",
        "So disappointed! Really {adj}!",
        "Hate this! Completely {adj}!",
        "Awful experience! Very {adj}!",
        "Never buying again! Too {adj}!",
        "This sucks! Totally {adj}!",
        "Horrible {noun}! So {adj}!",
        "Waste of money! Very {adj}!",
        "Terrible quality! So {adj}!",
        "Disappointed! Really {adj}!",
        "Bad experience! Very {adj}!",
        "Worst purchase! So {adj}!",
        "Don't recommend! Too {adj}!",
        "Ruined my day! Very {adj}!",
        "Frustrating! Really {adj}!",
        "Useless {noun}! So {adj}!",
        "Regret buying this! {adj}!",
        "Broken already! Very {adj}!",
        "Poor quality! So {adj}!",
    ]
    
    positive_adj = ["great", "amazing", "wonderful", "fantastic", "excellent", "perfect", "awesome", "lovely", "beautiful", "outstanding"]
    negative_adj = ["bad", "terrible", "awful", "horrible", "poor", "disappointing", "frustrating", "annoying", "useless", "broken"]
    nouns = ["product", "service", "experience", "purchase", "item", "delivery", "quality", "support", "day", "week"]
    
    samples = []
    
    for i in range(n_samples):
        if i % 2 == 0:
            # Positive (target = 4)
            template = random.choice(positive_templates)
            adj = random.choice(positive_adj)
            noun = random.choice(nouns)
            text = template.format(adj=adj, noun=noun)
            target = 4
        else:
            # Negative (target = 0)
            template = random.choice(negative_templates)
            adj = random.choice(negative_adj)
            noun = random.choice(nouns)
            text = template.format(adj=adj, noun=noun)
            target = 0
        
        # Add some variation
        if random.random() > 0.7:
            text = text.upper()
        if random.random() > 0.8:
            text = text + " " + random.choice(["!!!", "...", "???", ":)", ":(", "@user"])
        
        samples.append({
            'target': target,
            'id': i + 1,
            'date': 'Mon Jan 01 00:00:00 UTC 2024',
            'flag': 'NO_QUERY',
            'user': f'user{i}',
            'text': text
        })
    
    # Shuffle
    random.shuffle(samples)
    
    # Write to CSV
    print(f"Writing to {DATASET_PATH}...")
    with open(DATASET_PATH, 'w', newline='', encoding='latin-1') as f:
        writer = csv.writer(f)
        for sample in samples:
            writer.writerow([
                sample['target'],
                sample['id'],
                sample['date'],
                sample['flag'],
                sample['user'],
                sample['text']
            ])
    
    print(f"[OK] Sample dataset created: {DATASET_PATH}")
    print(f"  Total samples: {len(samples)}")
    print(f"  Positive: {sum(1 for s in samples if s['target'] == 4)}")
    print(f"  Negative: {sum(1 for s in samples if s['target'] == 0)}")
    
    return DATASET_PATH


def download_sentiment140():
    """
    Attempt to download Sentiment140 dataset.
    Note: This may not work as Kaggle requires authentication.
    """
    print("=" * 60)
    print("SENTIMENT140 DATASET DOWNLOAD")
    print("=" * 60)
    print("\nThe Sentiment140 dataset needs to be downloaded from Kaggle:")
    print("  URL: https://www.kaggle.com/datasets/kazanova/sentiment140")
    print("\nSteps:")
    print("  1. Create a Kaggle account (free)")
    print("  2. Download 'training.1600000.processed.noemoticon.csv'")
    print(f"  3. Place it in: {DATA_DIR}")
    print("\nAlternatively, a sample dataset will be created for testing.")
    print("=" * 60)
    
    return None


def check_and_prepare_data():
    """
    Check if dataset exists, if not create sample data.
    """
    if os.path.exists(DATASET_PATH):
        # Check file size
        size_mb = os.path.getsize(DATASET_PATH) / (1024 * 1024)
        print(f"[OK] Dataset found: {DATASET_PATH}")
        print(f"  Size: {size_mb:.1f} MB")
        return DATASET_PATH
    
    print("Dataset not found.")
    print("\nOptions:")
    print("  1. Download from Kaggle (recommended for best results)")
    print("  2. Create sample dataset (for quick testing)")
    
    # Create sample dataset automatically
    print("\nCreating sample dataset for testing...")
    return create_sample_dataset(n_samples=20000)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset preparation')
    parser.add_argument('--sample', action='store_true', help='Create sample dataset')
    parser.add_argument('--samples', type=int, default=20000, help='Number of samples')
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_dataset(n_samples=args.samples)
    else:
        check_and_prepare_data()
