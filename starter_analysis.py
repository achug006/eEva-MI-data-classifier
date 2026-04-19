"""
Change Talk Analysis - Starter Implementation Script
This script implements the initial phases of sentiment analysis and BERT-based classification
for Motivational Interviewing therapist responses.
"""

import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# =====================================
# PHASE 1: DATA LOADING & PREPARATION
# =====================================

print("Loading dataset...")
df = pd.read_csv('/mnt/user-data/uploads/MI_Dataset_English_Filtered.csv')

# Convert string representation of lists to actual lists
print("Preprocessing tokens...")
df['tokens'] = df['tokens'].apply(literal_eval)

# Basic statistics
print(f"\nDataset Shape: {df.shape}")
print(f"Number of dialogues: {df['dialog_id'].nunique()}")
print(f"Number of unique MI labels: {df['final agreed label'].nunique()}")

# =====================================
# PHASE 2: SENTIMENT MAPPING
# =====================================

print("\n" + "="*60)
print("PHASE 2: Creating Sentiment Categories")
print("="*60)

# Theory-driven sentiment mapping based on Motivational Interviewing principles
sentiment_mapping = {
    # Positive: Supportive, empowering, collaborative
    'Affirm': 'positive',
    'Support': 'positive',
    'Complex Reflection': 'positive',
    'Simple Reflection': 'positive',
    'Emphasize Autonomy': 'positive',
    'Advise with Permission': 'positive',
    'Open Question': 'positive',
    
    # Negative: Confrontational, directive, limiting autonomy
    'Confront': 'negative',
    'Warn': 'negative',
    'Direct': 'negative',
    'Advise without Permission': 'negative',
    
    # Neutral: Informational, procedural
    'Give Information': 'neutral',
    'Closed Question': 'neutral',
    'Self-Disclose': 'neutral',
    'Other': 'neutral'
}

df['sentiment_category'] = df['final agreed label'].map(sentiment_mapping)

print("\nSentiment Distribution:")
print(df['sentiment_category'].value_counts())
print(f"\nPercentages:")
print(df['sentiment_category'].value_counts(normalize=True).round(3) * 100)

# =====================================
# PHASE 3: LEXICON-BASED SENTIMENT
# =====================================

print("\n" + "="*60)
print("PHASE 3: Lexicon-Based Sentiment Analysis")
print("="*60)

try:
    from textblob import TextBlob
    print("Calculating TextBlob sentiment...")
    df['textblob_polarity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['textblob_sentiment'] = df['textblob_polarity'].apply(
        lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
    )
    print("TextBlob sentiment calculated successfully!")
except ImportError:
    print("TextBlob not installed. Skipping...")
    df['textblob_sentiment'] = 'neutral'

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    print("Calculating VADER sentiment...")
    vader = SentimentIntensityAnalyzer()
    df['vader_compound'] = df['text'].apply(
        lambda x: vader.polarity_scores(str(x))['compound']
    )
    df['vader_sentiment'] = df['vader_compound'].apply(
        lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
    )
    print("VADER sentiment calculated successfully!")
except ImportError:
    print("VADER not installed. Skipping...")
    df['vader_sentiment'] = 'neutral'

# =====================================
# PHASE 4: DIALOGUE PATTERN ANALYSIS
# =====================================

print("\n" + "="*60)
print("PHASE 4: Change Talk Pattern Analysis")
print("="*60)

print("Analyzing strategy transitions within dialogues...")

dialogue_groups = df.groupby('dialog_id')

def analyze_strategy_shifts(group):
    """Identify changes in therapist strategy within a conversation"""
    if len(group) < 2:
        return None
    
    group = group.sort_values('turn')
    labels = group['final agreed label'].tolist()
    sentiments = group['sentiment_category'].tolist()
    
    # Count transitions
    transitions = []
    for i in range(len(labels)-1):
        transitions.append((labels[i], labels[i+1]))
    
    return {
        'total_turns': len(labels),
        'unique_strategies': len(set(labels)),
        'transitions': transitions,
        'sentiment_progression': sentiments
    }

# Analyze patterns
dialogue_patterns = []
for dialog_id, group in dialogue_groups:
    pattern = analyze_strategy_shifts(group)
    if pattern is not None:
        dialogue_patterns.append(pattern)

# Extract all transitions
all_transitions = []
for pattern in dialogue_patterns:
    all_transitions.extend(pattern['transitions'])

# Count most common transitions
transition_counts = Counter(all_transitions)
top_transitions = transition_counts.most_common(15)

print(f"\nAnalyzed {len(dialogue_patterns)} dialogues")
print(f"Total strategy transitions: {len(all_transitions)}")
print("\nTop 15 Most Common Strategy Transitions:")
print("-" * 70)
for i, ((from_label, to_label), count) in enumerate(top_transitions, 1):
    print(f"{i:2d}. {from_label:30s} → {to_label:30s} ({count:4d} times)")

# =====================================
# PHASE 5: LINGUISTIC FEATURES
# =====================================

print("\n" + "="*60)
print("PHASE 5: Linguistic Feature Extraction")
print("="*60)

print("Extracting basic linguistic features...")

# Basic features without spacy
df['text_length'] = df['text'].str.len()
df['word_count_calc'] = df['text'].str.split().str.len()
df['num_questions'] = df['text'].str.count(r'\?')
df['num_exclamations'] = df['text'].str.count('!')
df['has_first_person'] = df['text'].str.lower().str.contains(r'\b(i|we|my|our)\b')
df['has_second_person'] = df['text'].str.lower().str.contains(r'\b(you|your)\b')

print("Linguistic features extracted!")

# Feature summary by sentiment
print("\nAverage linguistic features by sentiment category:")
feature_cols = ['text_length', 'word_count_calc', 'num_questions', 'num_exclamations']
print(df.groupby('sentiment_category')[feature_cols].mean().round(2))

# =====================================
# PHASE 6: VISUALIZATIONS
# =====================================

print("\n" + "="*60)
print("PHASE 6: Creating Visualizations")
print("="*60)

# Create output directory
import os
os.makedirs('/home/claude/visualizations', exist_ok=True)

# 1. Sentiment distribution by MI label
print("Creating sentiment distribution plot...")
plt.figure(figsize=(14, 6))
sentiment_by_label = df.groupby(['final agreed label', 'sentiment_category']).size().unstack(fill_value=0)
sentiment_by_label.plot(kind='bar', stacked=True, color=['#d62728', '#ffbb33', '#2ca02c'])
plt.title('Motivational Interviewing Technique Distribution by Sentiment Category', fontsize=14, fontweight='bold')
plt.xlabel('MI Technique', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Sentiment', fontsize=10)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('/home/claude/visualizations/sentiment_distribution.png', dpi=300)
plt.close()
print("✓ Sentiment distribution saved")

# 2. Transition heatmap (top techniques only)
print("Creating transition matrix heatmap...")
top_labels = df['final agreed label'].value_counts().head(10).index.tolist()
filtered_transitions = [(f, t) for f, t in all_transitions if f in top_labels and t in top_labels]

transition_matrix = pd.DataFrame(0, index=top_labels, columns=top_labels)
for from_label, to_label in filtered_transitions:
    transition_matrix.loc[from_label, to_label] += 1

plt.figure(figsize=(12, 10))
sns.heatmap(transition_matrix, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Frequency'})
plt.title('Therapist Strategy Transition Matrix (Top 10 Techniques)', fontsize=14, fontweight='bold')
plt.xlabel('Next Strategy', fontsize=12)
plt.ylabel('Current Strategy', fontsize=12)
plt.tight_layout()
plt.savefig('/home/claude/visualizations/transition_matrix.png', dpi=300)
plt.close()
print("✓ Transition matrix saved")

# 3. Word count distribution by sentiment
print("Creating word count distribution plot...")
plt.figure(figsize=(10, 6))
for sentiment in ['positive', 'neutral', 'negative']:
    subset = df[df['sentiment_category'] == sentiment]['word_count_calc']
    plt.hist(subset, bins=30, alpha=0.5, label=sentiment)
plt.xlabel('Word Count', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Response Length Distribution by Sentiment Category', fontsize=14, fontweight='bold')
plt.legend()
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig('/home/claude/visualizations/word_count_distribution.png', dpi=300)
plt.close()
print("✓ Word count distribution saved")

# 4. Question usage by technique
print("Creating question usage analysis...")
question_usage = df.groupby('final agreed label')['num_questions'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
question_usage.plot(kind='barh', color='steelblue')
plt.xlabel('Average Number of Questions per Response', fontsize=12)
plt.ylabel('MI Technique', fontsize=12)
plt.title('Question Usage by Motivational Interviewing Technique', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/visualizations/question_usage.png', dpi=300)
plt.close()
print("✓ Question usage plot saved")

# =====================================
# PHASE 7: SUMMARY REPORT
# =====================================

print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)

print("\n1. Dataset Overview:")
print(f"   - Total responses: {len(df):,}")
print(f"   - Unique dialogues: {df['dialog_id'].nunique():,}")
print(f"   - Unique MI techniques: {df['final agreed label'].nunique()}")

print("\n2. Sentiment Distribution:")
for sentiment, count in df['sentiment_category'].value_counts().items():
    pct = (count / len(df)) * 100
    print(f"   - {sentiment.capitalize()}: {count:,} ({pct:.1f}%)")

print("\n3. Top 5 MI Techniques:")
for i, (label, count) in enumerate(df['final agreed label'].value_counts().head(5).items(), 1):
    pct = (count / len(df)) * 100
    sentiment = sentiment_mapping[label]
    print(f"   {i}. {label} ({sentiment}): {count:,} ({pct:.1f}%)")

print("\n4. Dialogue Characteristics:")
avg_turns = df.groupby('dialog_id').size().mean()
print(f"   - Average turns per dialogue: {avg_turns:.1f}")
print(f"   - Total unique strategy transitions: {len(set(all_transitions))}")
print(f"   - Most common transition: {top_transitions[0][0][0]} → {top_transitions[0][0][1]} ({top_transitions[0][1]} times)")

print("\n5. Linguistic Patterns:")
print(f"   - Average response length: {df['text_length'].mean():.1f} characters")
print(f"   - Average word count: {df['word_count_calc'].mean():.1f} words")
print(f"   - Responses with questions: {(df['num_questions'] > 0).sum():,} ({(df['num_questions'] > 0).mean()*100:.1f}%)")
print(f"   - Responses with second-person pronouns: {df['has_second_person'].sum():,} ({df['has_second_person'].mean()*100:.1f}%)")

# Save processed dataframe
print("\n" + "="*60)
print("Saving processed dataset...")
df.to_csv('/home/claude/processed_mi_dataset.csv', index=False)
print("✓ Processed dataset saved to: /home/claude/processed_mi_dataset.csv")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nOutputs saved:")
print("  - Processed dataset: /home/claude/processed_mi_dataset.csv")
print("  - Visualizations: /home/claude/visualizations/")
print("\nNext Steps:")
print("  1. Review the visualizations to understand patterns")
print("  2. Implement BERT fine-tuning (see guide for details)")
print("  3. Build predictive models for strategy recommendations")
print("  4. Conduct deeper sequential pattern analysis")
print("="*60)
