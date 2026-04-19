"""
BERT Fine-Tuning for MI Technique Classification
This script fine-tunes a BERT model to classify therapist responses into MI techniques
and optionally creates embeddings for further analysis.
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    BertModel,
    Trainer, 
    TrainingArguments
)
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("BERT-BASED CHANGE TALK ANALYSIS")
print("="*70)

# =====================================
# CONFIGURATION
# =====================================

CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_length': 128,
    'batch_size': 16,
    'num_epochs': 3,
    'learning_rate': 2e-5,
    'test_size': 0.2,
    'random_state': 42,
    'output_dir': '/home/claude/bert_models'
}

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# =====================================
# LOAD DATA
# =====================================

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

df = pd.read_csv('/home/claude/processed_mi_dataset.csv')
print(f"Loaded {len(df):,} responses")
print(f"Unique MI techniques: {df['final agreed label'].nunique()}")

# Create label encoding
label_list = sorted(df['final agreed label'].unique())
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

df['label_id'] = df['final agreed label'].map(label2id)

print(f"\nLabel encoding created for {len(label2id)} classes")

# =====================================
# PREPARE DATASET
# =====================================

print("\n" + "="*70)
print("PREPARING TRAIN/TEST SPLIT")
print("="*70)

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(),
    df['label_id'].tolist(),
    test_size=CONFIG['test_size'],
    stratify=df['label_id'],
    random_state=CONFIG['random_state']
)

print(f"Training samples: {len(train_texts):,}")
print(f"Testing samples: {len(test_texts):,}")

# =====================================
# TOKENIZATION
# =====================================

print("\n" + "="*70)
print("TOKENIZING TEXT")
print("="*70)

tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=CONFIG['max_length'],
    return_tensors='pt'
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=CONFIG['max_length'],
    return_tensors='pt'
)

print("✓ Tokenization complete")

# =====================================
# CREATE PYTORCH DATASET
# =====================================

class MIDataset(Dataset):
    """Custom Dataset for MI technique classification"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = MIDataset(train_encodings, train_labels)
test_dataset = MIDataset(test_encodings, test_labels)

print(f"✓ PyTorch datasets created")

# =====================================
# MODEL SETUP
# =====================================

print("\n" + "="*70)
print("SETTING UP MODEL")
print("="*70)

model = BertForSequenceClassification.from_pretrained(
    CONFIG['model_name'],
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

model.to(device)
print(f"✓ Model loaded: {CONFIG['model_name']}")
print(f"✓ Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# =====================================
# TRAINING CONFIGURATION
# =====================================

training_args = TrainingArguments(
    output_dir=CONFIG['output_dir'],
    num_train_epochs=CONFIG['num_epochs'],
    per_device_train_batch_size=CONFIG['batch_size'],
    per_device_eval_batch_size=CONFIG['batch_size'] * 2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f"{CONFIG['output_dir']}/logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    push_to_hub=False,
    report_to="none"  # Disable wandb/tensorboard
)

# =====================================
# TRAINING
# =====================================

print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70)
print(f"Epochs: {CONFIG['num_epochs']}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Learning rate: {CONFIG['learning_rate']}")
print("\nTraining in progress...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
train_result = trainer.train()

print("\n✓ Training complete!")
print(f"Final training loss: {train_result.metrics['train_loss']:.4f}")

# =====================================
# EVALUATION
# =====================================

print("\n" + "="*70)
print("EVALUATING MODEL")
print("="*70)

# Get predictions
predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = test_labels

# Classification report
print("\nClassification Report:")
print("="*70)
report = classification_report(
    true_labels, 
    pred_labels, 
    target_names=label_list,
    digits=3
)
print(report)

# Save classification report
with open('/home/claude/bert_classification_report.txt', 'w') as f:
    f.write("BERT Model Classification Report\n")
    f.write("="*70 + "\n\n")
    f.write(report)
print("✓ Classification report saved")

# Calculate per-class metrics
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    true_labels, pred_labels, labels=range(len(label_list))
)

# Create results dataframe
results_df = pd.DataFrame({
    'MI_Technique': label_list,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
}).sort_values('F1-Score', ascending=False)

print("\nTop 10 Best Performing Techniques:")
print(results_df.head(10).to_string(index=False))

results_df.to_csv('/home/claude/bert_per_class_performance.csv', index=False)
print("\n✓ Per-class performance saved")

# =====================================
# CONFUSION MATRIX
# =====================================

print("\n" + "="*70)
print("CREATING CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(true_labels, pred_labels)

# Plot confusion matrix (full)
plt.figure(figsize=(16, 14))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=label_list,
    yticklabels=label_list
)
plt.title('BERT Model Confusion Matrix - All MI Techniques', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/home/claude/visualizations/bert_confusion_matrix_full.png', dpi=300)
plt.close()
print("✓ Full confusion matrix saved")

# Plot confusion matrix (top 10 techniques only)
top_10_indices = results_df.head(10)['MI_Technique'].tolist()
top_10_ids = [label2id[label] for label in top_10_indices]

# Filter confusion matrix
mask = np.isin(true_labels, top_10_ids) & np.isin(pred_labels, top_10_ids)
filtered_true = [true_labels[i] for i in range(len(true_labels)) if mask[i]]
filtered_pred = [pred_labels[i] for i in range(len(pred_labels)) if mask[i]]

# Remap to 0-9
label_mapping = {old_id: new_id for new_id, old_id in enumerate(top_10_ids)}
remapped_true = [label_mapping[label] for label in filtered_true]
remapped_pred = [label_mapping[label] for label in filtered_pred]

cm_top10 = confusion_matrix(remapped_true, remapped_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm_top10,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=top_10_indices,
    yticklabels=top_10_indices
)
plt.title('BERT Model Confusion Matrix - Top 10 MI Techniques', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/home/claude/visualizations/bert_confusion_matrix_top10.png', dpi=300)
plt.close()
print("✓ Top-10 confusion matrix saved")

# =====================================
# EXTRACT BERT EMBEDDINGS
# =====================================

print("\n" + "="*70)
print("EXTRACTING BERT EMBEDDINGS")
print("="*70)

# Load BERT model (without classification head) for embeddings
embedding_model = BertModel.from_pretrained(CONFIG['model_name'])
embedding_model.to(device)
embedding_model.eval()

def get_bert_embedding(text, model, tokenizer, device):
    """Extract [CLS] token embedding from BERT"""
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=CONFIG['max_length']
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token embedding (first token)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# Extract embeddings for a sample (extracting all would be memory intensive)
print("Extracting embeddings for 1000 random samples...")
sample_indices = np.random.choice(len(df), size=1000, replace=False)
sample_df = df.iloc[sample_indices].copy()

embeddings = []
for text in sample_df['text'].tolist():
    emb = get_bert_embedding(text, embedding_model, tokenizer, device)
    embeddings.append(emb)

embeddings_array = np.array(embeddings)
print(f"✓ Extracted embeddings shape: {embeddings_array.shape}")

# Save embeddings
np.save('/home/claude/bert_embeddings_sample.npy', embeddings_array)
sample_df.to_csv('/home/claude/bert_embeddings_metadata.csv', index=False)
print("✓ Embeddings saved")

# =====================================
# VISUALIZE EMBEDDINGS (t-SNE)
# =====================================

print("\n" + "="*70)
print("VISUALIZING EMBEDDINGS WITH t-SNE")
print("="*70)

from sklearn.manifold import TSNE

print("Computing t-SNE (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=CONFIG['random_state'], perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings_array)

# Plot by sentiment
plt.figure(figsize=(12, 8))
sentiment_colors = {'positive': '#2ca02c', 'neutral': '#ffbb33', 'negative': '#d62728'}

for sentiment in ['positive', 'neutral', 'negative']:
    mask = sample_df['sentiment_category'] == sentiment
    plt.scatter(
        embeddings_2d[mask, 0],
        embeddings_2d[mask, 1],
        c=sentiment_colors[sentiment],
        label=sentiment.capitalize(),
        alpha=0.6,
        s=50
    )

plt.title('BERT Embeddings: Therapist Responses by Sentiment (t-SNE)', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/visualizations/bert_tsne_sentiment.png', dpi=300)
plt.close()
print("✓ t-SNE sentiment visualization saved")

# Plot by MI technique (top 5 only for clarity)
top_5_labels = df['final agreed label'].value_counts().head(5).index.tolist()
plt.figure(figsize=(14, 10))

colors = plt.cm.Set3(np.linspace(0, 1, len(top_5_labels)))
for i, label in enumerate(top_5_labels):
    mask = sample_df['final agreed label'] == label
    plt.scatter(
        embeddings_2d[mask, 0],
        embeddings_2d[mask, 1],
        c=[colors[i]],
        label=label,
        alpha=0.6,
        s=50
    )

plt.title('BERT Embeddings: Top 5 MI Techniques (t-SNE)', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Component 1', fontsize=12)
plt.ylabel('t-SNE Component 2', fontsize=12)
plt.legend(fontsize=9, loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/visualizations/bert_tsne_techniques.png', dpi=300)
plt.close()
print("✓ t-SNE technique visualization saved")

# =====================================
# SAVE MODEL
# =====================================

print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)

model.save_pretrained(f"{CONFIG['output_dir']}/final_model")
tokenizer.save_pretrained(f"{CONFIG['output_dir']}/final_model")
print(f"✓ Model saved to: {CONFIG['output_dir']}/final_model")

# =====================================
# SUMMARY
# =====================================

print("\n" + "="*70)
print("BERT ANALYSIS COMPLETE")
print("="*70)

overall_accuracy = (pred_labels == true_labels).mean()
print(f"\nOverall Accuracy: {overall_accuracy:.3f}")
print(f"Macro-averaged F1: {results_df['F1-Score'].mean():.3f}")
print(f"Weighted-averaged F1: {f1.mean():.3f}")

print("\nOutputs saved:")
print(f"  - Model: {CONFIG['output_dir']}/final_model/")
print(f"  - Classification report: /home/claude/bert_classification_report.txt")
print(f"  - Per-class performance: /home/claude/bert_per_class_performance.csv")
print(f"  - Embeddings: /home/claude/bert_embeddings_sample.npy")
print(f"  - Visualizations: /home/claude/visualizations/")

print("\nBest performing techniques:")
for i, row in results_df.head(5).iterrows():
    print(f"  {row['MI_Technique']}: F1={row['F1-Score']:.3f}")

print("\nNext steps:")
print("  1. Use the fine-tuned model to predict MI techniques on new data")
print("  2. Analyze embeddings for semantic similarity")
print("  3. Build sequential models using embeddings + dialogue context")
print("  4. Experiment with different BERT variants (RoBERTa, ALBERT, etc.)")

print("="*70)
