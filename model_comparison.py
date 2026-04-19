"""
Model Comparison for MI Classification (4 Classes)
Tests: BERT, RoBERTa, DeBERTa with class weights

Run time: ~1-2 hours for all 3 models
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MODEL COMPARISON FOR MI CLASSIFICATION (4 CLASSES)")
print("="*80)

# ==========================================
# CONFIGURATION
# ==========================================

CONFIG = {
    'data_path': '/home/claude/processed_mi_dataset.csv',
    'label_column': 'bucketed_label',  # Your 4-class labels
    'text_column': 'text',
    'test_size': 0.2,
    'random_state': 42,
    'max_length': 128,
    'batch_size': 16,
    'num_epochs': 5,  # Increased for better learning
    'learning_rate': 1e-5,  # Lower for fine-tuning
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'output_dir': '/home/claude/model_comparison_results'
}

# Models to compare
MODELS = {
    'BERT': 'bert-base-uncased',
    'RoBERTa': 'roberta-base',
    'DeBERTa': 'microsoft/deberta-v3-base'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# ==========================================
# LOAD AND PREPARE DATA
# ==========================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

df = pd.read_csv(CONFIG['data_path'])
print(f"Total samples: {len(df):,}")

# Check for the bucketed label column
if CONFIG['label_column'] not in df.columns:
    print(f"\nError: Column '{CONFIG['label_column']}' not found!")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

# Get unique labels
label_list = sorted(df[CONFIG['label_column']].unique())
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

print(f"\nClasses ({len(label_list)}):")
for label in label_list:
    count = (df[CONFIG['label_column']] == label).sum()
    pct = (count / len(df)) * 100
    print(f"  {label:20s}: {count:5,} ({pct:5.1f}%)")

# Encode labels
df['label_id'] = df[CONFIG['label_column']].map(label2id)

# Train/val split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df[CONFIG['text_column']].tolist(),
    df['label_id'].tolist(),
    test_size=CONFIG['test_size'],
    stratify=df['label_id'],
    random_state=CONFIG['random_state']
)

print(f"\nTrain samples: {len(train_texts):,}")
print(f"Val samples: {len(val_texts):,}")

# Compute class weights for imbalanced data
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

print(f"\nClass weights (for imbalance):")
for label, weight in zip(label_list, class_weights):
    print(f"  {label:20s}: {weight:.3f}")

# ==========================================
# CUSTOM TRAINER WITH CLASS WEIGHTS
# ==========================================

class WeightedLossTrainer(Trainer):
    """Custom trainer that uses class weights in loss function"""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Use weighted cross-entropy loss
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(self.class_weights, dtype=torch.float).to(model.device)
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# ==========================================
# DATASET CLASS
# ==========================================

class MIDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for MI classification"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# ==========================================
# TRAIN AND EVALUATE FUNCTION
# ==========================================

def train_and_evaluate_model(model_name, model_path):
    """Train a single model and return results"""
    
    print("\n" + "="*80)
    print(f"TRAINING: {model_name}")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = datetime.now()
    
    try:
        # Load tokenizer and model
        print("\nLoading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        
        # Tokenize
        print("Tokenizing data...")
        train_encodings = tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=CONFIG['max_length'],
            return_tensors='pt'
        )
        
        val_encodings = tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=CONFIG['max_length'],
            return_tensors='pt'
        )
        
        # Create datasets
        train_dataset = MIDataset(train_encodings, train_labels)
        val_dataset = MIDataset(val_encodings, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{CONFIG['output_dir']}/{model_name.replace('/', '_')}",
            num_train_epochs=CONFIG['num_epochs'],
            per_device_train_batch_size=CONFIG['batch_size'],
            per_device_eval_batch_size=CONFIG['batch_size'] * 2,
            learning_rate=CONFIG['learning_rate'],
            warmup_ratio=CONFIG['warmup_ratio'],
            weight_decay=CONFIG['weight_decay'],
            logging_dir=f"{CONFIG['output_dir']}/{model_name}/logs",
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            save_total_limit=2
        )
        
        # Create trainer with class weights
        print("\nInitializing trainer with class weights...")
        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            class_weights=class_weights,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        print("\nTraining started...")
        train_result = trainer.train()
        print(f"✓ Training complete!")
        print(f"  Final training loss: {train_result.metrics['train_loss']:.4f}")
        
        # Evaluate
        print("\nEvaluating on validation set...")
        predictions = trainer.predict(val_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        pred_probs = torch.nn.functional.softmax(
            torch.tensor(predictions.predictions), 
            dim=-1
        ).numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(val_labels, pred_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            val_labels,
            pred_labels,
            labels=range(len(label_list)),
            average=None
        )
        
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            val_labels,
            pred_labels,
            average='weighted'
        )
        
        # Get per-class metrics
        per_class_metrics = []
        for i, label in enumerate(label_list):
            per_class_metrics.append({
                'class': label,
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            })
        
        # Confusion matrix
        cm = confusion_matrix(val_labels, pred_labels)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✓ Evaluation complete!")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")
        print(f"  Training time: {training_time:.1f} seconds")
        
        # Save model
        model_save_path = f"{CONFIG['output_dir']}/{model_name}_final"
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"✓ Model saved to: {model_save_path}")
        
        return {
            'model_name': model_name,
            'model_path': model_path,
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'training_loss': train_result.metrics['train_loss'],
            'training_time_seconds': training_time,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm,
            'predictions': pred_labels,
            'probabilities': pred_probs,
            'true_labels': val_labels
        }
        
    except Exception as e:
        print(f"\n✗ Error training {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ==========================================
# RUN COMPARISONS
# ==========================================

print("\n" + "="*80)
print("STARTING MODEL COMPARISON")
print("="*80)

results = []
for model_name, model_path in MODELS.items():
    result = train_and_evaluate_model(model_name, model_path)
    if result is not None:
        results.append(result)

# ==========================================
# COMPARATIVE ANALYSIS
# ==========================================

print("\n" + "="*80)
print("COMPARATIVE RESULTS")
print("="*80)

# Create comparison table
comparison_df = pd.DataFrame([
    {
        'Model': r['model_name'],
        'Accuracy': r['accuracy'],
        'Macro P': r['macro_precision'],
        'Macro R': r['macro_recall'],
        'Macro F1': r['macro_f1'],
        'Weighted F1': r['weighted_f1'],
        'Train Loss': r['training_loss'],
        'Time (s)': r['training_time_seconds']
    }
    for r in results
]).sort_values('macro_f1', ascending=False)

print("\n" + comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv(f"{CONFIG['output_dir']}/model_comparison_summary.csv", index=False)
print(f"\n✓ Summary saved to: {CONFIG['output_dir']}/model_comparison_summary.csv")

# ==========================================
# DETAILED PER-CLASS COMPARISON
# ==========================================

print("\n" + "="*80)
print("PER-CLASS PERFORMANCE COMPARISON")
print("="*80)

for class_name in label_list:
    print(f"\n{class_name}:")
    print("-" * 60)
    print(f"{'Model':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    
    for result in results:
        class_metrics = next(
            (m for m in result['per_class_metrics'] if m['class'] == class_name),
            None
        )
        if class_metrics:
            print(
                f"{result['model_name']:<15} "
                f"{class_metrics['precision']:<12.3f} "
                f"{class_metrics['recall']:<12.3f} "
                f"{class_metrics['f1']:<12.3f}"
            )

# ==========================================
# VISUALIZATIONS
# ==========================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

import os
os.makedirs(f"{CONFIG['output_dir']}/visualizations", exist_ok=True)

# 1. Overall metrics comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['accuracy', 'macro_f1', 'weighted_f1']
metric_names = ['Accuracy', 'Macro F1', 'Weighted F1']

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx]
    models = [r['model_name'] for r in results]
    values = [r[metric] for r in results]
    
    bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel(name, fontsize=12)
    ax.set_title(f'{name} Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/visualizations/overall_comparison.png", dpi=300)
plt.close()
print("✓ Overall comparison plot saved")

# 2. Per-class F1 comparison
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(label_list))
width = 0.25

for idx, result in enumerate(results):
    f1_scores = [m['f1'] for m in result['per_class_metrics']]
    ax.bar(x + idx * width, f1_scores, width, label=result['model_name'])

ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Per-Class F1 Score Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(label_list, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/visualizations/per_class_f1.png", dpi=300)
plt.close()
print("✓ Per-class F1 plot saved")

# 3. Confusion matrices
for result in results:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cm = result['confusion_matrix']
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_list,
        yticklabels=label_list,
        ax=ax
    )
    
    ax.set_title(f"Confusion Matrix - {result['model_name']}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(
        f"{CONFIG['output_dir']}/visualizations/confusion_matrix_{result['model_name']}.png",
        dpi=300
    )
    plt.close()
    print(f"✓ Confusion matrix for {result['model_name']} saved")

# 4. Training time comparison
fig, ax = plt.subplots(figsize=(8, 6))

models = [r['model_name'] for r in results]
times = [r['training_time_seconds'] / 60 for r in results]  # Convert to minutes

bars = ax.barh(models, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_xlabel('Training Time (minutes)', fontsize=12)
ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')

for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.1f} min',
            ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/visualizations/training_time.png", dpi=300)
plt.close()
print("✓ Training time plot saved")

# ==========================================
# DETAILED CLASSIFICATION REPORTS
# ==========================================

print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORTS")
print("="*80)

for result in results:
    print(f"\n{result['model_name']}:")
    print("="*70)
    report = classification_report(
        result['true_labels'],
        result['predictions'],
        target_names=label_list,
        digits=3
    )
    print(report)
    
    # Save to file
    with open(f"{CONFIG['output_dir']}/{result['model_name']}_classification_report.txt", 'w') as f:
        f.write(f"Classification Report - {result['model_name']}\n")
        f.write("="*70 + "\n\n")
        f.write(report)

# ==========================================
# SAVE DETAILED RESULTS
# ==========================================

print("\n" + "="*80)
print("SAVING DETAILED RESULTS")
print("="*80)

# Save detailed results as JSON (excluding numpy arrays)
detailed_results = []
for result in results:
    detailed_results.append({
        'model_name': result['model_name'],
        'model_path': result['model_path'],
        'accuracy': float(result['accuracy']),
        'macro_f1': float(result['macro_f1']),
        'weighted_f1': float(result['weighted_f1']),
        'training_time_seconds': float(result['training_time_seconds']),
        'per_class_metrics': result['per_class_metrics']
    })

with open(f"{CONFIG['output_dir']}/detailed_results.json", 'w') as f:
    json.dump(detailed_results, f, indent=2)

print(f"✓ Detailed results saved to: {CONFIG['output_dir']}/detailed_results.json")

# ==========================================
# RECOMMENDATIONS
# ==========================================

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

best_model = max(results, key=lambda x: x['macro_f1'])
fastest_model = min(results, key=lambda x: x['training_time_seconds'])

print(f"\n🏆 Best Overall Performance:")
print(f"   Model: {best_model['model_name']}")
print(f"   Macro F1: {best_model['macro_f1']:.4f}")
print(f"   Accuracy: {best_model['accuracy']:.4f}")

print(f"\n⚡ Fastest Training:")
print(f"   Model: {fastest_model['model_name']}")
print(f"   Time: {fastest_model['training_time_seconds']/60:.1f} minutes")
print(f"   Macro F1: {fastest_model['macro_f1']:.4f}")

# Find weakest class across all models
all_class_f1s = {}
for class_name in label_list:
    f1_scores = []
    for result in results:
        class_metric = next(m for m in result['per_class_metrics'] if m['class'] == class_name)
        f1_scores.append(class_metric['f1'])
    all_class_f1s[class_name] = np.mean(f1_scores)

weakest_class = min(all_class_f1s, key=all_class_f1s.get)
print(f"\n⚠️  Weakest Class (across all models):")
print(f"   Class: {weakest_class}")
print(f"   Average F1: {all_class_f1s[weakest_class]:.4f}")
print(f"   Suggestion: Consider data augmentation or collecting more samples")

print("\n" + "="*80)
print("COMPARISON COMPLETE!")
print("="*80)
print(f"\nAll results saved to: {CONFIG['output_dir']}/")
print(f"Best model saved to: {CONFIG['output_dir']}/{best_model['model_name']}_final/")
print("\nNext steps:")
print("  1. Review visualizations in: {}/visualizations/".format(CONFIG['output_dir']))
print("  2. Check detailed classification reports")
print("  3. Use best model for your GPT integration")
print("="*80)