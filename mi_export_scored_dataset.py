# =============================================================================
#  SCRIPT 1 — mi_export_scored_dataset.py
#  Run this AFTER training in RoBERTa_Model.ipynb (after Cell 34)
#  Scores every record in your full dataset and saves mi_scored_dataset.csv
#  That CSV is the knowledge base fed into Script 2 (mi_gpt_classifier.py)
# =============================================================================

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# =============================================================================
#  CONFIG — must match your RoBERTa_Model.ipynb CONFIG exactly
# =============================================================================
CONFIG = {
    'data_path':    'MI Dataset English Filtered Bucketed.csv',
    'label_column': 'MITI Code',
    'text_column':  'text',
    'max_length':   128,
    'batch_size':   32,
    'model_path':   './roberta_model_results/final_model',   # saved by Cell 34
    'output_path':  'mi_scored_dataset.csv',
}

# =============================================================================
#  DATASET — same MIDataset class from your notebook
# =============================================================================
class MIDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# =============================================================================
#  MAIN
# =============================================================================
def main():
    print('=' * 70)
    print('MI SCORED DATASET EXPORT')
    print('=' * 70)

    # ── 1. Load device ────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── 2. Load full dataset ──────────────────────────────────────────────────
    print(f'\nLoading dataset: {CONFIG["data_path"]}')
    df = pd.read_csv(CONFIG['data_path'])
    print(f'Total records: {len(df):,}')
    print(f'Columns: {df.columns.tolist()}')

    # ── 3. Build label maps ───────────────────────────────────────────────────
    label_list = sorted(df[CONFIG['label_column']].unique())
    label2id   = {label: idx for idx, label in enumerate(label_list)}
    id2label   = {idx: label for label, idx in label2id.items()}
    print(f'\nClasses: {label_list}')

    # ── 4. Load trained model + tokenizer ────────────────────────────────────
    print(f'\nLoading model from: {CONFIG["model_path"]}')
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_path'])
    model     = AutoModelForSequenceClassification.from_pretrained(CONFIG['model_path'])
    model.to(device)
    model.eval()
    print(f'Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # ── 5. Tokenize full dataset ──────────────────────────────────────────────
    print('\nTokenizing full dataset...')
    texts  = df[CONFIG['text_column']].fillna('').tolist()
    labels = df[CONFIG['label_column']].map(label2id).tolist()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=CONFIG['max_length'],
        return_tensors='pt'
    )

    dataset    = MIDataset(encodings, labels)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    print(f'Tokenization complete. Batches: {len(dataloader)}')

    # ── 6. Run inference on every record ─────────────────────────────────────
    print('\nScoring all records...')
    all_probs       = []
    all_pred_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            all_probs.extend(probs)
            all_pred_labels.extend([id2label[p.argmax()] for p in probs])

            if (batch_idx + 1) % 50 == 0:
                print(f'  Processed {(batch_idx + 1) * CONFIG["batch_size"]:,} / {len(texts):,} records...')

    print(f'Scoring complete. {len(all_pred_labels):,} records scored.')

    # ── 7. Build output dataframe ─────────────────────────────────────────────
    print('\nBuilding output dataframe...')
    scored_rows = []
    for i, (prob_row, pred_label) in enumerate(zip(all_probs, all_pred_labels)):
        scores = {id2label[j]: round(float(p), 4) for j, p in enumerate(prob_row)}
        scored_rows.append({
            'text':                   texts[i],
            'true_label':             id2label[labels[i]],
            'predicted_label':        pred_label,
            'correct':                pred_label == id2label[labels[i]],
            'score_mi_adherent':      scores.get('MI Adherent',     0.0),
            'score_mi_non_adherent':  scores.get('MI Non-Adherent', 0.0),
            'score_other':            scores.get('Other',           0.0),
            'confidence':             round(float(max(prob_row)), 4),
        })

    output_df = pd.DataFrame(scored_rows)

    # ── 8. Print summary stats ────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('EXPORT SUMMARY')
    print('=' * 70)
    print(f'Total records exported: {len(output_df):,}')
    print(f'Overall accuracy:       {output_df["correct"].mean():.4f}')
    print(f'\nPredicted label distribution:')
    print(output_df['predicted_label'].value_counts().to_string())
    print(f'\nAverage confidence score: {output_df["confidence"].mean():.4f}')

    # ── 9. Save CSV ───────────────────────────────────────────────────────────
    output_df.to_csv(CONFIG['output_path'], index=False)
    print(f'\nSaved to: {CONFIG["output_path"]}')
    print('This file is the knowledge base for mi_gpt_classifier.py')
    print('=' * 70)

    return output_df


if __name__ == '__main__':
    df = main()
    print(df.head())
