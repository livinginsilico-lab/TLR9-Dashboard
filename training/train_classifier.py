import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
from tqdm import tqdm
import json

class RNADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        label = float(self.labels[idx])
        
        # Tokenize the sequence
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def load_and_prepare_data(data_path, target_column, test_size=0.2, random_state=42):
    """Load RNA data and prepare for training."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Filter out rows with missing target values
    df = df.dropna(subset=[target_column])
    
    print(f"Total sequences with {target_column}: {len(df)}")
    
    # Extract sequences and labels
    sequences = df['RNA_Sequence'].values
    labels = df[target_column].values
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=test_size, random_state=random_state
    )
    
    # Scale the labels
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler.transform(y_val.reshape(-1, 1)).flatten()
    
    return X_train, X_val, y_train_scaled, y_val_scaled, scaler

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    
    # Calculate MSE and RMSE
    mse = np.mean((predictions - labels) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate MAE
    mae = np.mean(np.abs(predictions - labels))
    
    # Calculate R-squared
    ss_res = np.sum((labels - predictions) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def train_model(args):
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare data
    X_train, X_val, y_train, y_val, scaler = load_and_prepare_data(
        args.data_path, 
        args.target_column,
        args.test_size
    )
    
    # Save the scaler
    scaler_path = os.path.join(args.output_dir, f'{args.target_column.lower()}_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Create datasets
    train_dataset = RNADataset(X_train, y_train, tokenizer, args.max_length)
    val_dataset = RNADataset(X_val, y_val, tokenizer, args.max_length)
    
    # Initialize model
    print("Initializing model...")
    model = GPT2ForSequenceClassification.from_pretrained(
        'gpt2',
        num_labels=1,
        problem_type="regression"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to="none"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    final_model_path = os.path.join(args.output_dir, 'final_model')
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Evaluate on validation set
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    
    # Save evaluation results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    print("\nTraining completed!")
    print(f"Model saved to: {final_model_path}")
    print(f"Evaluation results: {eval_results}")
    
    return trainer, model

def main():
    parser = argparse.ArgumentParser(description='Train RNA binding affinity prediction model')
    parser.add_argument('--data_path', type=str, default='master_rna_data.csv',
                        help='Path to the RNA data CSV file')
    parser.add_argument('--target_column', type=str, required=True,
                        choices=['Score', 'RMSD'],
                        help='Target column to predict (Score or RMSD)')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer',
                        help='Path to the tokenizer directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the trained model')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Validation set size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    train_model(args)

if __name__ == "__main__":
    main()
