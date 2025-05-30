import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def analyze_rna_data(data_path='master_rna_data.csv', output_dir='data_analysis'):
    """Analyze RNA data and generate statistics."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Total sequences: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Sequences with Score: {df['Score'].notna().sum()}")
    print(f"Sequences with RMSD: {df['RMSD'].notna().sum()}")
    
    # Score analysis
    if 'Score' in df.columns:
        scores = df['Score'].dropna()
        print(f"\nScore Statistics:")
        print(f"  Mean: {scores.mean():.2f}")
        print(f"  Std: {scores.std():.2f}")
        print(f"  Min: {scores.min():.2f}")
        print(f"  Max: {scores.max():.2f}")
        print(f"  25%: {scores.quantile(0.25):.2f}")
        print(f"  50%: {scores.quantile(0.50):.2f}")
        print(f"  75%: {scores.quantile(0.75):.2f}")
        
        # Plot score distribution
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(x=-6633.01, color='red', linestyle='--', label='ANOVA Threshold (-6633.01)')
        plt.xlabel('Binding Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of RNA Binding Scores')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'score_distribution.png'))
        plt.close()
        
        # Count strong binders
        strong_binders = (scores < -6633.01).sum()
        print(f"\nStrong binders (Score < -6633.01): {strong_binders} ({strong_binders/len(scores)*100:.1f}%)")
    
    # RMSD analysis
    if 'RMSD' in df.columns:
        rmsd = df['RMSD'].dropna()
        print(f"\nRMSD Statistics:")
        print(f"  Mean: {rmsd.mean():.2f}")
        print(f"  Std: {rmsd.std():.2f}")
        print(f"  Min: {rmsd.min():.2f}")
        print(f"  Max: {rmsd.max():.2f}")
        
        # Plot RMSD distribution
        plt.figure(figsize=(10, 6))
        plt.hist(rmsd, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('RMSD')
        plt.ylabel('Frequency')
        plt.title('Distribution of RMSD Values')
        plt.savefig(os.path.join(output_dir, 'rmsd_distribution.png'))
        plt.close()
    
    # Sequence length analysis
    df['Sequence_Length'] = df['RNA_Sequence'].apply(len)
    print(f"\nSequence Length Statistics:")
    print(f"  Mean: {df['Sequence_Length'].mean():.0f}")
    print(f"  Std: {df['Sequence_Length'].std():.0f}")
    print(f"  Min: {df['Sequence_Length'].min()}")
    print(f"  Max: {df['Sequence_Length'].max()}")
    
    # Nucleotide composition
    print("\n=== Nucleotide Composition Analysis ===")
    nucleotides = {'A': 0, 'U': 0, 'G': 0, 'C': 0}
    total_length = 0
    
    for seq in df['RNA_Sequence']:
        for nuc in seq:
            if nuc in nucleotides:
                nucleotides[nuc] += 1
                total_length += 1
    
    print("Average nucleotide composition:")
    for nuc, count in nucleotides.items():
        print(f"  {nuc}: {count/total_length*100:.1f}%")
    
    # GC content analysis
    df['GC_Content'] = df['RNA_Sequence'].apply(
        lambda x: (x.count('G') + x.count('C')) / len(x) * 100
    )
    
    print(f"\nGC Content Statistics:")
    print(f"  Mean: {df['GC_Content'].mean():.1f}%")
    print(f"  Std: {df['GC_Content'].std():.1f}%")
    
    # Save processed data
    summary_path = os.path.join(output_dir, 'data_summary.csv')
    df.to_csv(summary_path, index=False)
    print(f"\nData summary saved to: {summary_path}")
    
    return df

def prepare_training_splits(data_path='master_rna_data.csv', output_dir='training_data'):
    """Prepare training and validation splits."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    
    # For Score prediction
    score_df = df[df['Score'].notna()][['RNA_Sequence', 'Score', 'RNA_Name']]
    score_train_path = os.path.join(output_dir, 'score_train.csv')
    score_df.to_csv(score_train_path, index=False)
    print(f"Score training data saved to: {score_train_path}")
    
    # For RMSD prediction
    rmsd_df = df[df['RMSD'].notna()][['RNA_Sequence', 'RMSD', 'RNA_Name']]
    rmsd_train_path = os.path.join(output_dir, 'rmsd_train.csv')
    rmsd_df.to_csv(rmsd_train_path, index=False)
    print(f"RMSD training data saved to: {rmsd_train_path}")
    
    return score_train_path, rmsd_train_path

if __name__ == "__main__":
    # Analyze the data
    print("Analyzing RNA data...")
    df = analyze_rna_data()
    
    print("\n" + "="*50 + "\n")
    
    # Prepare training splits
    print("Preparing training data...")
    prepare_training_splits()
