import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer
import pickle
import plotly.express as px

# Page configuration
st.set_page_config(page_title="RNA Binding Predictor", layout="wide")
st.title("RNA-Protein Binding Prediction Tool")

# Load your data
# In a real implementation, you would load your actual model and data
df = pd.read_csv("merged_rna_data.csv")

# Simple sequence analysis section
st.header("RNA Sequence Binding Predictor")
sequence_input = st.text_area("Enter RNA sequence:", 
                             height=100,
                             placeholder="GAAGAGAUAAUCUGAAACAACAGUAUAUGACUCAAACUCUCC...")

if st.button("Predict Binding"):
    # In a real implementation, this would use your actual model
    # For demonstration purposes, we'll simulate a prediction
    
    # Simulate prediction result
    prediction = -7200  # Example value
    
    # Determine binding strength based on score
    if prediction > -6900:
        binding_strength = "Weak"
    elif prediction > -7150:
        binding_strength = "Moderate"
    else:
        binding_strength = "Strong"
    
    # Display prediction result
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Predicted Binding Score", f"{prediction:.2f}")
        st.metric("Binding Strength", binding_strength)
    
    with col2:
        # Display key sequence features
        st.subheader("Sequence Features")
        
        # Calculate nucleotide composition (would be from your model in real implementation)
        length = len(sequence_input)
        a_count = sequence_input.count('A')
        u_count = sequence_input.count('U')
        g_count = sequence_input.count('G')
        c_count = sequence_input.count('C')
        
        a_percent = (a_count / length) * 100
        u_percent = (u_count / length) * 100
        g_percent = (g_count / length) * 100
        c_percent = (c_count / length) * 100
        gc_content = (g_count + c_count) / length * 100
        
        st.write(f"Length: {length} nucleotides")
        st.write(f"GC Content: {gc_content:.1f}%")
        st.write(f"Cytosine Content: {c_percent:.1f}%")
    
    # Display binding insights
    st.subheader("Binding Insights")
    
    # These would be dynamically generated based on your model's analysis
    if c_percent > 25:
        st.write("- High cytosine content contributes to stronger binding")
    
    if gc_content > 50:
        st.write("- High GC content enhances structural stability and binding")
    
    # Display nucleotide composition chart
    st.subheader("Nucleotide Composition")
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([a_percent, u_percent, g_percent, c_percent], 
           labels=['A', 'U', 'G', 'C'], 
           autopct='%1.1f%%',
           colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    ax.set_title("Nucleotide Distribution")
    st.pyplot(fig)

# Dataset overview section
st.header("Dataset Overview")

# Display sample of the dataset
st.subheader("Sample RNA Sequences")
if 'RNA_Name' in df.columns and 'Score' in df.columns:
    st.dataframe(df[['RNA_Name', 'Score']].head(10))

# Distribution of binding scores
st.subheader("Distribution of Binding Scores")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df['Score'], kde=True, ax=ax)
ax.set_xlabel("Binding Score")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Key findings section
st.header("Key Binding Factors")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Factors That Enhance Binding")
    st.write("- Higher cytosine content (> 25%)")
    st.write("- Higher GC content (> 50%)")
    st.write("- Beneficial motifs: 'AAGAGA', 'AGCCUG', 'AGAAAG'")
    st.write("- A-rich regions like 'AAAAAA'")

with col2:
    st.subheader("Factors That Weaken Binding")
    st.write("- UG/GU-rich repetitive motifs")
    st.write("- Low cytosine content (< 18%)")
    st.write("- High UG/GU dinucleotide frequency (> 12%)")
    st.write("- G nucleotides at positions 2, 6, 9, and 19")

# Top motifs visualization
st.subheader("Common Motifs in Strong Binding Sequences")

# Sample data for motifs
motifs = [
    ('AAAAAA', 19),
    ('AGAGAA', 18),
    ('UUUUUU', 18),
    ('AAGAAA', 16),
    ('GCCUGG', 16),
    ('CAGCUG', 16),
    ('AGAAAG', 16),
    ('CUGCAG', 15),
    ('AGCCUG', 15),
    ('CAGCAG', 15)
]

motif_df = pd.DataFrame(motifs, columns=['Motif', 'Frequency'])

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Motif', y='Frequency', data=motif_df, ax=ax)
ax.set_title("Frequent Motifs in High-Binding Sequences")
ax.set_ylabel("Frequency")
ax.set_xlabel("Motif")
plt.xticks(rotation=45)
st.pyplot(fig)

# Model performance section
st.header("Model Performance")

# Display calibration results
st.subheader("Effect of Calibration on Prediction Errors")

# Sample data for calibration effect
calibration_data = {
    'RNA_Name': ['lnc-SSTR4-24', 'lnc-KHSRP-12', 'lnc-BPY2C-22', 'lnc-BRF2-171', 'lnc-ALG1L-41'],
    'Original_Error': [588.11, 515.30, 864.55, 564.00, 508.00],
    'Calibrated_Error': [188.11, 115.30, 464.55, 164.00, 108.00]
}

calib_df = pd.DataFrame(calibration_data)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(calib_df['RNA_Name']))
width = 0.35

ax.bar(x - width/2, calib_df['Original_Error'], width, label='Original Error')
ax.bar(x + width/2, calib_df['Calibrated_Error'], width, label='Calibrated Error')

ax.set_xlabel('RNA Sequence')
ax.set_ylabel('Prediction Error')
ax.set_title('Error Reduction with Calibration')
ax.set_xticks(x)
ax.set_xticklabels(calib_df['RNA_Name'], rotation=45)
ax.legend()

fig.tight_layout()
st.pyplot(fig)
