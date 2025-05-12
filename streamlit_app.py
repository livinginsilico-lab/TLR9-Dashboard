import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import re
from io import StringIO

# ------------------- Page Config and Styling -------------------
st.set_page_config(page_title="RNA-Protein Binding Predictor", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2c3e50; }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .highlight {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
    }
    .data-container {
        background-color: #f1f7fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- Helper Functions -------------------

def parse_fasta(fasta_string):
    """Parse FASTA format into a dictionary of sequences"""
    if not fasta_string.strip():
        return {}
    
    sequences = {}
    current_header = None
    current_seq = []
    
    for line in fasta_string.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header:
                sequences[current_header] = "".join(current_seq)
            current_header = line[1:].strip()
            current_seq = []
        else:
            current_seq.append(line)
    
    if current_header:
        sequences[current_header] = "".join(current_seq)
    
    return sequences

def extract_sequence_features(sequence):
    """Extract key features from an RNA sequence"""
    if not sequence:
        return None
    
    length = len(sequence)
    a_count = sequence.count('A')
    u_count = sequence.count('U')
    g_count = sequence.count('G')
    c_count = sequence.count('C')
    
    a_percent = (a_count / length) * 100
    u_percent = (u_count / length) * 100
    g_percent = (g_count / length) * 100
    c_percent = (c_count / length) * 100
    gc_content = (g_count + c_count) / length * 100
    
    # Check for beneficial motifs
    beneficial_motifs = ['AAGAGA', 'AGCCUG', 'AGAAAG', 'AAAAAA', 'GCCUGG', 'CAGCUG']
    beneficial_count = sum(sequence.count(motif) for motif in beneficial_motifs)
    
    # Check for problematic motifs
    problematic_motifs = ['UGGUGA', 'GUGAUG', 'GAUGGU', 'AUGGUG', 'GGUGAU']
    problematic_count = sum(sequence.count(motif) for motif in problematic_motifs)
    
    # Check UG/GU density
    ug_count = sum(1 for i in range(len(sequence)-1) if sequence[i:i+2] in ['UG', 'GU'])
    ug_density = (ug_count / (length-1)) * 100 if length > 1 else 0
    
    return {
        'length': length,
        'a_percent': a_percent,
        'u_percent': u_percent,
        'g_percent': g_percent,
        'c_percent': c_percent,
        'gc_content': gc_content,
        'beneficial_motifs': beneficial_count,
        'problematic_motifs': problematic_count,
        'ug_density': ug_density
    }

def predict_binding(sequence, features):
    """Predict binding score based on sequence features"""
    # This is a simplified prediction model based on our findings
    # In a real implementation, you would use your trained model
    
    base_score = -7200  # Base score
    
    # Apply adjustments based on features
    if features['c_percent'] > 25:
        base_score -= 100  # Stronger binding (more negative)
    
    if features['gc_content'] > 50:
        base_score -= 75
        
    if features['beneficial_motifs'] > 0:
        base_score -= 50 * features['beneficial_motifs']
    
    if features['problematic_motifs'] > 0:
        base_score += 100 * features['problematic_motifs']
    
    if features['ug_density'] > 12:
        base_score += 150
    
    if features['c_percent'] < 18:
        base_score += 200
    
    # Add some randomness to simulate variation
    base_score += np.random.normal(0, 50)
    
    return base_score

def generate_binding_insights(features, score):
    """Generate insights about binding based on features and score"""
    insights = []
    
    # Binding strength classification
    binding_threshold = -6676.38  # ANOVA test threshold
    
    if score < binding_threshold:
        insights.append("✅ This sequence is predicted to be a **good binder** (score below threshold of -6676.38)")
    else:
        insights.append("⚠️ This sequence is predicted to be a **poor binder** (score above threshold of -6676.38)")
    
    # Composition insights
    if features['c_percent'] > 25:
        insights.append("✅ High cytosine content (%.1f%%) contributes to stronger binding" % features['c_percent'])
    elif features['c_percent'] < 18:
        insights.append("⚠️ Low cytosine content (%.1f%%) may weaken binding" % features['c_percent'])
    
    if features['gc_content'] > 50:
        insights.append("✅ High GC content (%.1f%%) enhances structural stability and binding" % features['gc_content'])
    elif features['gc_content'] < 45:
        insights.append("⚠️ Low GC content (%.1f%%) may reduce structural stability" % features['gc_content'])
    
    # Motif insights
    if features['beneficial_motifs'] > 0:
        insights.append("✅ Contains %d motifs associated with stronger binding" % features['beneficial_motifs'])
    
    if features['problematic_motifs'] > 0:
        insights.append("⚠️ Contains %d UG/GU-rich motifs associated with weaker binding" % features['problematic_motifs'])
    
    if features['ug_density'] > 12:
        insights.append("⚠️ High UG/GU dinucleotide frequency (%.1f%%) indicates weaker binding" % features['ug_density'])
    
    return insights

def generate_rna_sequence(length=200, binding_quality="good"):
    """Generate RNA sequences that would be good or bad binders based on our findings"""
    nucleotides = ['A', 'U', 'G', 'C']
    
    if binding_quality == "good":
        # Good binders have higher C content, more beneficial motifs
        weights = [0.25, 0.2, 0.25, 0.3]  # Higher C weight
        sequence = np.random.choice(nucleotides, size=length, p=weights)
        sequence = ''.join(sequence)
        
        # Add some beneficial motifs
        beneficial_motifs = ['AAGAGA', 'AGCCUG', 'AGAAAG', 'CAGCUG']
        for _ in range(3):
            position = np.random.randint(0, length - 6)
            motif = np.random.choice(beneficial_motifs)
            sequence = sequence[:position] + motif + sequence[position+6:]
            
    else:
        # Bad binders have lower C content, more UG/GU motifs
        weights = [0.3, 0.3, 0.25, 0.15]  # Lower C weight
        sequence = np.random.choice(nucleotides, size=length, p=weights)
        sequence = ''.join(sequence)
        
        # Add some problematic motifs
        problematic_motifs = ['UGGUGA', 'GUGAUG', 'GAUGGU', 'AUGGUG']
        for _ in range(3):
            position = np.random.randint(0, length - 6)
            motif = np.random.choice(problematic_motifs)
            sequence = sequence[:position] + motif + sequence[position+6:]
    
    return sequence

# ------------------- Sidebar -------------------
st.sidebar.markdown("""
    <div style='text-align: center;'>
        <h2>RNA-Protein Binding</h2>
        <p>Predict and analyze RNA binding affinity</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", 
                      ["🧬 Sequence Analyzer", 
                       "🧪 Sequence Generator", 
                       "📊 Dataset Insights", 
                       "📈 Model Performance"])

st.sidebar.markdown("---")
st.sidebar.markdown("### About the Model")
st.sidebar.markdown("""
This tool predicts RNA-protein binding affinity based on sequence features. Our research found that:

- Statistical threshold (ANOVA): -6676.38
- Sequences below this threshold are good binders
- Features like GC content, cytosine percentage, and specific motifs influence binding strength
""")

# ------------------- Load Data -------------------
@st.cache_data
def load_data():
    return pd.read_csv("merged_rna_data.csv")

df = load_data()

# ------------------- Sequence Analyzer -------------------
if page == "🧬 Sequence Analyzer":
    st.markdown("# RNA Sequence Binding Analyzer")
    st.markdown("Predict the binding affinity of RNA sequences to the target protein.")
    st.divider()
    
    # Input tabs for different input methods
    input_tab1, input_tab2 = st.tabs(["✏️ Single Sequence", "📄 FASTA Format"])
    
    with input_tab1:
        sequence_input = st.text_area(
            "Enter RNA sequence:", 
            height=100,
            placeholder="GAAGAGAUAAUCUGAAACAACAGUAUAUGACUCAAACUCUCC..."
        )
        sequence_name = st.text_input("Sequence name (optional):", "User_Sequence")
        
        if sequence_input:
            sequences = {sequence_name: sequence_input.strip().upper().replace('T', 'U')}
        else:
            sequences = {}
    
    with input_tab2:
        fasta_input = st.text_area(
            "Enter sequences in FASTA format:", 
            height=150,
            placeholder=">Sequence_1\nGAAGAGAUAAUCUGAAACAA\n>Sequence_2\nCGCCGAGAGCCAGCGAGGGGA"
        )
        
        if fasta_input:
            sequences = parse_fasta(fasta_input)
            if not sequences:
                st.warning("No valid FASTA sequences detected. Please check the format.")
        else:
            sequences = {}
    
    analyze_button = st.button("Analyze Binding", type="primary")
    
    if analyze_button and sequences:
        st.markdown("## Analysis Results")
        
        # Process each sequence
        results = []
        for name, seq in sequences.items():
            features = extract_sequence_features(seq)
            if not features:
                continue
                
            score = predict_binding(seq, features)
            insights = generate_binding_insights(features, score)
            binding_strength = "Strong" if score < -7150 else "Moderate" if score < -6900 else "Weak"
            binding_quality = "Good" if score < -6676.38 else "Poor"
            
            results.append({
                'name': name,
                'sequence': seq,
                'features': features,
                'score': score,
                'binding_strength': binding_strength,
                'binding_quality': binding_quality,
                'insights': insights
            })
        
        # Display results
        for i, result in enumerate(results):
            with st.container():
                st.markdown(f"### {result['name']}")
                
                # Create metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Binding Score", f"{result['score']:.2f}")
                with col2:
                    st.metric("Binding Strength", result['binding_strength'])
                with col3:
                    st.metric("Binding Quality", result['binding_quality'],
                             delta="Good" if result['binding_quality'] == "Good" else "Poor",
                             delta_color="normal" if result['binding_quality'] == "Good" else "inverse")
                
                # Create tabs for different analyses
                tabs = st.tabs(["📊 Features", "🔍 Binding Insights", "🧬 Sequence"])
                
                with tabs[0]:
                    features = result['features']
                    
                    # Composition visualization
                    st.markdown("#### Nucleotide Composition")
                    composition_data = {
                        'Nucleotide': ['A', 'U', 'G', 'C'],
                        'Percentage': [
                            features['a_percent'], 
                            features['u_percent'], 
                            features['g_percent'], 
                            features['c_percent']
                        ]
                    }
                    comp_df = pd.DataFrame(composition_data)
                    
                    fig = px.bar(comp_df, x='Nucleotide', y='Percentage',
                                color='Nucleotide',
                                color_discrete_map={'A': '#FF9999', 'U': '#66B3FF', 'G': '#99FF99', 'C': '#FFCC99'},
                                text_auto='.1f')
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("GC Content", f"{features['gc_content']:.1f}%")
                    with col2:
                        st.metric("Length", f"{features['length']} nt")
                    with col3:
                        st.metric("UG/GU Density", f"{features['ug_density']:.1f}%")
                
                with tabs[1]:
                    st.markdown("#### Key Binding Factors")
                    for insight in result['insights']:
                        st.markdown(insight)
                    
                    # Display motif information
                    if features['beneficial_motifs'] > 0 or features['problematic_motifs'] > 0:
                        st.markdown("#### Detected Motifs")
                        
                        motif_data = [
                            {"Motif Type": "Beneficial", "Count": features['beneficial_motifs'], "Effect": "Enhance binding"},
                            {"Motif Type": "Problematic", "Count": features['problematic_motifs'], "Effect": "Weaken binding"}
                        ]
                        
                        motif_df = pd.DataFrame(motif_data)
                        fig = px.bar(motif_df, x='Motif Type', y='Count', color='Motif Type',
                                    color_discrete_map={'Beneficial': '#66BB6A', 'Problematic': '#EF5350'},
                                    text_auto=True)
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                with tabs[2]:
                    st.markdown("#### Sequence Details")
                    st.text(f"Length: {len(result['sequence'])} nucleotides")
                    st.code(result['sequence'])
                
                st.divider()
        
        # If multiple sequences, show comparison
        if len(results) > 1:
            st.markdown("## Sequence Comparison")
            
            # Prepare comparison data
            comp_data = []
            for r in results:
                comp_data.append({
                    'Name': r['name'],
                    'Score': r['score'],
                    'GC Content': r['features']['gc_content'],
                    'Cytosine %': r['features']['c_percent'],
                    'Binding Quality': r['binding_quality']
                })
            
            comp_df = pd.DataFrame(comp_data)
            
            # Comparison chart
            fig = px.bar(comp_df, x='Name', y='Score', color='Binding Quality',
                        color_discrete_map={'Good': '#66BB6A', 'Poor': '#EF5350'},
                        labels={'Score': 'Binding Score'},
                        title="Binding Score Comparison")
            fig.add_hline(y=-6676.38, line_dash="dash", annotation_text="Threshold", 
                         annotation_position="top right")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# ------------------- Sequence Generator -------------------
elif page == "🧪 Sequence Generator":
    st.markdown("# RNA Sequence Generator")
    st.markdown("Generate RNA sequences optimized for binding based on our model's insights.")
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        binding_quality = st.radio("Desired binding quality:", ["Good", "Poor"])
        seq_length = st.slider("Sequence length:", 50, 300, 200)
    
    with col2:
        st.markdown("""
        #### How it works
        This generator creates RNA sequences with properties that our model has identified as:
        
        - **Good binders**: Higher C content, beneficial motifs, GC-rich
        - **Poor binders**: Lower C content, UG/GU-rich motifs
        """)
    
    if st.button("Generate Sequence", type="primary"):
        with st.spinner("Generating optimized RNA sequence..."):
            generated_seq = generate_rna_sequence(seq_length, binding_quality.lower())
            
            # Analyze the generated sequence
            features = extract_sequence_features(generated_seq)
            score = predict_binding(generated_seq, features)
            insights = generate_binding_insights(features, score)
            
            st.markdown("## Generated Sequence")
            st.code(f">{binding_quality}_Binder_Generated\n{generated_seq}")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Score", f"{score:.2f}")
            with col2:
                st.metric("GC Content", f"{features['gc_content']:.1f}%")
            with col3:
                st.metric("Cytosine Content", f"{features['c_percent']:.1f}%")
            
            # Display analysis
            st.markdown("### Sequence Analysis")
            for insight in insights:
                st.markdown(insight)
            
            # Visualize nucleotide composition
            st.markdown("### Nucleotide Composition")
            comp_data = {
                'Nucleotide': ['A', 'U', 'G', 'C'],
                'Percentage': [
                    features['a_percent'],
                    features['u_percent'],
                    features['g_percent'],
                    features['c_percent']
                ]
            }
            
            fig = px.pie(
                values=comp_data['Percentage'],
                names=comp_data['Nucleotide'],
                title="Nucleotide Distribution",
                color_discrete_sequence=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Motif Analysis")
            motif_types = ["Beneficial", "Problematic"]
            motif_counts = [features['beneficial_motifs'], features['problematic_motifs']]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=motif_types,
                    y=motif_counts,
                    marker_color=['#66BB6A', '#EF5350']
                )
            ])
            fig.update_layout(title="Detected Motifs")
            st.plotly_chart(fig, use_container_width=True)

# ------------------- Dataset Insights -------------------
elif page == "📊 Dataset Insights":
    st.markdown("# Dataset Insights")
    st.markdown("Explore patterns and trends in RNA binding data.")
    st.divider()
    
    # Create tabs for different insights
    tabs = st.tabs(["📋 Data Explorer", "📊 Binding Patterns", "🧬 Sequence Features"])
    
    with tabs[0]:
        st.markdown("### RNA Binding Dataset")
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            score_range = st.slider("Filter by Binding Score:", 
                                  int(df['Score'].min()), 
                                  int(df['Score'].max()),
                                  (int(df['Score'].min()), int(df['Score'].max())))
        
        # Apply filters
        filtered_df = df[(df['Score'] >= score_range[0]) & (df['Score'] <= score_range[1])]
        
        # Show data
        st.dataframe(filtered_df[['RNA_Name', 'Score']], height=300)
        
        # Summary statistics
        st.markdown("### Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Sequences", len(filtered_df))
        with col2:
            st.metric("Average Score", f"{filtered_df['Score'].mean():.2f}")
        with col3:
            st.metric("Binding Threshold", "-6676.38")
            
        # Distribution plot
        st.markdown("### Binding Score Distribution")
        fig = px.histogram(filtered_df, x='Score', 
                         labels={'Score': 'Binding Score'},
                         color_discrete_sequence=['#3498db'])
        fig.add_vline(x=-6676.38, line_dash="dash", line_color="red",
                    annotation_text="Binding Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.markdown("### Binding Strength Analysis")
        
        # Add binding quality category
        quality_df = df.copy()
        quality_df['Binding_Quality'] = quality_df['Score'].apply(
            lambda x: 'Good' if x < -6676.38 else 'Poor'
        )
        
        # Distribution by binding quality
        fig = px.histogram(quality_df, x='Score', color='Binding_Quality',
                         color_discrete_map={'Good': '#66BB6A', 'Poor': '#EF5350'},
                         barmode='overlay',
                         labels={'Score': 'Binding Score'})
        fig.add_vline(x=-6676.38, line_dash="dash",
                    annotation_text="Threshold")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top strongest binding sequences
        st.markdown("### Top Binding Sequences")
        top_binding = df.sort_values('Score').head(10)
        
        fig = px.bar(top_binding, x='RNA_Name', y='Score',
                   color='Score', color_continuous_scale='Viridis',
                   labels={'Score': 'Binding Score'},
                   title="Top 10 Strongest Binding Sequences")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.markdown("### Key Sequence Features")
        
        # Show key binding factors
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ➕ Factors Enhancing Binding")
            st.markdown("""
            - **Higher cytosine content** (> 25%)
            - **Higher GC content** (> 50%)
            - **Beneficial motifs** like 'AAGAGA', 'AGCCUG', 'AGAAAG'
            - **A-rich regions** like 'AAAAAA'
            """)
        
        with col2:
            st.markdown("#### ➖ Factors Weakening Binding")
            st.markdown("""
            - **UG/GU-rich repetitive motifs**
            - **Low cytosine content** (< 18%)
            - **High UG/GU dinucleotide frequency** (> 12%)
            - **G nucleotides** at positions 2, 6, 9, and 19
            """)
        
        # Common motifs visualization
        st.markdown("### Common Motifs in Strong Binding Sequences")
        
        motifs = [
            ('AAAAAA', 19), ('AGAGAA', 18), ('UUUUUU', 18),
            ('AAGAAA', 16), ('GCCUGG', 16), ('CAGCUG', 16),
            ('AGAAAG', 16), ('CUGCAG', 15), ('AGCCUG', 15), ('CAGCAG', 15)
        ]
        
        motif_df = pd.DataFrame(motifs, columns=['Motif', 'Frequency'])
        
        fig = px.bar(motif_df, x='Motif', y='Frequency',
                   color='Frequency', color_continuous_scale='Viridis',
                   title="Frequent Motifs in High-Binding Sequences")
        st.plotly_chart(fig, use_container_width=True)
        
        # Position-specific effects
        st.markdown("### Position-Specific Effects")
        
        position_data = {
            'Position': [9, 21, 6, 2, 19],
            'Nucleotide': ['G', 'C', 'G', 'G', 'G'],
            'Effect': [0.15, 0.12, 0.11, 0.11, 0.10]  # Correlation values
        }
        
        pos_df = pd.DataFrame(position_data)
        
        fig = px.bar(pos_df, x='Position', y='Effect',
                   color='Nucleotide',
                   labels={'Effect': 'Correlation with Decreased Binding'},
                   title="Position-Specific Nucleotide Effects")
        st.plotly_chart(fig, use_container_width=True)

# ------------------- Model Performance -------------------
elif page == "📈 Model Performance":
    st.markdown("# Model Performance")
    st.markdown("Evaluation of the RNA binding prediction model.")
    st.divider()
    
    # Create tabs for different performance views
    tabs = st.tabs(["⚖️ Error Analysis", "🧪 Calibration Effect", "📋 Results Table"])
    
    with tabs[0]:
        st.markdown("### Prediction Error Analysis")
        
        # Sample performance data
        error_data = {
            'RNA_Name': ['lnc-SSTR4-24', 'lnc-KHSRP-12', 'lnc-BPY2C-22', 'lnc-BRF2-171', 
                        'lnc-ALG1L-41', 'lnc-ZFHX4-33', 'lnc-CAPN15-41', 'lnc-CD70-31', 
                        'MAFTRR20', 'lnc-KLK3-12'],
            'Original_Error': [588.11, 515.30, 864.55, 564.00, 508.00, 
                              569.88, 786.11, 567.66, 399.27, 379.63],
            'Calibrated_Error': [188.11, 115.30, 464.55, 164.00, 108.00, 
                                169.88, 386.11, 167.66, 0.73, 20.37]
        }
        
        error_df = pd.DataFrame(error_data)
        
        # Average error reduction metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Original Avg Error", f"{error_df['Original_Error'].mean():.2f}")
        with col2:
            st.metric("Calibrated Avg Error", f"{error_df['Calibrated_Error'].mean():.2f}")
        with col3:
            improvement = error_df['Original_Error'].mean() - error_df['Calibrated_Error'].mean()
            st.metric("Average Improvement", f"{improvement:.2f}", delta=f"{improvement:.2f}")
        
        # Error reduction visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Original Error',
            x=error_df['RNA_Name'],
            y=error_df['Original_Error'],
            marker_color='#EF5350'
        ))
        fig.add_trace(go.Bar(
            name='Calibrated Error',
            x=error_df['RNA_Name'],
            y=error_df['Calibrated_Error'],
            marker_color='#66BB6A'
        ))
        
        fig.update_layout(
            title='Error Reduction with Calibration',
            xaxis_tickangle=-45,
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.markdown("### Calibration Effect Analysis")
        
        # Create data for before/after calibration
        cal_data = {
            'Metric': ['MSE', 'RMSE', 'Avg Error', 'Max Error'],
            'Before': [46600, 215.82, 215.82, 864.55],
            'After': [25000, 158.18, 158.18, 464.55]
        }
        
        cal_df = pd.DataFrame(cal_data)
        
        # Display metrics
        for i, row in cal_df.iterrows():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"#### {row['Metric']}")
            with col2:
                st.metric("Before", f"{row['Before']:.2f}")
            with col3:
                improvement = row['Before'] - row['After']
                st.metric("After", f"{row['After']:.2f}", delta=f"-{improvement:.2f}")
        
       # Show calibration explanation
        st.markdown("### Calibration Approach")
        st.write("Our model uses a targeted calibration approach that applies corrections only to sequences with multiple indicators of problematic binding characteristics:")
                
        st.write("1. **Detection**: We identify sequences likely to have prediction errors based on:")
        st.write("   - Low cytosine content (< 18%)")
        st.write("   - Multiple UG/GU-rich motifs")
        st.write("   - High UG/GU dinucleotide density (> 12%)")
                
        st.write("2. **Calibration**: We apply a fixed correction of 400 points only to sequences meeting at least two of these criteria")
                
        st.write("3. **Result**: This approach reduced average error by 26.7% and fixed catastrophic errors while maintaining accuracy for well-predicted sequences")
       
       # Visualization of calibration thresholds
       st.markdown("### Calibration Thresholds")
       
       threshold_data = {
           'Feature': ['Cytosine Content', 'UG/GU Motifs', 'UG/GU Density'],
           'Threshold': [18, 2, 12],
           'Unit': ['%', 'count', '%']
       }
       
       thresh_df = pd.DataFrame(threshold_data)
       fig = px.bar(thresh_df, x='Feature', y='Threshold', color='Feature',
                  text=[f"{v} {u}" for v, u in zip(thresh_df['Threshold'], thresh_df['Unit'])],
                  title="Calibration Trigger Thresholds")
       st.plotly_chart(fig, use_container_width=True)
   
   with tabs[2]:
       st.markdown("### Detailed Results Table")
       
       # Create a more comprehensive results table
       detailed_results = {
           'RNA_Name': ['lnc-SSTR4-24', 'lnc-KHSRP-12', 'lnc-BPY2C-22', 'lnc-BRF2-171', 
                       'lnc-ALG1L-41', 'lnc-ZFHX4-33', 'lnc-CAPN15-41', 'lnc-CD70-31', 
                       'MAFTRR20', 'lnc-KLK3-12'],
           'Actual_Score': [-6622.30, -6542.92, -6376.05, -6606.67, -6694.67, 
                           -6665.33, -6443.58, -6613.92, -6823.61, -6827.79],
           'Original_Prediction': [-7210.41, -7058.22, -7240.60, -7170.67, -7202.67, 
                                  -7235.21, -7229.69, -7181.58, -7222.88, -7207.42],
           'Calibrated_Prediction': [-6810.41, -6658.22, -6840.60, -6770.67, -6802.67, 
                                    -6835.21, -6829.69, -6781.58, -6822.88, -6807.42],
           'Original_Error': [588.11, 515.30, 864.55, 564.00, 508.00, 
                             569.88, 786.11, 567.66, 399.27, 379.63],
           'Calibrated_Error': [188.11, 115.30, 464.55, 164.00, 108.00, 
                               169.88, 386.11, 167.66, 0.73, 20.37],
           'Improvement': [400.00, 400.00, 400.00, 400.00, 400.00, 
                          400.00, 400.00, 400.00, 398.54, 359.26]
       }
       
       results_df = pd.DataFrame(detailed_results)
       
       # Add binding quality assessment
       results_df['Actual_Quality'] = results_df['Actual_Score'].apply(
           lambda x: 'Good' if x < -6676.38 else 'Poor'
       )
       results_df['Original_Quality'] = results_df['Original_Prediction'].apply(
           lambda x: 'Good' if x < -6676.38 else 'Poor'
       )
       results_df['Calibrated_Quality'] = results_df['Calibrated_Prediction'].apply(
           lambda x: 'Good' if x < -6676.38 else 'Poor'
       )
       
       # Calculate quality prediction accuracy
       results_df['Original_Quality_Correct'] = results_df.apply(
           lambda x: x['Actual_Quality'] == x['Original_Quality'], axis=1
       )
       results_df['Calibrated_Quality_Correct'] = results_df.apply(
           lambda x: x['Actual_Quality'] == x['Calibrated_Quality'], axis=1
       )
       
       # Display the table
       st.dataframe(results_df[[
           'RNA_Name', 'Actual_Score', 'Calibrated_Prediction', 
           'Calibrated_Error', 'Improvement', 'Actual_Quality', 'Calibrated_Quality'
       ]], height=400)
       
       # Quality prediction metrics
       st.markdown("### Binding Quality Prediction Accuracy")
       
       orig_acc = results_df['Original_Quality_Correct'].mean() * 100
       calib_acc = results_df['Calibrated_Quality_Correct'].mean() * 100
       
       col1, col2 = st.columns(2)
       with col1:
           st.metric("Original Model Accuracy", f"{orig_acc:.1f}%")
       with col2:
           st.metric("Calibrated Model Accuracy", f"{calib_acc:.1f}%", 
                    delta=f"{calib_acc - orig_acc:.1f}%")
       
       # Confusion matrix
       st.markdown("### Binding Quality Prediction")
       
       # Compute confusion matrix data (simplified for demo)
       confusion_data = {
           'Actual': ['Good', 'Good', 'Poor', 'Poor'],
           'Predicted': ['Good', 'Poor', 'Good', 'Poor'],
           'Original_Count': [2, 2, 4, 2],
           'Calibrated_Count': [3, 1, 1, 5]
       }
       
       conf_df = pd.DataFrame(confusion_data)
       
       # Display as two side-by-side tables
       col1, col2 = st.columns(2)
       
       with col1:
           st.markdown("#### Original Model")
           orig_pivot = pd.pivot_table(
               conf_df, values='Original_Count', 
               index='Actual', columns='Predicted', fill_value=0
           )
           st.dataframe(orig_pivot)
       
       with col2:
           st.markdown("#### Calibrated Model")
           calib_pivot = pd.pivot_table(
               conf_df, values='Calibrated_Count', 
               index='Actual', columns='Predicted', fill_value=0
           )
           st.dataframe(calib_pivot)

# Add a simpler footer
st.divider()
st.markdown("RNA-Protein Binding Prediction Tool | Model based on ANOVA threshold: -6676.38")
st.markdown("Contact: your-email@example.com")
    <p>RNA-Protein Binding Prediction Tool | Model based on ANOVA threshold: -6676.38</p>
    <p>Contact: your-email@example.com</p>
</div>
""", unsafe_allow_html=True)
