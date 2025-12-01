import streamlit as st
import pandas as pd
import os
import shutil
import numpy as np

# Import our backend engines
from src.loader import TensorFlowLoader
from src.profiler import ModelProfiler
from src.evaluator import AccuracyEngine
from src.ranker import ModelRanker

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="EdgeOpt Profiler", page_icon="⚡", layout="wide")

st.title("⚡ EdgeOpt: Model Deployment Optimizer")
st.markdown("Upload your model, define your constraints, and find the perfect deployment strategy.")

# --- SIDEBAR: INPUTS & CONTROLS ---
with st.sidebar:
    st.header("1. Upload Model")
    uploaded_file = st.file_uploader("Upload .h5 or .zip (SavedModel)", type=["h5", "zip"])
    
    st.header("2. Upload Validation Data (Optional)")
    st.info("Required for Pruning & Real Accuracy")
    uploaded_data_x = st.file_uploader("Upload Input Data (.npy)", type=["npy"])
    uploaded_data_y = st.file_uploader("Upload Labels (.npy)", type=["npy"])

    st.markdown("---")
    st.header("3. Optimization Priorities")
    st.write("Adjust weights for the ranking equation:")
    
    # The Sliders for your Equation
    alpha = st.slider("Alpha (Size Priority)", 0.0, 1.0, 0.5, help="Higher = You want smaller files")
    beta = st.slider("Beta (Accuracy Priority)", 0.0, 1.0, 0.5, help="Higher = You want better accuracy")
    gamma = st.slider("Gamma (Speed Priority)", 0.0, 1.0, 0.5, help="Higher = You want faster inference")

# --- MAIN LOGIC ---

# Initialize Session State (To keep results on screen while you move sliders)
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

def run_optimization():
    # 1. Save uploaded file to disk so loader can find it
    os.makedirs("inputs", exist_ok=True)
    temp_path = os.path.join("inputs", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 2. Handle Optional Data
    X_val, y_val = None, None
    if uploaded_data_x and uploaded_data_y:
        X_val = np.load(uploaded_data_x)
        y_val = np.load(uploaded_data_y)
        st.success(f"✅ Loaded Validation Data: {X_val.shape}")

    # 3. Initialize Engines
    loader = TensorFlowLoader()
    profiler = ModelProfiler()
    evaluator = AccuracyEngine()
    
    # 4. Run Pipeline
    with st.spinner("⚙️ Optimizing Model (Baseline, Quantization, Pruning)..."):
        # This calls your code from Day 2/3!
        model_paths = loader.process_pipeline(temp_path, X_val, y_val)
    
    # 5. Profile & Evaluate
    results = []
    progress_bar = st.progress(0)
    
    step = 0
    total_steps = len(model_paths)
    
    for variant_name, path in model_paths.items():
        # Update UI
        step += 1
        progress_bar.progress(step / total_steps)
        
        # Profile
        metrics = profiler.profile_model(path)
        # Evaluate
        acc = evaluator.get_accuracy(path, variant_name, X_test=X_val, y_test=y_val)
        
        entry = {'model': variant_name}
        entry.update(metrics)
        entry['accuracy'] = round(acc, 4)
        results.append(entry)
        
    st.session_state.results_df = pd.DataFrame(results)
    st.success("Analysis Complete!")

# --- UI DISPLAY ---

if uploaded_file:
    if st.button("🚀 Start Optimization Process", type="primary"):
        run_optimization()

# Show Results if they exist
if st.session_state.results_df is not None:
    df = st.session_state.results_df
    
    # 1. RANKING ENGINE (Day 4 Logic)
    ranker = ModelRanker()
    ranked_df = ranker.rank_models(df, alpha, beta, gamma)
    
    # Get the Winner (Row 0)
    winner = ranked_df.iloc[0]
    
    st.divider()
    
    # --- THE WINNER CARD ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🏆 Recommended Model")
        st.metric(label="Model Variant", value=winner['model'].upper())
        st.write(f"**Score:** {winner['final_score']:.4f} (Lower is better)")
        
    with col2:
        st.subheader("Why this model?")
        c1, c2, c3 = st.columns(3)
        c1.metric("Size", f"{winner['size_mb']:.2f} MB")
        c2.metric("Latency", f"{winner['latency_ms']:.2f} ms")
        c3.metric("Accuracy", f"{winner['accuracy']:.2%}")

    # --- DETAILED DATA ---
    st.divider()
    st.subheader("📊 Detailed Comparison")
    
    # Highlight the winner in the table
    st.dataframe(
        ranked_df.style.highlight_min(subset=['final_score'], color='#d4edda', axis=0),
        use_container_width=True
    )
    
    # --- VISUALIZATION ---
    st.subheader("📈 Trade-off Visualization")
    st.caption("Bubble size = Model File Size")
    
    # Simple Scatter Plot using Streamlit
    st.scatter_chart(
        ranked_df,
        x='latency_ms',
        y='accuracy',
        size='size_mb',
        color='model',
        height=400
    )

else:
    if not uploaded_file:
        st.info("👈 Please upload a model in the sidebar to begin.")