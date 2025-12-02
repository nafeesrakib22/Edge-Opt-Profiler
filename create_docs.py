import os

# --- CONTENT FOR README.MD ---
readme_text = r"""# ⚡ EdgeOpt: Model Deployment Optimizer

**EdgeOpt** is a decision-support tool designed for Machine Learning Engineers deploying models to resource-constrained edge devices (IoT, Mobile, Embedded). It automatically profiles deep learning models, applies industry-standard optimization techniques, and ranks the results based on user-defined priorities for **Accuracy**, **Latency**, and **Size**.

<p align="center">
  <br>
  <img src="assets/EdgeOpt Demo.gif" alt="EdgeOpt Tool Demo" width="800">
  <br>
  <em>Real-time optimization of MobileNetV2 showing upto 4x size reduction with minimal accuracy loss accross the chosen optimization techniques.</em>
  <br>
</p>


## 🚀 Key Features

* **Universal Ingestion:** Seamlessly handles `.h5` (Keras) and `.zip` (TensorFlow SavedModel) formats.
* **Automated Optimization Pipeline:**
    1.  **Dynamic Quantization (INT8):** Reduces model size by ~4x.
    2.  **Float16 Quantization (FP16):** Reduces model size by ~2x (GPU-friendly).
    3.  **Sparse Pruning:** Magnitude-based weight pruning with "Smart Schedule" fine-tuning.
* **Real-Time Profiling:** Measures on-device inference latency (ms) and storage footprint (MB).
* **Smart Evaluation:** Calculates real accuracy using uploaded validation sets or simulates performance for rapid prototyping.
* **Decision Engine:** Uses a weighted scoring algorithm to recommend the optimal deployment strategy based on specific constraints.

## 🛠️ Installation & Setup

To run this project, you need **Python 3.10 or 3.11**.

### 1. Clone the Repository
```bash
git clone [https://github.com/nafeesrakib22/Edge-Opt-Profiler.git](https://github.com/nafeesrakib22/Edge-Opt-Profiler.git)
cd Edge-Opt-Profiler
```

### 2. Create & Activate Virtual Environment (Recommended)
This ensures the dependencies don't conflict with your system.

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```
*The app will open automatically in your browser at `http://localhost:8501`.*

---

## 📊 How to Use

### 1. Prepare Your Model
You can use your own model or generate a demo model using the included scripts:
* **MobileNetV2 (Vision):** Run `python generate_mobilenet.py`
* **ResNet50 (Complex Vision):** Run `python generate_resnet_sponge.py`
* **MLP (Tabular):** Run `python generate_tabular_demo.py`

**NOTE:** The repository comes with a pre-generated `demo_mobilenet` folder containing a trained model and validation data, so you can test the tool immediately without running any generation scripts!


### 2. In the App
1.  **Upload Model:** Select the `.h5` file from the sidebar.
2.  **Upload Data (Crucial):** Upload `X_val.npy` and `y_val.npy`.
    * *Note: Providing data unlocks the **Pruning** optimizer and **Real Accuracy** measurement.*
3.  **Run Optimization:** Click the **🚀 Start Optimization Process** button.
4.  **Analyze Results:**
    * The tool will generate a table of results (`Baseline`, `Dynamic`, `FP16`, `Pruning`).
    * Use the **Alpha (Size)**, **Beta (Accuracy)**, and **Gamma (Speed)** sliders to change the recommendation logic.

## 📂 Project Structure

```text
/EdgeOpt_Project
│
├── /src
│   ├── loader.py       # Pipeline orchestration & TFLite conversion
│   ├── pruner.py       # Intelligent pruning logic (Polynomial Decay + Smart Epochs)
│   ├── profiler.py     # Latency & Size measurement engine
│   ├── evaluator.py    # Accuracy verification (supports Classification & Regression)
│   └── ranker.py       # Weighted Min-Max Normalization algorithm
│
├── app.py              # Streamlit Interactive Dashboard
├── requirements.txt    # Dependency lockfile
└── generate_*.py       # Scripts to create real-world test cases
```

## 🧪 Optimization Techniques Explanation

| Technique | Description | Why we chose it |
| :--- | :--- | :--- |
| **Baseline (FP32)** | The original model converted to TFLite without optimization. | Serves as the "Gold Standard" for accuracy comparison. |
| **Dynamic Quantization** | Converts weights to 8-bit integers (INT8) but computes in float. | **Best for Size:** Offers 4x compression with minimal accuracy loss on most CPU architectures. |
| **Float16 (FP16)** | Converts weights to 16-bit floating point. | **Best for GPU:** Reduces size by 2x while maintaining near-perfect accuracy. Ideal for mobile GPUs. |
| **Pruning** | Removes 30-50% of the lowest-magnitude weights and fine-tunes the model. | **Best for Efficiency:** Demonstrates how removing redundancy can sometimes *improve* accuracy (Regularization effect). |

## 🏆 Ranking Methodology

The tool ranks models using a **Weighted Min-Max Cost Function**. Lower scores are better.

$$ Cost = \alpha \cdot \text{Norm}(Size) + \beta \cdot (1 - \text{Norm}(Accuracy)) + \gamma \cdot \text{Norm}(Latency) $$

* **Alpha ($\alpha$):** Weight for File Size.
* **Beta ($\beta$):** Weight for Accuracy.
* **Gamma ($\gamma$):** Weight for Inference Speed.

This allows engineers to mathematically define "Best" based on their specific constraints (e.g., "I need the smallest model possible, speed doesn't matter").

---

# 🧠 Technical Design Decisions

## 1. Core Architecture: The "Funnel" Pattern
We architected the system using a **Funnel Pattern** to handle multiple input formats.
* **Decision:** All inputs (`.h5`, `.zip`) are immediately converted into a standardized **TensorFlow Lite (TFLite)** intermediate representation.
* **Reasoning:** Supporting profiling for raw PyTorch, Keras, and ONNX models simultaneously would create dependency conflicts. By standardizing on TFLite, we ensure consistent metrics (Size/Latency) across all variants.

## 2. Pruning Strategy & The "Sparsity Cliff"
During development, we encountered significant challenges with Model Pruning, specifically regarding "Model Collapse" on efficient architectures.

### The Problem
When pruning **MobileNetV2** (an architecture already highly optimized), applying a standard pruning schedule caused accuracy to drop from **70% to 19%**.
* **Root Cause:** MobileNet is a "Dry Sponge" (parameter-efficient). Pruning 50% of its weights cut into critical features.
* **Secondary Cause:** The Transfer Learning base layers were frozen, preventing the model from "healing" the cut connections.

### The Solution: "Safe Mode" Pruning
We implemented a custom `ModelPruner` class with specific safeguards:
1.  **Unfreezing:** We force-unfreeze `model.trainable = True` before pruning to allow the entire network to adapt.
2.  **Polynomial Decay:** Replaced constant sparsity with a gradual schedule that ramps up difficulty over time.
3.  **Adaptive Epochs:** The system detects dataset size.
    * *Small Datasets (<5k samples):* Boosts fine-tuning to **15 epochs** to allow convergence on limited signal.
    * *Large Datasets:* Uses 6 epochs to prevent overfitting.
4.  **Conservative Targets:** Reduced target sparsity to **30%** for sensitive models, while allowing up to **50%** for redundant models (like ResNet).

## 3. Handling Keras Version Conflicts
We faced compatibility issues between Keras 3.x files and our stable TensorFlow 2.15 environment (e.g., `DTypePolicy` errors).
* **Decision:** Instead of creating complex "Patching" logic to hack file headers (which proved unstable), we enforced a **Standardized Environment**.
* **Implementation:** We provided generator scripts (`generate_mobilenet.py`) that run within the user's environment to create guaranteed-compatible `.h5` files. This ensures reproducibility and stability.

## 4. Measurement & Evaluation
* **Latency:** We measure "Warm" latency. The profiler runs 10 inference passes to warm up the CPU cache before measuring the average time, ensuring results reflect real-world runtime performance.
* **Accuracy:** We implemented a hybrid evaluator that supports both **Classification** (argmax) and **Regression** (threshold) tasks, allowing the tool to support diverse model types (CNNs vs. MLPs).

## 5. Why Local Processing?
The brief required "no cloud API dependencies."
* **Implementation:** All processing uses the local CPU via `tensorflow` and `tensorflow-model-optimization`.
* **Benefit:** This allows the tool to run on secure, air-gapped machines often found in industrial edge deployment scenarios.
"""

# --- WRITE FILES ---
print("📄 Generating documentation...")

# Write README.md
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_text)
print("✅ Created README.md")

print("🎉 Documentation complete!")