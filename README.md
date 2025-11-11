# Leukemia Classification with Qiskit (VQC)

This repository contains the **Qiskit implementation** of our research project

---

## 🧩 Repository Structure
leukemia_qiskit_vqc/
│
├── src/ # Qiskit scripts
│ ├── qiskit_vqc.py # Main variational quantum circuit implementation
│ ├── preprocess_golub.py # Data preprocessing pipeline
│ ├── visualize_circuit.py# Quantum circuit visualization utility
│
├── data/ # Gene expression data (excluded from repo)
├── results/ # Output plots, metrics, and saved models
├── requirements.txt # Python dependencies
└── .gitignore # Ignore environment and temp files

---

## ⚙️ Installation & Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/3em07/leukemia_qiskit_vqc.git
   cd leukemia_qiskit_vqc
   python3 -m venv qiskit_env
2. Create and activate virtual environment (optional )
   python3 -m venv qiskit_env
   source qiskit_env/bin/activate  # macOS/Linux
   qiskit_env\Scripts\activate     # Windows
3. download dependencies 
   pip install -r requirements.txt

   running model :
python3 src/qiskit_vqc.py --csv data/processed/golub_combined.csv --features 16 --reps 2

   
