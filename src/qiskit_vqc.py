import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_algorithms.optimizers import COBYLA


# --- DATA HANDLING ---
def load_golub(csv_path: str):
    """Load the cleaned Golub dataset."""
    df = pd.read_csv(csv_path)
    y = df["label"].values
    X = df.drop(columns=["label"]).values
    return X, y, df.drop(columns=["label"]).columns.tolist()


def select_features(X, y, k=16):
    """Select k most informative genes using ANOVA F-score."""
    selector = SelectKBest(score_func=f_classif, k=k)
    Xk = selector.fit_transform(X, y)
    mask = selector.get_support(indices=True)
    return Xk, mask


def scale_to_angle(X):
    """Scale features to [0, π] for angle encoding."""
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    return scaler.fit_transform(X), scaler


# --- CIRCUIT BUILDING ---
def angle_encoding_circuit(n_qubits: int):
    """Create the angle encoding circuit."""
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name="AngleEncode")
    for i in range(n_qubits):
        qc.ry(x[i], i)
    return qc, x


def build_vqc(num_features: int, reps: int = 2):
    """Build a variational quantum circuit (VQC)."""
    feature_map, x_params = angle_encoding_circuit(num_features)
    ansatz = TwoLocal(num_features, "ry", "cz", entanglement="full", reps=reps)
    qc = QuantumCircuit(num_features, name="VQC")
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    return qc, list(x_params), list(ansatz.parameters)


# --- TRAINING ---
def train_eval_vqc(X, y, reps=2, test_size=0.3, seed=42):
    """Train and evaluate a VQC classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    n_features = X_train.shape[1]
    circuit, x_params, w_params = build_vqc(n_features, reps=reps)

    # Define the estimator and QNN
    estimator = Estimator()
    qnn = EstimatorQNN(circuit=circuit, input_params=x_params, weight_params=w_params)

    # Use COBYLA optimizer (Qiskit >=1.2)
    optimizer = COBYLA(maxiter=200)
    clf = NeuralNetworkClassifier(qnn, optimizer=optimizer)

    # Train model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    auc = None
    try:
        y_proba = getattr(clf, "predict_proba", None)
        if y_proba:
            auc = roc_auc_score(y_test, y_proba(X_test)[:, 1])
    except Exception:
        pass

    # Print results
    print("\n=== RESULTS ===")
    print("Accuracy:", round(acc, 4))
    if auc:
        print("ROC-AUC:", round(auc, 4))
    print(classification_report(y_test, y_pred, digits=4))

    return circuit


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to cleaned Golub CSV file")
    parser.add_argument("--features", type=int, default=16, help="Number of genes to select")
    parser.add_argument("--reps", type=int, default=2, help="Circuit depth (TwoLocal reps)")
    args = parser.parse_args()

    # Load dataset
    X, y, gene_names = load_golub(args.csv)

    # Feature selection
    Xk, mask = select_features(X, y, k=args.features)
    selected = [gene_names[i] for i in mask]
    print(f"\nSelected {len(selected)} genes:\n", selected)

    # Scale and train
    X_scaled, _ = scale_to_angle(Xk)
    train_eval_vqc(X_scaled, y, reps=args.reps)


if __name__ == "__main__":
    main()

