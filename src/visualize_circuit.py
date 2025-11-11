# visual circuit for first run
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import TwoLocal
import matplotlib.pyplot as plt

# --- configuration ---
num_features = 16   
reps = 2           

# --- angle encoding layer ---
x = ParameterVector("x", num_features)
qc = QuantumCircuit(num_features, name="AngleEncoding")
for i in range(num_features):
    qc.ry(x[i], i)

# --- variational (entangling) layer ---
ansatz = TwoLocal(
    num_qubits=num_features,
    rotation_blocks="ry",
    entanglement_blocks="cz",
    entanglement="full",
    reps=reps
)

# combine the encoding + ansatz
qc.compose(ansatz, inplace=True)

# --- draw and show ---
qc.draw("mpl")      # matplotlib drawer
plt.title(f"{num_features}-Qubit Variational Quantum Circuit (reps={reps})")
plt.tight_layout()
plt.savefig("results/vqc_circuit.png", dpi=300)
plt.show()

