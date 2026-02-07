# quantum/qaoa_solver.py

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import Aer


def build_qaoa_circuit(Q):
    """
    Build a single-layer QAOA circuit for a given QUBO matrix Q.

    Q is assumed symmetric (upper triangle used).
    Returns:
        qc, gamma, beta
    """
    n = Q.shape[0]

    gamma = Parameter("gamma")
    beta = Parameter("beta")

    qc = QuantumCircuit(n)

    # Start in equal superposition
    qc.h(range(n))

    # --- Cost Hamiltonian (Problem unitary) ---
    # Diagonal terms
    for i in range(n):
        if Q[i, i] != 0:
            qc.rz(2 * gamma * Q[i, i], i)

    # Off-diagonal terms
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] != 0:
                qc.cx(i, j)
                qc.rz(2 * gamma * Q[i, j], j)
                qc.cx(i, j)

    # --- Mixer Hamiltonian ---
    for i in range(n):
        qc.rx(2 * beta, i)

    return qc, gamma, beta


def compute_energy(bitstring, Q):
    """
    Compute QUBO energy E = x^T Q x for a given bitstring.
    bitstring is like "0101" (Qiskit order is reversed vs our x vector).
    """
    x = np.array([int(b) for b in bitstring[::-1]])
    return float(x @ Q @ x)


def run_qaoa_and_extract_solution(qc, gamma, beta, params, Q, shots=1024):
    """
    Run QAOA on Aer simulator and extract best solution by minimum energy.
    """
    backend = Aer.get_backend("aer_simulator")

    qc_bound = qc.assign_parameters({
        gamma: params["gamma"],
        beta: params["beta"]
    })

    qc_bound.measure_all()

    compiled = transpile(qc_bound, backend)
    result = backend.run(compiled, shots=shots).result()

    counts = result.get_counts()

    # Choose best by minimum energy (not just max counts)
    best_bit = None
    best_energy = float("inf")

    for bit, _cnt in counts.items():
        E = compute_energy(bit, Q)
        if E < best_energy:
            best_energy = E
            best_bit = bit

    return best_bit, best_energy, counts