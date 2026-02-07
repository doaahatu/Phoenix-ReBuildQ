# quantum/qubo.py

import numpy as np


def build_qubo(df, budget, lambda_penalty):
    """
    Build a QUBO matrix for the reconstruction problem.

    Objective:
    - Maximize humanitarian impact
    - Respect budget constraint via penalty
    """

    # -------------------------------------------------
    # 1. Extract values
    # -------------------------------------------------
    impact = df["impact"].values

    cost = df["final_cost"].values

    # Normalize cost to improve numerical stability
    cost = cost / cost.max()

    # Normalize budget accordingly
    budget = budget / cost.max()

    n = len(df)

    Q = np.zeros((n, n))

    # -------------------------------------------------
    # 2. Objective: maximize impact → minimize -impact
    # -------------------------------------------------
    for i in range(n):
        Q[i, i] += -impact[i]

    # -------------------------------------------------
    # 3. Budget penalty: (Σ cost_i x_i − B)^2
    # -------------------------------------------------
    for i in range(n):
        Q[i, i] += lambda_penalty * cost[i] ** 2
        Q[i, i] += -2 * lambda_penalty * budget * cost[i]

    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += 2 * lambda_penalty * cost[i] * cost[j]

    return Q