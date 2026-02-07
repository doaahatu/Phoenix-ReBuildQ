# quantum/qubo.py

import numpy as np


def build_qubo(df, budget, lambda_penalty):
    """
    Build a strong QUBO for road reconstruction using
    quadratic budget constraint (knapsack-style).

    Objective:
        maximize total impact
        subject to sum(cost_i * x_i) <= budget

    Converted to minimization:
        - sum(impact_i * x_i)
        + lambda * (sum(cost_i * x_i) - budget)^2
    """

    n = len(df)
    Q = np.zeros((n, n))

    impacts = df["impact"].values
    costs = df["final_cost"].values

    # -------------------------
    # Diagonal terms
    # -------------------------
    for i in range(n):
        # Impact term (maximize impact → minimize -impact)
        Q[i, i] += -impacts[i]

        # Budget quadratic expansion:
        # c_i^2 * x_i  - 2 * B * c_i * x_i
        Q[i, i] += lambda_penalty * (costs[i] ** 2)
        Q[i, i] += -2 * lambda_penalty * budget * costs[i]

    # -------------------------
    # Off-diagonal terms
    # -------------------------
    for i in range(n):
        for j in range(i + 1, n):
            # 2 * c_i * c_j * x_i * x_j
            Q[i, j] += 2 * lambda_penalty * costs[i] * costs[j]
            Q[j, i] = Q[i, j]  # symmetric

    return Q