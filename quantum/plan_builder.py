# quantum/plan_builder.py

def generate_recovery_plan(df, bitstring):
    """
    Convert QAOA bitstring solution into a structured recovery plan.

    bitstring example: "1010"
    """

    # -------------------------------------------------
    # 1. Decode bitstring (Qiskit order is reversed)
    # -------------------------------------------------
    selection = [int(b) for b in bitstring[::-1]]
    df["selected"] = selection

    # -------------------------------------------------
    # 2. Separate selected vs deferred
    # -------------------------------------------------
    selected_df = df[df["selected"] == 1]
    deferred_df = df[df["selected"] == 0]

    # -------------------------------------------------
    # 3. Build summary report
    # -------------------------------------------------
    summary = {
        "selected_roads": selected_df["id"].tolist(),
        "deferred_roads": deferred_df["id"].tolist(),
        "total_cost": round(selected_df["final_cost"].sum(), 2),
        "total_impact": round(selected_df["impact"].sum(), 3),
        "population_served": int(selected_df["population"].sum()),
        "num_selected": int(selected_df.shape[0]),
        "num_deferred": int(deferred_df.shape[0])
    }

    return df, summary