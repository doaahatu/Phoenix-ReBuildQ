# quantum/feature_engineering.py

def engineer_context_features(df):
    """
    Add contextual and engineering-aware features to road data.

    This stage reflects real-world humanitarian and urban-planning logic:
    - Roads closer to hospitals are higher priority.
    - Residential areas receive higher weight.
    - Soil type affects reconstruction cost.
    """

    # -------------------------------------------------
    # 1. Hospital Proximity Score
    # -------------------------------------------------
    # Non-linear importance:
    # Very close hospitals matter MUCH more than far ones
    df["hospital_score"] = 1 / (1 + df["distance_to_hospital"])


    # -------------------------------------------------
    # 2. Land Use Priority Factor
    # -------------------------------------------------
    # Policy-driven priorities
    land_use_factor = {
        "residential": 1.2,   # civilians first
        "mixed": 1.0,
        "industrial": 0.8     # lower humanitarian priority
    }

    df["land_use_factor"] = df["land_use"].map(land_use_factor)


    # -------------------------------------------------
    # 3. Soil Difficulty Factor
    # -------------------------------------------------
    # Engineering reality:
    # Sandy soil is harder and more expensive to rebuild
    soil_factor = {
        "sandy": 1.4,
        "compact": 1.0
    }

    df["soil_factor"] = df["soil"].map(soil_factor)


    # -------------------------------------------------
    # 4. Final Reconstruction Cost
    # -------------------------------------------------
    df["final_cost"] = df["base_cost"] * df["soil_factor"]


    return df