# quantum/impact_scoring.py

from sklearn.preprocessing import MinMaxScaler


def compute_impact_scores(df, weights):
    """
    Compute a humanitarian impact score for each road.

    This score represents how critical it is to reconstruct the road,
    based on normalized and weighted humanitarian indicators.
    """

    # -------------------------------------------------
    # 1. Normalize Features
    # -------------------------------------------------
    # We normalize to avoid scale dominance
    scaler = MinMaxScaler()

    df[
        ["damage_n", "population_n", "hospital_n", "aid_n"]
    ] = scaler.fit_transform(
        df[
            ["damage", "population", "hospital_score", "aid_route"]
        ]
    )

    # -------------------------------------------------
    # 2. Weighted Impact Score
    # -------------------------------------------------
    df["impact"] = (
        weights["damage"]     * df["damage_n"] +
        weights["population"] * df["population_n"] +
        weights["hospital"]   * df["hospital_n"] +
        weights["aid"]        * df["aid_n"]
    )

    # -------------------------------------------------
    # 3. Apply Land Use Priority
    # -------------------------------------------------
    # Residential areas get higher priority
    df["impact"] *= df["land_use_factor"]

    return df