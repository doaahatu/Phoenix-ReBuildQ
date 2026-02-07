import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import os

# =========================================================
# Quantum modules (keep as we built)
# =========================================================
from quantum.data_loader import load_road_data
from quantum.feature_engineering import engineer_context_features
from quantum.impact_scoring import compute_impact_scores
from quantum.qubo import build_qubo
from quantum.qaoa_solver import build_qaoa_circuit, run_qaoa_and_extract_solution
from quantum.plan_builder import generate_recovery_plan
from visualization.map_view import visualize_gaza_dashboard


# =========================================================
# Streamlit compatibility helpers (fix use_container_width error)
# =========================================================
def st_image_compat(img, caption=None):
   
    try:
        st.image(img, caption=caption)
    except TypeError:
        st.image(img, caption=caption)


def show_image_if_exists(path, caption=None):
    if path and os.path.exists(path):
        st_image_compat(path, caption=caption)


# =========================================================
# Page + Theme (KEEP OLD UI STYLE)
# =========================================================
st.set_page_config(
    page_title="Phoenix ReBuildIQ – Gaza",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Top hero image (keep)
show_image_if_exists("assets/hero_gaza.png")

st.markdown("""
<style>
:root{
  --bg:#0e1117;
  --card:#151a22;
  --card2:#1c1f26;
  --accent:#ff914d;
  --muted:#9aa4b2;
  --good:#2dd4bf;
  --warn:#fbbf24;
  --bad:#fb7185;
  --ink:#e5e7eb;
}
body { background-color: var(--bg); }
h1,h2,h3,h4 { color: var(--accent); }
.small-muted { color: var(--muted); font-size: 13px; }
.card {
  background: linear-gradient(180deg, var(--card) 0%, var(--card2) 100%);
  border: 1px solid rgba(255,255,255,0.06);
  padding: 16px 18px; border-radius: 16px; margin-bottom: 14px;
}
.kpi {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  padding: 14px; border-radius: 14px; text-align: center;
}
.tag {
  display:inline-block; padding:6px 10px; border-radius: 999px;
  background: rgba(255,145,77,0.15); color: var(--accent);
  border:1px solid rgba(255,145,77,0.25); font-size: 12px;
  margin-right: 8px;
}
.badge-good{color:var(--good); font-weight:700;}
.badge-warn{color:var(--warn); font-weight:700;}
.badge-bad{color:var(--bad); font-weight:700;}
hr { border: none; height: 1px; background: rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)

# =========================================================
# Assets (KEEP)
# =========================================================
ASSETS = {
    "hero": "assets/hero_gaza.png",
    "masterplan": "assets/masterplan_realistic.png",
    "blueprint": "assets/blueprint.png",
    "zone_map": "assets/zone_map.png",
    "timeline": "assets/timeline.png",
}

# =========================================================
# Title (KEEP)
# =========================================================
st.title("🐦‍🔥 Phoenix ReBuildIQ — Gaza")
st.caption("Quantum-AI Decision Engine for Post-War Urban Recovery (Hackathon Demo)")

colH1, colH2 = st.columns([1.35, 1])
with colH1:
    st.markdown("""
    <div class="card">
      <span class="tag">Gaza Case Study</span>
      <span class="tag">AI Need & Deficit Scoring</span>
      <span class="tag">Quantum Planning + QAOA</span>
      <p style="color:var(--ink); font-size:15px; margin-top:10px;">
      Phoenix ReBuildIQ helps decision-makers answer: <b>What to rebuild first, where, and why</b> —
      balancing <b>impact</b>, <b>budget</b>, <b>time</b>, and <b>fairness</b>.
      </p>
      <p class="small-muted">
      Data note: Values are approximations suitable for hackathon demos and decision-support prototyping.
      </p>
    </div>
    """, unsafe_allow_html=True)

with colH2:
    show_image_if_exists(ASSETS["hero"], caption="Header Visual")

st.divider()

# =========================================================
# ------------------- (OLD PART) CITY / ZONES MODEL -------------------
# =========================================================

# Gaza Zones (approx realistic)
zones_data = [
    ["Gaza City",      650000, 85, 5, 0.55],
    ["North Gaza",     350000, 80, 4, 0.60],
    ["Jabalia Camp",   120000, 90, 2, 0.70],
    ["Deir al-Balah",  300000, 70, 3, 0.45],
    ["Nuseirat Camp",   90000, 88, 2, 0.65],
    ["Khan Younis",    430000, 75, 4, 0.50],
    ["Rafah",          280000, 65, 3, 0.40],
    ["Bureij Camp",     75000, 85, 2, 0.60],
]
df_city = pd.DataFrame(zones_data, columns=["Zone", "Population", "DamagePct", "ServiceAvail", "DisplacedRatio"])

# Planning Assumptions (per-capita targets)
TARGETS = {
    "HospitalsPer100k": 1.0,
    "SchoolsPer50k": 4.0,
    "HousingUnitsPerPerson": 1/5
}

BASE_COUNTS = {
    "Gaza City":     {"Hospitals": 6, "Schools": 55, "HousingUnits": 110000, "RoadIndex": 0.55},
    "North Gaza":    {"Hospitals": 3, "Schools": 28, "HousingUnits": 55000,  "RoadIndex": 0.50},
    "Jabalia Camp":  {"Hospitals": 1, "Schools": 12, "HousingUnits": 18000,  "RoadIndex": 0.40},
    "Deir al-Balah": {"Hospitals": 2, "Schools": 24, "HousingUnits": 52000,  "RoadIndex": 0.52},
    "Nuseirat Camp": {"Hospitals": 1, "Schools": 10, "HousingUnits": 15000,  "RoadIndex": 0.42},
    "Khan Younis":   {"Hospitals": 4, "Schools": 36, "HousingUnits": 72000,  "RoadIndex": 0.54},
    "Rafah":         {"Hospitals": 2, "Schools": 22, "HousingUnits": 47000,  "RoadIndex": 0.50},
    "Bureij Camp":   {"Hospitals": 1, "Schools": 8,  "HousingUnits": 12000,  "RoadIndex": 0.40},
}

df_city["Hospitals_Base"] = df_city["Zone"].map(lambda z: BASE_COUNTS[z]["Hospitals"])
df_city["Schools_Base"] = df_city["Zone"].map(lambda z: BASE_COUNTS[z]["Schools"])
df_city["HousingUnits_Base"] = df_city["Zone"].map(lambda z: BASE_COUNTS[z]["HousingUnits"])
df_city["RoadIndex_Base"] = df_city["Zone"].map(lambda z: BASE_COUNTS[z]["RoadIndex"])

df_city["Hospitals_Target"] = np.ceil((df_city["Population"]/100000) * TARGETS["HospitalsPer100k"]).astype(int)
df_city["Schools_Target"] = np.ceil((df_city["Population"]/50000) * TARGETS["SchoolsPer50k"]).astype(int)
df_city["HousingUnits_Target"] = np.ceil(df_city["Population"] * TARGETS["HousingUnitsPerPerson"]).astype(int)

df_city["Hospitals_Shortage_Pre"] = (df_city["Hospitals_Target"] - df_city["Hospitals_Base"]).clip(lower=0)
df_city["Schools_Shortage_Pre"] = (df_city["Schools_Target"] - df_city["Schools_Base"]).clip(lower=0)
df_city["Housing_Shortage_Pre"] = (df_city["HousingUnits_Target"] - df_city["HousingUnits_Base"]).clip(lower=0)

damage_factor = (df_city["DamagePct"]/100.0)
df_city["Hospitals_Shortage"] = (df_city["Hospitals_Shortage_Pre"] + np.ceil(df_city["Hospitals_Base"] * damage_factor * 0.6)).astype(int)
df_city["Schools_Shortage"]   = (df_city["Schools_Shortage_Pre"]   + np.ceil(df_city["Schools_Base"]   * damage_factor * 0.5)).astype(int)
df_city["Housing_Shortage"]   = (df_city["Housing_Shortage_Pre"]   + np.ceil(df_city["HousingUnits_Base"] * damage_factor * 0.35)).astype(int)

# Sidebar Controls (KEEP + add Quantum knobs without breaking old)
with st.sidebar:
    st.header("Planning Controls")

    total_budget = st.slider("Total Budget (Million USD)", 150, 1500, 450, 25)
    horizon_months = st.slider("Planning Horizon (months)", 12, 60, 36, 6)

    st.subheader("Top-K Plans")
    k_plans = st.slider("How many plans to generate?", 2, 3, 3, 1)

    st.subheader("Include Project Types")
    project_types = st.multiselect(
        "Select types",
        ["Housing", "Hospitals", "Schools", "Infrastructure", "Roads", "Water & Sanitation", "Power Grid", "Public Spaces"],
        default=["Housing", "Hospitals", "Schools", "Infrastructure", "Roads"]
    )

    st.subheader("Visuals")
    show_masterplan = st.checkbox("Show Master Plan Visuals", value=True)

    st.divider()
    st.subheader("Quantum Roads (QAOA) Controls")
    q_budget = st.slider("Road Reconstruction Budget (relative units)", 5, 20, 12, 1)
    q_lambda = st.slider("Budget Penalty (λ)", 1, 20, 5, 1)
    q_gamma = st.slider("QAOA γ", 0.0, 3.0, 0.8, 0.05)
    q_beta = st.slider("QAOA β", 0.0, 3.0, 0.7, 0.05)

    st.subheader("Road Impact Weights")
    q_weights = {
        "damage": st.slider("Road Damage Weight", 0.0, 1.0, 0.35),
        "population": st.slider("Population Weight", 0.0, 1.0, 0.35),
        "hospital": st.slider("Hospital Proximity Weight", 0.0, 1.0, 0.20),
        "aid": st.slider("Aid Route Weight", 0.0, 1.0, 0.10),
    }

    st.subheader("Run")
    run = st.button("🚀 Generate AI Insights + Top-K Plans")
    run_quantum = st.button("⚛️ Run Quantum Roads (QAOA)")

# =========================
# AI-like scoring (Explainable) + deficits per project (KEEP)
# =========================
df_city["PopW"] = df_city["Population"] / df_city["Population"].max()
df_city["DamageW"] = df_city["DamagePct"] / 100.0
df_city["ServiceGap"] = 1.0 - (df_city["ServiceAvail"] / 5.0)

df_city["NeedScore"] = (0.40*df_city["DamageW"] + 0.28*df_city["PopW"] + 0.12*df_city["ServiceGap"] + 0.20*df_city["DisplacedRatio"]).clip(0,1)

def norm_series(s):
    mx = max(1.0, float(s.max()))
    return (s / mx).clip(0,1)

df_city["HospShortW"] = norm_series(df_city["Hospitals_Shortage"])
df_city["SchoolShortW"] = norm_series(df_city["Schools_Shortage"])
df_city["HouseShortW"] = norm_series(df_city["Housing_Shortage"])

df_city["HousingDef"]  = (0.35*df_city["DamageW"] + 0.30*df_city["DisplacedRatio"] + 0.20*df_city["PopW"] + 0.15*df_city["HouseShortW"]).clip(0,1)
df_city["HospitalDef"] = (0.40*df_city["DamageW"] + 0.25*df_city["ServiceGap"] + 0.15*df_city["PopW"] + 0.20*df_city["HospShortW"]).clip(0,1)
df_city["SchoolDef"]   = (0.25*df_city["DamageW"] + 0.25*df_city["PopW"] + 0.25*df_city["DisplacedRatio"] + 0.25*df_city["SchoolShortW"]).clip(0,1)
df_city["InfraDef"]    = (0.55*df_city["DamageW"] + 0.45*df_city["ServiceGap"]).clip(0,1)
df_city["RoadDef"]     = (0.55*df_city["DamageW"] + 0.35*df_city["ServiceGap"] + 0.10*(1-df_city["RoadIndex_Base"])).clip(0,1)
df_city["WaterSanDef"] = (0.55*df_city["DamageW"] + 0.45*df_city["ServiceGap"]).clip(0,1)
df_city["PowerDef"]    = (0.65*df_city["DamageW"] + 0.35*df_city["ServiceGap"]).clip(0,1)
df_city["PublicDef"]   = (0.20*df_city["DamageW"] + 0.40*df_city["PopW"] + 0.40*df_city["DisplacedRatio"]).clip(0,1)

DEF_COL = {
    "Housing": "HousingDef",
    "Hospitals": "HospitalDef",
    "Schools": "SchoolDef",
    "Infrastructure": "InfraDef",
    "Roads": "RoadDef",
    "Water & Sanitation": "WaterSanDef",
    "Power Grid": "PowerDef",
    "Public Spaces": "PublicDef",
}

PROJECT_META = {
    "Housing": {"unit_cost": 18, "unit_time": 10},
    "Hospitals": {"unit_cost": 40, "unit_time": 16},
    "Schools": {"unit_cost": 22, "unit_time": 12},
    "Infrastructure": {"unit_cost": 30, "unit_time": 14},
    "Roads": {"unit_cost": 16, "unit_time": 10},
    "Water & Sanitation": {"unit_cost": 22, "unit_time": 12},
    "Power Grid": {"unit_cost": 32, "unit_time": 14},
    "Public Spaces": {"unit_cost": 10, "unit_time": 8},
}

# =========================
# Quantum-inspired planner (Top-K variants) (KEEP)
# =========================
def plan_variant_weights(name: str):
    if name == "Plan A — Max Impact":
        return {"w_impact": 0.70, "w_speed": 0.15, "w_fair": 0.15}
    if name == "Plan C — Fairness First":
        return {"w_impact": 0.45, "w_speed": 0.10, "w_fair": 0.45}
    return {"w_impact": 0.55, "w_speed": 0.15, "w_fair": 0.30}

def generate_plan(df_in: pd.DataFrame, types: list, total_budget_m: int, horizon_m: int, weights: dict):
    dfp = df_in.copy()
    wI, wS, wF = weights["w_impact"], weights["w_speed"], weights["w_fair"]

    PHASE_ALLOWED = {
        0: {"Housing", "Hospitals", "Water & Sanitation"},
        1: {"Housing", "Hospitals", "Schools", "Roads", "Infrastructure", "Water & Sanitation"},
        2: {"Schools", "Roads", "Infrastructure", "Power Grid", "Public Spaces", "Water & Sanitation"}
    }

    phase_split = [0.45, 0.35, 0.20]
    phase_names = ["Phase 1 — Emergency Recovery", "Phase 2 — Core Services", "Phase 3 — Long-Term Urban Recovery"]

    MIN_ZONE_COVERAGE_RATIO = [0.65, 0.85, 1.00]
    MAX_ACTIONS_PER_ZONE_PER_PHASE = [2, 2, 2]
    MAX_REPEAT_SAME_TYPE_IN_ZONE_TOTAL = 1
    ZONE_REPEAT_PENALTY = 0.22
    DIMINISHING_RETURNS = 0.35

    zone_total_count = {z: 0 for z in dfp["Zone"].tolist()}
    zone_phase_count = {z: 0 for z in dfp["Zone"].tolist()}
    zone_type_used_total = set()

    candidates = []
    for _, row in dfp.iterrows():
        zone = row["Zone"]
        need = float(row["NeedScore"])
        popw = float(row["PopW"])

        for t in types:
            if t not in DEF_COL:
                continue
            deficit = float(row[DEF_COL[t]])

            meta = PROJECT_META[t]
            cost, ttime = meta["unit_cost"], meta["unit_time"]

            impact = need * deficit * (0.55 + 0.45 * popw)
            speed = 1.0 / max(1.0, ttime)
            base_score = (wI * impact) + (wS * speed)

            candidates.append([base_score, impact, speed, zone, t, cost, ttime, deficit, need])

    candidates.sort(key=lambda x: x[0], reverse=True)

    phases = []
    for p_idx, p_ratio in enumerate(phase_split):
        allowed_types = PHASE_ALLOWED[p_idx].intersection(set(types))
        phase_budget = int(total_budget_m * p_ratio)
        phase_time = int(horizon_m * p_ratio)
        local_budget = phase_budget

        for z in zone_phase_count:
            zone_phase_count[z] = 0

        picks = []
        covered = set()
        target_cover = int(np.ceil(len(dfp) * MIN_ZONE_COVERAGE_RATIO[p_idx]))

        def try_pick(pass_mode="coverage"):
            nonlocal local_budget, picks, covered

            for base_score, impact, speed, zone, t, cost, ttime, deficit, need in candidates:
                if local_budget <= 0:
                    break
                if t not in allowed_types:
                    continue
                if cost > local_budget:
                    continue
                if ttime > phase_time + 8:
                    continue

                if (zone, t) in zone_type_used_total and MAX_REPEAT_SAME_TYPE_IN_ZONE_TOTAL <= 1:
                    continue

                if zone_phase_count[zone] >= MAX_ACTIONS_PER_ZONE_PER_PHASE[p_idx]:
                    continue

                if pass_mode == "coverage" and zone in covered:
                    continue

                if p_idx == 0 and deficit < 0.35:
                    continue

                fairness_boost = wF * (1.0 / (1 + zone_total_count[zone]))
                repeat_penalty = ZONE_REPEAT_PENALTY * zone_total_count[zone]
                dim_penalty = DIMINISHING_RETURNS * max(0, zone_total_count[zone] - 1)

                final_score = base_score + fairness_boost - repeat_penalty - dim_penalty

                picks.append({
                    "Plan": "",
                    "Phase": phase_names[p_idx],
                    "Zone": zone,
                    "ProjectType": t,
                    "EstCost_M$": cost,
                    "EstTime_wks": ttime,
                    "NeedScore": round(need, 3),
                    "Deficit": round(deficit, 3),
                    "ImpactScore": round(impact, 4),
                    "SpeedScore": round(speed, 4),
                    "FairnessBoost": round(fairness_boost, 4),
                    "ZonePenalty": round(repeat_penalty + dim_penalty, 4),
                    "FinalScore": round(final_score, 4),
                })

                covered.add(zone)
                zone_total_count[zone] += 1
                zone_phase_count[zone] += 1
                zone_type_used_total.add((zone, t))
                local_budget -= cost

                if pass_mode == "coverage" and len(covered) >= target_cover:
                    break

        try_pick("coverage")
        try_pick("fill")

        phases.append({
            "name": phase_names[p_idx],
            "budget": phase_budget,
            "time": phase_time,
            "actions": picks,
            "remaining": local_budget
        })

    plan_df = pd.DataFrame([a for ph in phases for a in ph["actions"]])
    return phases, plan_df

def compute_metrics(plan_df: pd.DataFrame, total_budget_m: int):
    if plan_df.empty:
        return {
            "TotalImpact": 0.0,
            "ZonesCovered": 0,
            "TotalCost": 0,
            "BudgetUsedPct": 0.0,
            "FairnessIndex": 0.0,
            "AvgTime": 0.0,
        }
    total_impact = float(plan_df["ImpactScore"].sum())
    zones_covered = int(plan_df["Zone"].nunique())
    total_cost = int(plan_df["EstCost_M$"].sum())
    budget_used_pct = 100.0 * total_cost / max(1, total_budget_m)

    counts = plan_df["Zone"].value_counts().values
    if len(counts) <= 1:
        fairness = 1.0
    else:
        fairness = float(1.0 - (np.std(counts) / max(1e-9, np.mean(counts))))
        fairness = float(np.clip(fairness, 0, 1))

    avg_time = float(plan_df["EstTime_wks"].mean())
    return {
        "TotalImpact": round(total_impact, 4),
        "ZonesCovered": zones_covered,
        "TotalCost": total_cost,
        "BudgetUsedPct": round(budget_used_pct, 1),
        "FairnessIndex": round(fairness, 3),
        "AvgTime": round(avg_time, 2),
    }

# =========================================================
# ------------------- (NEW PART) QUANTUM ROADS QAOA -------------------
# =========================================================
def run_quantum_roads_pipeline(q_budget, q_lambda, q_gamma, q_beta, q_weights):
    # Stage 1-3 (same as quantum pipeline)
    df_roads = load_road_data()
    df_roads = engineer_context_features(df_roads)
    df_roads = compute_impact_scores(df_roads, q_weights)

    # Stage 4 QUBO
    Q = build_qubo(df_roads, q_budget, q_lambda)

    # Stage 5 QAOA
    qc, gamma, beta = build_qaoa_circuit(Q)
    best_bit, best_energy, counts = run_qaoa_and_extract_solution(
        qc=qc,
        gamma=gamma,
        beta=beta,
        params={"gamma": q_gamma, "beta": q_beta},
        Q=Q
    )

    # Stage 6 Plan
    df_roads, summary = generate_recovery_plan(df_roads, best_bit)

    # Stage 7 Map
    gaza_map = visualize_gaza_dashboard(df_roads)

    return df_roads, summary, gaza_map, best_energy, counts

# Keep results in session (so it doesn't disappear when switching tabs)
if "qaoa_ready" not in st.session_state:
    st.session_state.qaoa_ready = False
    st.session_state.df_roads = None
    st.session_state.q_summary = None
    st.session_state.q_map = None
    st.session_state.q_energy = None
    st.session_state.q_counts = None

if run_quantum:
    with st.spinner("⚛️ Running QAOA for Roads..."):
        df_roads, q_summary, q_map, q_energy, q_counts = run_quantum_roads_pipeline(
            q_budget=q_budget,
            q_lambda=q_lambda,
            q_gamma=q_gamma,
            q_beta=q_beta,
            q_weights=q_weights
        )
    st.session_state.qaoa_ready = True
    st.session_state.df_roads = df_roads
    st.session_state.q_summary = q_summary
    st.session_state.q_map = q_map
    st.session_state.q_energy = q_energy
    st.session_state.q_counts = q_counts

# =========================================================
# Tabs (KEEP + add Quantum tab)
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏙️ Overview", "🤖 AI Insights", "🧱 Top-K Plans", "🗺️ Visuals", "⚛️ Quantum Roads"]
)

# ---- Tab 1 (KEEP)
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='kpi'>Zones<br><h2 style='color:var(--accent);margin:0;'>{len(df_city)}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi'>Total Population<br><h2 style='color:var(--accent);margin:0;'>{df_city['Population'].sum():,}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi'>Avg Damage<br><h2 style='color:var(--accent);margin:0;'>{df_city['DamagePct'].mean():.1f}%</h2></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi'>Budget<br><h2 style='color:var(--accent);margin:0;'>${total_budget}M</h2></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><b>Gaza Zones + Planning Capacity (counts & shortages)</b><br><span class='small-muted'>Includes approximate baseline counts and post-war shortage estimates for hospitals, schools, and housing.</span></div>", unsafe_allow_html=True)

    show_cols = [
        "Zone","Population","DamagePct","ServiceAvail","DisplacedRatio","NeedScore",
        "Hospitals_Base","Hospitals_Target","Hospitals_Shortage",
        "Schools_Base","Schools_Target","Schools_Shortage",
        "HousingUnits_Base","HousingUnits_Target","Housing_Shortage"
    ]
    st.dataframe(
    df_city[show_cols].sort_values(show_cols[1], ascending=False),
    height=360
)
    
# ---- Tab 2 (KEEP)
with tab2:
    st.markdown("<div class='card'><b>AI Need & Deficit Scoring</b><br><span class='small-muted'>Explainable scoring using damage, population, service gap, displacement, and shortages.</span></div>", unsafe_allow_html=True)

    st.markdown("""
<div class="card">
<b>Where is the AI?</b><br><br>
The AI layer computes a <b>Need Score</b> for each zone using:
<ul>
<li>Damage severity</li>
<li>Population size</li>
<li>Service availability gap</li>
<li>Displacement ratio</li>
</ul>
These scores represent <b>human urgency</b> rather than just infrastructure loss.
</div>

<div class="card">
<b>Where is the Quantum?</b><br><br>
We use a QUBO formulation and run <b>QAOA</b> (quantum algorithm)
to select which <b>roads</b> to rebuild under a budget constraint.
The output is visualized on an interactive Gaza map.
</div>
""", unsafe_allow_html=True)

    colA, colB = st.columns([1.1, 1])
    with colA:
        st.subheader("Overall Need Score (by zone)")
        st.bar_chart(df_city.set_index("Zone")["NeedScore"])

    with colB:
        st.subheader("Project Deficits (selected types)")
        show_cols = ["Zone"]
        for t in project_types:
            show_cols.append(DEF_COL[t])
        st.dataframe(
        df_city[show_cols].sort_values(show_cols[1], ascending=False),
        height=360
        )

# ---- Tab 3 (KEEP)
with tab3:
    if not run:
        st.info("اختاري project types والbudget ثم اضغطي **Generate** من الsidebar.")
    else:
        st.markdown("<div class='card'><b>Top-K Recovery Plans</b><br><span class='small-muted'>We generate multiple strategies so decision-makers can choose under uncertainty.</span></div>", unsafe_allow_html=True)

        plan_names = ["Plan A — Max Impact", "Plan B — Balanced", "Plan C — Fairness First"][:k_plans]
        plans = {}
        metrics_rows = []

        for pname in plan_names:
            w = plan_variant_weights(pname)
            phases, plan_df = generate_plan(df_city, project_types, total_budget, horizon_months, w)
            if not plan_df.empty:
                plan_df["Plan"] = pname
            plans[pname] = {"phases": phases, "df": plan_df, "weights": w}
            m = compute_metrics(plan_df, total_budget)
            m["Plan"] = pname
            m["Weights"] = f"I:{w['w_impact']:.2f}  S:{w['w_speed']:.2f}  F:{w['w_fair']:.2f}"
            metrics_rows.append(m)

        metrics_df = pd.DataFrame(metrics_rows)[["Plan","Weights","TotalImpact","ZonesCovered","TotalCost","BudgetUsedPct","FairnessIndex","AvgTime"]]
        st.subheader("Plan Comparison (Key Metrics)")
        st.dataframe(metrics_df)

        st.divider()

        chosen = st.selectbox("Select a plan to view details", plan_names, index=1 if len(plan_names) > 1 else 0)
        chosen_plan = plans[chosen]
        plan_df = chosen_plan["df"]
        phases = chosen_plan["phases"]
        w = chosen_plan["weights"]

        m = compute_metrics(plan_df, total_budget)
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f"<div class='kpi'>Total Impact<br><h2 style='color:var(--good);margin:0;'>{m['TotalImpact']}</h2></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi'>Zones Covered<br><h2 style='color:var(--good);margin:0;'>{m['ZonesCovered']}/{len(df_city)}</h2></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi'>Budget Used<br><h2 style='color:var(--good);margin:0;'>{m['BudgetUsedPct']}%</h2></div>", unsafe_allow_html=True)
        k4.markdown(f"<div class='kpi'>Fairness Index<br><h2 style='color:var(--good);margin:0;'>{m['FairnessIndex']}</h2></div>", unsafe_allow_html=True)

        st.markdown(
            f"<div class='card'>"
            f"<b>Plan Strategy Weights</b><br>"
            f"<span class='small-muted'>Impact={w['w_impact']:.2f}, Speed={w['w_speed']:.2f}, Fairness={w['w_fair']:.2f}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        for ph in phases:
            st.markdown(f"### {ph['name']}")
            st.markdown(
                f"<div class='card'>"
                f"<b>Phase Budget:</b> ${ph['budget']}M &nbsp;&nbsp; | &nbsp;&nbsp; "
                f"<b>Phase Time:</b> ~{ph['time']} months &nbsp;&nbsp; | &nbsp;&nbsp; "
                f"<b>Remaining:</b> ${ph['remaining']}M"
                f"</div>",
                unsafe_allow_html=True
            )

            if ph["actions"]:
                df_phase = pd.DataFrame(ph["actions"]).sort_values("FinalScore", ascending=False)
                st.dataframe(
                    df_phase[["Zone","ProjectType","EstCost_M$","EstTime_wks","NeedScore","Deficit","ImpactScore","FairnessBoost","FinalScore"]],
                    height=280
                )

                top = df_phase.iloc[0]
                st.markdown(
                    f"<div class='card'>"
                    f"<b>Why this top action?</b><br>"
                    f"Chosen <b>{top['ProjectType']}</b> in <b>{top['Zone']}</b> because: "
                    f"<ul style='color:var(--ink);'>"
                    f"<li>High zone need (NeedScore={top['NeedScore']})</li>"
                    f"<li>High deficit for this service (Deficit={top['Deficit']})</li>"
                    f"<li>Optimizes the selected strategy (Impact/Speed/Fairness)</li>"
                    f"</ul>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.warning("No actions selected in this phase under current constraints. Try increasing budget or selecting fewer project types.")

        st.divider()
        st.subheader("Export Selected Plan")
        st.download_button(
            "Download Selected Plan CSV",
            plan_df.to_csv(index=False).encode("utf-8"),
            file_name=f"phoenix_rebuildiq_{chosen.replace(' ','_').replace('—','-')}.csv",
            mime="text/csv"
        )

# ---- Tab 4 (KEEP)
with tab4:
    st.subheader("Project Visuals (for PPT & Demo Screenshots)")
    st.markdown("<div class='card'><span class='small-muted'>Use these visuals to strengthen storytelling and make the demo memorable.</span></div>", unsafe_allow_html=True)

    if show_masterplan:
        colV1, colV2 = st.columns(2)
        with colV1:
            st.markdown("#### Master Plan (Urban Planning Report Style)")
            show_image_if_exists(ASSETS["masterplan"], caption="Master Plan View — Recovery Layout")
        with colV2:
            st.markdown("#### Blueprint View")
            show_image_if_exists(ASSETS["blueprint"], caption="Blueprint-style Planning Map")

        colV3, colV4 = st.columns(2)
        with colV3:
            st.markdown("#### Zone Map (Need Heat Zones)")
            show_image_if_exists(ASSETS["zone_map"], caption="Zone Map — Priority Areas")
        with colV4:
            st.markdown("#### Recovery Timeline")
            show_image_if_exists(ASSETS["timeline"], caption="Phased Recovery Timeline (1–3)")

    st.markdown("<div class='card'><b>Screenshot checklist:</b> Overview table + AI bar chart + Plan comparison + Selected plan Phase 1 + Quantum roads map.</div>", unsafe_allow_html=True)

# ---- Tab 5 (Quantum Roads — DOES NOT REMOVE ANYTHING)
with tab5:
    st.markdown(
        "<div class='card'>"
        "<b>⚛️ Quantum Roads — QAOA Selection</b><br>"
        "<span class='small-muted'>"
        "This tab runs a QUBO + QAOA pipeline to decide which roads to rebuild first "
        "under a budget constraint. This is an additional quantum layer on top of the city-level plans."
        "</span>"
        "</div>",
        unsafe_allow_html=True
    )

    # Safety check: session state
    if "qaoa_ready" not in st.session_state or not st.session_state.qaoa_ready:
        st.info("**⚛️ Run Quantum Roads (QAOA)** from the sidebar to execute the quantum pipeline and see results here.")
    else:
        df_roads   = st.session_state.df_roads
        q_summary  = st.session_state.q_summary
        q_map      = st.session_state.q_map
        q_energy   = st.session_state.q_energy

        # -------------------------
        # KPIs
        # -------------------------
        c1, c2, c3 = st.columns(3)

        c1.markdown(
            f"<div class='kpi'>Selected Roads<br>"
            f"<h2 style='color:var(--good);margin:0;'>{len(q_summary['selected_roads'])}</h2>"
            f"</div>",
            unsafe_allow_html=True
        )

        c2.markdown(
            f"<div class='kpi'>Total Cost<br>"
            f"<h2 style='color:var(--accent);margin:0;'>"
            f"${q_summary['total_cost']:.1f}M"
            f"</h2></div>",
            unsafe_allow_html=True
        )

        c3.markdown(
            f"<div class='kpi'>QAOA Energy<br>"
            f"<h2 style='color:var(--warn);margin:0;'>"
            f"{q_energy:.3f}"
            f"</h2></div>",
            unsafe_allow_html=True
        )

        st.divider()

        # -------------------------
        # Table View
        # -------------------------
        st.subheader("📋 Road-Level Quantum Decisions")

        display_roads = df_roads.copy()
        display_roads["selected"] = display_roads["selected"].map(
            {1: "✅ Rebuild", 0: "⏸️ Deferred"}
        )

        show_cols = [
            "id",
            "selected",
            "impact",
            "final_cost",
            "population",
            "damage"
        ]

        st.dataframe(
            display_roads[show_cols].sort_values("impact", ascending=False),
            use_container_width=True,
            height=380
        )

        st.divider()

        # -------------------------
        # Map View
        # -------------------------
        st.subheader("🗺️ Quantum-Selected Roads Map")

        st.components.v1.html(
            q_map._repr_html_(),
            height=600,
            scrolling=False
        )

        st.markdown(
            "<div class='card'>"
            "<b>How to interpret this:</b><br>"
            "<ul style='color:var(--ink);'>"
            "<li>Each road is a binary decision variable</li>"
            "<li>QUBO encodes impact maximization + budget penalty</li>"
            "<li>QAOA searches for a low-energy (high-quality) solution</li>"
            "</ul>"
            "</div>",
            unsafe_allow_html=True
        )