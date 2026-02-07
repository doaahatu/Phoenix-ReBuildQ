# 🐦‍🔥 Phoenix | ReBuildQ  
### Quantum–AI Assisted Urban Reconstruction Planning for Gaza 🇵🇸

Phoenix | ReBuildQ is a **decision-support platform** developed for the  
**NYUAD International Hackathon for Social Good**.  
The project focuses on **post-war reconstruction planning in Gaza**, combining:

- 🧠 Explainable AI for human-need and deficit analysis  
- ⚛️ Quantum Computing (QAOA) for constrained optimization  
- 🏙️ Urban planning logic for phased, fair recovery  
- 🗺️ Interactive visualization for transparent decision-making  

---

## 🌍 Problem Statement

Post-conflict reconstruction in Gaza faces severe challenges:

- Massive urban and infrastructure damage  
- Limited budgets and resources  
- Unequal service availability between zones  
- High population displacement  
- Difficult prioritization decisions  

Traditional planning methods struggle to balance **impact**, **speed**, and **fairness** under these constraints.

---

## 💡 Proposed Solution

Phoenix | ReBuildQ introduces a **hybrid Quantum–AI reconstruction engine** that:

1. Computes **Need Scores** for each Gaza zone using explainable AI  
2. Models reconstruction decisions as an optimization problem  
3. Uses **Quantum Approximate Optimization Algorithm (QAOA)** to select optimal actions  
4. Generates **multiple high-quality recovery plans**  
5. Provides **visual and explainable outputs** for decision-makers  

---

## 🧠 Explainable AI Layer

The AI layer evaluates each zone based on:

- Damage percentage  
- Population size  
- Service availability gap  
- Displacement ratio  
- Infrastructure shortages (housing, schools, hospitals, roads)

All scores are **transparent and interpretable**, enabling trust in the system’s recommendations.

---

## ⚛️ Quantum Optimization Layer

The quantum component focuses on **road reconstruction prioritization**:

- Roads are encoded as binary decision variables  
- Costs and humanitarian impact form a **QUBO problem**  
- **QAOA** explores optimal subsets under budget constraints  
- Executed using **Qiskit Aer simulator**

This demonstrates practical near-term quantum optimization for social good.

---

## 🏗️ Urban Planning Strategy

Reconstruction is organized into **three phases**:

### Phase 1 — Emergency Recovery
- Housing
- Hospitals
- Water & sanitation

### Phase 2 — Core Services
- Schools
- Roads
- Infrastructure

### Phase 3 — Long-Term Recovery
- Power grid
- Public spaces
- Urban resilience projects

Fairness constraints ensure balanced service distribution across all zones.

---

## 🗺️ Platform Features

- 📊 Gaza-wide statistics (population, damage, need scores)  
- 🤖 AI-based insights with explainability  
- 🧱 Top-K reconstruction plans  
- ⚛️ Quantum road selection (QAOA)  
- 🗺️ Interactive reconstruction map  
- 📥 Exportable plans (CSV)  
- 🎨 Visual assets for storytelling and presentations  

Built using **Streamlit** for clarity and rapid prototyping.

---

## 🧩 Project Structure

```text
Phoenix-ReBuildQ/
│
├── app.py                     # Main Streamlit application
│
├── quantum/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── impact_scoring.py
│   ├── qubo.py
│   ├── qaoa_solver.py
│   └── plan_builder.py
│
├── visualization/
│   └── map_view.py
│
├── assets/
│   ├── hero_gaza.png
│   ├── masterplan_realistic.png
│   ├── blueprint.png
│   ├── zone_map.png
│   └── timeline.png
│
├── requirements.txt
└── README.md
