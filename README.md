# Materials_Informatics_Project
Machine Learning for Electronic Phase Classification (notebook works well as intended) and ML-Based Detection of Semimetals (needs better and improved approach)
-- Machine Learning for Electronic Phase Classification  
## (From First-Principles Materials Data)

---

# 1.  Objective

The goal of this project is to:

> Build a machine learning model that can classify materials as **metallic or non-metallic** using first-principles (DFT-derived) data, and extract **physical insight** from the model.

---

# 2.  Dataset

- Source: Materials Project (MatPES r2SCAN dataset) (https://materialsproject-contribs.s3.amazonaws.com/index.html#MatPES_2025_1/)
- Size used: ~20,000 structures
- Each entry contains:
  - Band gap (eV)
  - Total energy
  - Cohesive energy
  - Magnetic moments
  - Atomic structure (composition + geometry)

---

# 3.  Feature Engineering (Physics → ML)

Instead of raw data, we constructed **physically meaningful descriptors**:

| Feature | Meaning |
|--------|--------|
| `EN_range` | Electronegativity difference → ionic vs covalent bonding |
| `Z_mean` | Average atomic number → band width / relativistic effects |
| `Z_max` | Heaviest element → SOC contribution |
| `Z_var` | Chemical diversity |
| `SOC_proxy` | Approximate spin–orbit coupling strength |
| `magmom` | Magnetism / exchange splitting |
| `num_elements` | System complexity |

- This step is critical:  
We translated **materials physics into machine-learnable features**

---

# 4. Handling Imbalanced Data

- ~91% metals  
- ~9% non-metals  

To fix bias, we used: scale_pos_weight ≈ 10.5

- This ensures the model **does not ignore rare insulating systems**

---

# 5.  Model

- Algorithm: XGBoost (Gradient Boosted Trees)
- Learns nonlinear relationships between features and electronic phase

---

# 6. Model Performance

| Metric | Value |
|------|------|
| Accuracy | 93% |
| ROC-AUC | 0.946 |

### Class-wise:

| Class | Precision | Recall |
|------|----------|--------|
| Metal | 0.98 | 0.94 |
| Non-metal | 0.55 | 0.84 |

---

##  Interpretation

- High recall (0.84) → model **successfully detects insulating systems**
- Lower precision (0.55) → model **over-predicts insulators slightly**

- This is acceptable because:
> Missing rare insulating phases is worse than over-predicting them

---

# 7. SHAP Analysis (Core Insight)

SHAP tells us:

> **Which physical features drive the model’s decisions**

---

## Feature Importance (from SHAP)

| Feature | Importance |
|--------|-----------|
| EN_range | 1.47 |
| Z_mean | 0.92 |
| magmom | 0.79 |
| Z_max | 0.78 |
| SOC_proxy | 0.75 |
| Z_var | 0.56 |
| num_elements | 0.15 |

---

# 8. Physical Interpretation

## 1. Electronegativity Difference (EN_range)

- Most important feature

**Physics:**
- Large Δχ → strong charge transfer
- Leads to:
  - electron localization
  - band gap opening

**Conclusion:**
> Ionic bonding strongly favors insulating behavior

---

##  2. Atomic Number (Z_mean)

**Physics:**
- Heavier atoms → more delocalized orbitals
- Wider bands → metallicity

**Conclusion:**
> Heavy-element systems tend to be metallic

---

## 3. Magnetism (magmom)

**Physics:**
- Exchange splitting can open gaps
- Magnetic ordering affects band structure

**Conclusion:**
> Magnetism contributes to electronic phase transitions

---

## 4. Spin–Orbit Coupling (SOC_proxy)

**Physics:**
- SOC can:
  - induce band inversion
  - create topological phases

**Observation:**
- Important but **not dominant**

**Conclusion:**
> SOC is secondary compared to chemical bonding in this dataset

---

##  Overall Physical Picture

The model reveals a **competition between two effects**:

### 1. Ionicity (localization)
- Driven by EN_range
- Promotes insulating states

### 2. Delocalization (metallicity)
- Driven by Z_mean and band width
- Promotes metallic states

---

# 9. Interpretation of SHAP Plots

## SHAP Dependence Plots

Each plot shows:

- X-axis → feature value  
- Y-axis → contribution to prediction  

---

## EN_range Plot

Look for:
- Increasing trend → higher EN → insulating

- Interpretation:
> Strong ionic character opens a gap

---

## SOC_proxy Plot

Look for:
- nonlinear behavior  
- clustering  

- Interpretation:
> SOC influences band structure but is not dominant here

---

## magmom Plot

Look for:
- separation between phases  

- Interpretation:
> Magnetic exchange contributes to gap formation

---

## Interaction Plot (EN_range vs SOC)

- Shows interplay:

> Ionic bonding vs relativistic effects

---

# 10.  Key Scientific Result

> Electronic phase classification is primarily governed by **chemical ionicity**, with secondary contributions from **atomic weight and magnetism**, while **spin–orbit coupling plays a smaller role** in this dataset.

---

# 11.  What This Means

We have:

✅ Built a working ML model  
✅ Extracted interpretable physics  
✅ Identified dominant mechanisms  

---


# 12. Final Perspective

This project demonstrates:

> How machine learning can be used not just for prediction, but for discovering **underlying physical principles** in materials science.

---

# One-line Summary

> We built a physics-informed ML model that reveals **ionicity dominates electronic phase behavior**, while SOC and magnetism play secondary roles, setting the stage for topology-aware materials discovery.

---
