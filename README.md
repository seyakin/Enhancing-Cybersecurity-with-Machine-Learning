# Enhancing Cybersecurity with Machine Learning: Intrusion Detection System

> **Based on:** Oluwatobi O.A., Ajani W.A., Salim A.O., Akinmuda O.A., Israel O. (2025).
> *Enhancing Cybersecurity with Machine Learning: Development and Evaluation of Intrusion Detection Systems.*
> European Journal of Computer Science and Information Technology, 13(51), 138–173.
> DOI: https://doi.org/10.37745/ejcsit.2013/vol13n51138173

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
   - [Data Preprocessing](#1-data-preprocessing)
   - [Handling Class Imbalance](#2-handling-class-imbalance-smote)
   - [Dimensionality Reduction](#3-dimensionality-reduction-pca)
   - [Model Selection](#4-model-selection)
   - [Evaluation Metrics](#5-evaluation-metrics)
4. [Results Summary](#results-summary)
5. [Project Structure](#project-structure)
6. [Requirements & Usage](#requirements--usage)
7. [Citation](#citation)

---

## Project Overview

The widespread adoption of digital networks has led to a surge in sophisticated cyber threats
including malware, phishing, denial-of-service (DoS) attacks, ransomware, and advanced
persistent threats (APTs). Traditional rule-based security systems are increasingly ineffective
against these evolving threats — often failing to detect novel attack patterns and generating
excessive false positives and missed detections (Oluwatobi et al., 2025).

This project implements and evaluates five machine learning models for **binary network
intrusion detection** using the **UNSW-NB15** benchmark dataset. Machine learning is adopted
because, unlike static rule-based systems, ML algorithms can analyze large volumes of network
traffic, identify anomalies, and adapt to emerging attack patterns without requiring manual
rule updates (Manoharan & Sarker, 2023, as cited in Oluwatobi et al., 2025).

---

## Dataset

The **UNSW-NB15** dataset was created by the IXIA PerfectStorm tool in the Cyber Range Lab
of UNSW Canberra. It captures a hybrid of real modern normal activities and synthetic
contemporary attack behaviour (Oluwatobi et al., 2025).

| Split        | Records   |
|--------------|-----------|
| Training set | 175,341   |
| Testing set  | 82,332    |
| **Total**    | **257,673** |

The dataset contains **49 features** covering both categorical and numerical network traffic
attributes, including source/destination IP addresses, port numbers, transaction protocols,
packet sizes, and connection state information. The target variable (`label`) is binary:
`0` for normal traffic and `1` for attack traffic.

**Attack categories present in the dataset:**

| Category       | Description |
|----------------|-------------|
| Normal         | Benign network activity |
| Generic        | Largest malicious category |
| Exploits       | Attempts to leverage system vulnerabilities |
| Fuzzers        | Malformed input attacks |
| Reconnaissance | Network scanning and probing |
| DoS            | Denial-of-service floods |
| Backdoors      | Persistent unauthorised access |
| Analysis       | Port/vulnerability scanning variants |
| Shellcode      | Code injection attacks |
| Worms          | Self-replicating malware |

---

## Methodology

The methodology follows the pipeline described in Oluwatobi et al. (2025), Section 3.

### 1. Data Preprocessing

Raw data is loaded and the feature matrix (`X`) is separated from the target variable (`y`).
Categorical columns (`proto`, `service`, `state`, `attack_cat`) are encoded using
**Label Encoding**. Numerical features are then standardised using **StandardScaler**
(zero mean, unit variance), which is critical for distance-based models such as SVM and KNN
that are sensitive to features at different scales (Oluwatobi et al., 2025).

Highly correlated features identified from the correlation matrix heatmap are dropped prior
to modelling to reduce redundancy:

```
sloss, ct_srv_src, dloss, ct_srv_dst, ct_src_ltm,
sinpkt, ct_dst_ltm, spkts, dpkts, dwin, swin, synack, ct_dst_sport_ltm
```

### 2. Handling Class Imbalance — SMOTE

The dataset exhibits a slight class imbalance between attack and normal instances. To prevent
model bias toward the majority class, the **Synthetic Minority Over-sampling Technique
(SMOTE)** is applied to the training data (Oluwatobi et al., 2025, Section 3 — *Handling
Class Imbalance*). SMOTE generates synthetic instances for under-represented classes by
interpolating between existing minority-class samples, ensuring all classes are equally
represented during training.

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

### 3. Dimensionality Reduction — PCA

**Principal Component Analysis (PCA)** is applied as a preprocessing step to reduce feature
dimensionality while retaining the most informative components of the data. As described in
Oluwatobi et al. (2025), PCA was initially applied retaining **95% of variance**, and further
experimented with **70% variance** retention to simplify the dataset. This project uses
**50% variance** retention as a practical balance for computational efficiency. PCA both
reduces the risk of overfitting and accelerates training across all models.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)
```

### 4. Model Selection

Following Oluwatobi et al. (2025, Section 3 — *Model Selection*), five classifiers with
distinct algorithmic foundations are implemented and compared:

#### Support Vector Machine (SVM)
SVM finds an optimal hyperplane that maximises the margin between classes in a
high-dimensional feature space. It employs the **RBF (radial basis function) kernel**
to transform the input into a higher-dimensional space, making non-linearly separable
problems integrable. The decision function is:

```
D(x) = w·φ(x) + b
```

where `φ(x)` transforms data into a new M-dimensional space (Oluwatobi et al., 2025).

#### K-Nearest Neighbors (KNN)
KNN classifies each instance based on the majority label among its `k` nearest neighbours
in feature space. It is a non-parametric, instance-based method. Despite its simplicity,
the paper reports KNN achieving the **highest accuracy of 94.69%** on this dataset.

#### Artificial Neural Network (ANN)
A feed-forward ANN with the following architecture:
- Input layer → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
- Compiled with `Adam` optimiser and `binary_crossentropy` loss
- Trained for 10 epochs with batch size 32

The paper reports ANN accuracy of **95%**, with steady training improvement from
83.13% (Epoch 1) to 94.30% (Epoch 10) (Oluwatobi et al., 2025, Figure 17).

#### Decision Tree
A supervised tree-structured classifier that recursively partitions the feature space using
information gain. Split criterion is based on entropy:

```
Entropy(S) = -Σ P(x) · log₂ P(x)
Gain(S, A) = Entropy(S) - Σ (|Sv|/|S|) × Entropy(Sv)
```

(Oluwatobi et al., 2025, Section 2 — *Decision Tree*). `max_depth=10` is applied to
prevent overfitting.

#### Random Forest
An ensemble of decision trees trained on bootstrap samples of the data. Each tree votes
on the predicted class, and the majority vote determines the final output. The generalisation
error upper bound is given by:

```
PE* ≤ ρ̄(1 - s²) / s²
```

where `ρ̄` is the average correlation between trees and `s` is the strength of individual
trees. Crucially, as more trees are added, the model does **not overfit** (Oluwatobi et al.,
2025, Section 2 — *Random Forest*). Configured with `n_estimators=50, max_depth=10`.

#### Soft Voting Ensemble
All four classical classifiers are combined into a **soft voting ensemble** that averages
predicted class probabilities. This approach leverages the complementary strengths of
each model to produce more robust predictions.

### 5. Evaluation Metrics

Models are evaluated using four metrics as defined in Oluwatobi et al. (2025, Section 3 —
*Model Evaluation Metrics*):

| Metric    | Formula |
|-----------|---------|
| Accuracy  | (TP + TN) / (TP + TN + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall    | TP / (TP + FN) |
| F1 Score  | 2 × (Precision × Recall) / (Precision + Recall) |

Where **TP** = True Positives, **TN** = True Negatives, **FP** = False Positives,
**FN** = False Negatives. In cybersecurity, minimising **false negatives** (missed attacks)
is particularly critical as undetected intrusions pose direct security risks.

---

## Results Summary

Results from the referenced paper (Oluwatobi et al., 2025, Figure 22):

| Model         | Accuracy   | Precision  | Recall  | F1 Score   |
|---------------|------------|------------|---------|------------|
| KNN           | **94.69%** | **95.31%** | 93.96%  | **94.63%** |
| ANN           | 95.00%     | 94.00%     | 95.00%  | 94.50%     |
| Random Forest | 92.81%     | 93.67%     | 91.77%  | 92.71%     |
| Decision Tree | 92.22%     | 92.70%     | 91.59%  | 92.14%     |
| SVM           | 90.88%     | 92.29%     | 89.15%  | 90.69%     |
| Ensemble      | 93.81%     | 94.80%     | 92.66%  | 93.72%     |

> KNN achieved the highest accuracy with balanced precision, recall, and F1 score,
> demonstrating strong capability for classifying both attack and non-attack instances.
> Regular model retraining is recommended to address emerging attack patterns
> (Oluwatobi et al., 2025).

---

## Project Structure

```
network_ids_project/
│
├── network_ids_analysis.py           # Full ML pipeline script
├── README.md                         # This file
│
└── visuals/
    ├── fig1_attack_category_distribution.png
    ├── fig2_communication_states.png
    ├── fig3_bytes_boxplots.png
    ├── fig4_label_distribution.png
    ├── fig5_packet_distributions.png
    ├── fig6_correlation_heatmap.png
    ├── fig7_ann_training_curves.png
    ├── fig8_confusion_ann.png
    ├── fig9_confusion_svm.png
    ├── fig10_confusion_knn.png
    ├── fig11_confusion_decision_tree.png
    ├── fig12_confusion_random_forest.png
    ├── fig13_confusion_ensemble.png
    ├── fig14_model_comparison.png
    └── model_results.csv             # Performance metrics for all models
```

---

## Requirements & Usage

### Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn tensorflow
```

### Run the pipeline

```bash
# 1. Place UNSW_NB15_training-set.csv in the project root directory
# 2. Run the script:
python network_ids_analysis.py
```

> **Note:** The script includes a synthetic data generator that mirrors the UNSW-NB15
> schema for reproducibility without the original dataset. To use the real dataset,
> replace the synthetic generation block at the top of the script with:
> ```python
> df = pd.read_csv("UNSW_NB15_training-set.csv")
> ```

---

## Citation

If you use this code or reference this work, please cite the original paper:

```bibtex
@article{oluwatobi2025enhancing,
  title     = {Enhancing Cybersecurity with Machine Learning: Development and
               Evaluation of Intrusion Detection Systems},
  author    = {Ogundipe Ademola Oluwatobi and Waheed Azeez Ajani and
               Adedokun Okikiade Salim and Akinmuda Oluseye Ayobami and Odeajo Israel},
  journal   = {European Journal of Computer Science and Information Technology},
  volume    = {13},
  number    = {51},
  pages     = {138--173},
  year      = {2025},
  doi       = {10.37745/ejcsit.2013/vol13n51138173},
  publisher = {European Centre for Research Training and Development, UK}
}
```

---

*Published September 10, 2025 | LeadCity University, Ibadan, Nigeria & Join Momentum, USA*
#   E n h a n c i n g - C y b e r s e c u r i t y - w i t h - M a c h i n e - L e a r n i n g  
 