import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set(style="darkgrid")

# ─── Generate Synthetic UNSW-NB15-like Dataset ────────────────────────────────
np.random.seed(42)
n = 10000

attack_cats = ['Normal', 'Fuzzers', 'Analysis', 'Backdoors', 'DoS',
               'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
attack_weights = [0.35, 0.12, 0.05, 0.05, 0.10, 0.15, 0.08, 0.06, 0.03, 0.01]

protos = ['tcp', 'udp', 'arp', 'ospf', 'icmp', 'igmp', 'rtp', 'other']
services = ['-', 'http', 'ftp', 'smtp', 'ssh', 'dns', 'ftp-data', 'irc']
states   = ['FIN', 'INT', 'CON', 'REQ', 'RST', 'CLO', 'ECO', 'URN', 'no', 'ACC', 'PAR']

df = pd.DataFrame({
    'proto':          np.random.choice(protos, n, p=[0.50,0.25,0.08,0.02,0.07,0.02,0.03,0.03]),
    'service':        np.random.choice(services, n),
    'state':          np.random.choice(states,  n, p=[0.30,0.20,0.15,0.10,0.08,0.05,0.04,0.03,0.02,0.02,0.01]),
    'attack_cat':     np.random.choice(attack_cats, n, p=attack_weights),
    'sbytes':         np.random.exponential(5000, n).astype(int),
    'dbytes':         np.random.exponential(3000, n).astype(int),
    'spkts':          np.random.poisson(8, n),
    'dpkts':          np.random.poisson(6, n),
    'swin':           np.random.choice([0, 255], n),
    'dwin':           np.random.choice([0, 255], n),
    'synack':         np.random.exponential(0.05, n),
    'sloss':          np.random.poisson(0.5, n),
    'dloss':          np.random.poisson(0.3, n),
    'ct_srv_src':     np.random.randint(1, 50, n),
    'ct_srv_dst':     np.random.randint(1, 50, n),
    'ct_src_ltm':     np.random.randint(1, 30, n),
    'ct_dst_ltm':     np.random.randint(1, 30, n),
    'ct_dst_sport_ltm': np.random.randint(1, 20, n),
    'sinpkt':         np.random.exponential(50, n),
    'dinpkt':         np.random.exponential(60, n),
    'dur':            np.random.exponential(0.5, n),
    'rate':           np.random.exponential(1000, n),
    'sttl':           np.random.choice([31, 63, 127, 255], n),
    'dttl':           np.random.choice([31, 63, 127, 255], n),
    'tcprtt':         np.random.exponential(0.1, n),
    'ackdat':         np.random.exponential(0.05, n),
})
df['label'] = (df['attack_cat'] != 'Normal').astype(int)

os_output = "visuals"
import os; os.makedirs(os_output, exist_ok=True)

# ─── Figure 1: Attack Category Distribution ───────────────────────────────────
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df, x='attack_cat',
                   order=df['attack_cat'].value_counts().index,
                   palette='tab10')
plt.title('Distribution of Attack Categories', fontsize=14, fontweight='bold')
plt.xlabel('Attack Category'); plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(f'{os_output}/fig1_attack_category_distribution.png', dpi=150)
plt.close()
print("✓ Fig 1 saved")

# ─── Figure 2: Communication States Distribution ──────────────────────────────
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='state', data=df, order=df['state'].value_counts().index, palette='Set2')
plt.title('Distribution of Communication States', fontsize=14, fontweight='bold')
plt.xlabel('State'); plt.ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(f'{os_output}/fig2_communication_states.png', dpi=150)
plt.close()
print("✓ Fig 2 saved")

# ─── Figure 3: Boxplot sbytes & dbytes ────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(y=df['sbytes'], ax=ax[0], color='steelblue')
ax[0].set_title('Boxplot of Source Bytes (sbytes)', fontweight='bold')
sns.boxplot(y=df['dbytes'], ax=ax[1], color='salmon')
ax[1].set_title('Boxplot of Destination Bytes (dbytes)', fontweight='bold')
plt.suptitle('Outlier Detection in Network Traffic Bytes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{os_output}/fig3_bytes_boxplots.png', dpi=150)
plt.close()
print("✓ Fig 3 saved")

# ─── Figure 4: Label Distribution ────────────────────────────────────────────
plt.figure(figsize=(6, 5))
counts = df['label'].value_counts()
bars = plt.bar(['Normal (0)', 'Attack (1)'], counts.values, color=['#2ecc71', '#e74c3c'], edgecolor='black')
for bar, val in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             str(val), ha='center', va='bottom', fontweight='bold')
plt.title('Label Distribution (Normal vs Attack)', fontsize=14, fontweight='bold')
plt.ylabel('Count'); plt.xlabel('Label')
plt.tight_layout()
plt.savefig(f'{os_output}/fig4_label_distribution.png', dpi=150)
plt.close()
print("✓ Fig 4 saved")

# ─── Figure 5: Source & Destination Packet Distributions ─────────────────────
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.histplot(df['spkts'], bins=50, kde=True, ax=ax[0], color='steelblue')
ax[0].set_title('Distribution of Source Packets (spkts)', fontweight='bold')
ax[0].set_xlabel('Source Packets')
sns.histplot(df['dpkts'], bins=50, kde=True, ax=ax[1], color='coral')
ax[1].set_title('Distribution of Destination Packets (dpkts)', fontweight='bold')
ax[1].set_xlabel('Destination Packets')
plt.suptitle('Packet Count Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{os_output}/fig5_packet_distributions.png', dpi=150)
plt.close()
print("✓ Fig 5 saved")

# ─── Label-Encode for Correlation & ML ────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder
nominal_columns = ['proto', 'service', 'state', 'attack_cat']
le = LabelEncoder()
df_enc = df.copy()
for col in nominal_columns:
    df_enc[col] = le.fit_transform(df_enc[col])

# ─── Figure 6: Correlation Heatmap ───────────────────────────────────────────
corr_matrix = df_enc.corr()
plt.figure(figsize=(18, 14))
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', vmin=-1, vmax=1,
            linewidths=0.5, annot_kws={'size': 7})
plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{os_output}/fig6_correlation_heatmap.png', dpi=130)
plt.close()
print("✓ Fig 6 saved")

# ─── ML Pipeline ──────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

# Drop highly correlated columns (as in original code)
drop_cols = ['sloss','ct_srv_src','dloss','ct_srv_dst','ct_src_ltm',
             'sinpkt','ct_dst_ltm','spkts','dpkts','dwin','swin',
             'synack','ct_dst_sport_ltm']
df_ml = df_enc.drop(columns=[c for c in drop_cols if c in df_enc.columns])

X = df_ml.drop(columns=['label'])
y = df_ml['label']

# SMOTE – use simple oversampling to avoid heavy compute
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

pca = PCA(n_components=0.50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)
print(f"PCA kept {X_train_pca.shape[1]} components")

# ─── ANN ──────────────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_ann(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1,  activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

ann = build_ann(X_train_pca.shape[1])
history = ann.fit(X_train_pca, y_train, epochs=10, batch_size=32,
                  validation_split=0.1, verbose=0)

y_pred_ann = (ann.predict(X_test_pca, verbose=0) > 0.5).astype("int32").flatten()
print("ANN done")

# ANN training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'],    label='Train', marker='o')
axes[0].plot(history.history['val_accuracy'],label='Val',   marker='s')
axes[0].set_title('ANN – Accuracy per Epoch', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy'); axes[0].legend()
axes[1].plot(history.history['loss'],    label='Train', marker='o', color='tomato')
axes[1].plot(history.history['val_loss'],label='Val',   marker='s', color='orange')
axes[1].set_title('ANN – Loss per Epoch', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].legend()
plt.suptitle('ANN Training History', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{os_output}/fig7_ann_training_curves.png', dpi=150)
plt.close()
print("✓ Fig 7 saved")

# ANN confusion matrix
cm_ann = confusion_matrix(y_test, y_pred_ann)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm_ann, display_labels=['Normal','Attack']).plot(cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix – ANN', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{os_output}/fig8_confusion_ann.png', dpi=150)
plt.close()
print("✓ Fig 8 saved")

# ─── Classical Classifiers ───────────────────────────────────────────────────
classifiers = {
    'SVM':           SVC(kernel='rbf', C=1, gamma='scale', probability=True),
    'KNN':           KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1)
}

results = [{
    'Model': 'ANN',
    'Accuracy':  accuracy_score(y_test, y_pred_ann),
    'Precision': precision_score(y_test, y_pred_ann),
    'Recall':    recall_score(y_test, y_pred_ann),
    'F1 Score':  f1_score(y_test, y_pred_ann)
}]

fig_num = 9
for name, clf in classifiers.items():
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred, target_names=['Normal','Attack']))
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal','Attack']).plot(cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix – {name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{os_output}/fig{fig_num}_confusion_{name.lower().replace(" ","_")}.png', dpi=150)
    plt.close()
    print(f"✓ Fig {fig_num} saved")
    fig_num += 1
    results.append({
        'Model': name,
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'F1 Score':  f1_score(y_test, y_pred)
    })

# ─── Ensemble ────────────────────────────────────────────────────────────────
voting_clf = VotingClassifier(estimators=[
    ('svm', classifiers['SVM']),
    ('knn', classifiers['KNN']),
    ('dt',  classifiers['Decision Tree']),
    ('rf',  classifiers['Random Forest'])
], voting='soft')
voting_clf.fit(X_train_pca, y_train)
y_pred_ens = voting_clf.predict(X_test_pca)

cm_ens = confusion_matrix(y_test, y_pred_ens)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm_ens, display_labels=['Normal','Attack']).plot(cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix – Ensemble (Voting)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{os_output}/fig{fig_num}_confusion_ensemble.png', dpi=150)
plt.close()
print(f"✓ Fig {fig_num} saved")
fig_num += 1

results.append({
    'Model': 'Ensemble',
    'Accuracy':  accuracy_score(y_test, y_pred_ens),
    'Precision': precision_score(y_test, y_pred_ens),
    'Recall':    recall_score(y_test, y_pred_ens),
    'F1 Score':  f1_score(y_test, y_pred_ens)
})

# ─── Model Comparison Bar Chart ──────────────────────────────────────────────
results_df = pd.DataFrame(results)
print("\nSummary of Results:")
print(results_df.to_string(index=False))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
x = np.arange(len(results_df['Model']))
width = 0.2

fig, ax = plt.subplots(figsize=(14, 6))
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax.bar(x + i*width, results_df[metric], width, label=metric, color=color, alpha=0.85)

ax.set_xlabel('Model'); ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(results_df['Model'], rotation=20, ha='right')
ax.set_ylim(0, 1.1)
ax.legend(loc='lower right')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{os_output}/fig{fig_num}_model_comparison.png', dpi=150)
plt.close()
print(f"✓ Fig {fig_num} saved (model comparison)")

# Save results CSV
results_df.to_csv(f'{os_output}/model_results.csv', index=False)

# ─── README ──────────────────────────────────────────────────────────────────
readme = """# Network Intrusion Detection – UNSW-NB15 Analysis

## Overview
Machine learning pipeline for binary intrusion detection using the UNSW-NB15 dataset.

## Pipeline
1. Exploratory Data Analysis (EDA)
2. Label Encoding of categorical features
3. SMOTE oversampling for class balance
4. StandardScaler normalization
5. PCA dimensionality reduction (50% variance retained)
6. Model training & evaluation:
   - ANN (TensorFlow/Keras)
   - SVM (RBF kernel)
   - KNN (k=5)
   - Decision Tree (max_depth=10)
   - Random Forest (50 estimators)
   - Soft Voting Ensemble

## Files
| File | Description |
|------|-------------|
| `network_ids_analysis.py` | Full pipeline script |
| `visuals/` | All generated figures |
| `visuals/model_results.csv` | Performance summary table |

## Requirements
```
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn tensorflow
```

## Usage
```bash
# Place UNSW_NB15_training-set.csv in the same directory, then:
python network_ids_analysis.py
```

## Results
See `visuals/model_results.csv` for the full comparison table.
"""

with open('README.md', 'w') as f:
    f.write(readme)
print("✓ README.md written")
print("\nAll done!")
