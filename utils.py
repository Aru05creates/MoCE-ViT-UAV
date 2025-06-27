# utils

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, silhouette_score
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.manifold import TSNE

CLASS_NAMES = ['Healthy', 'Unhealthy']

def plot_conf_matrix(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()

def plot_roc(y_true, y_probs, path, title):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()

def plot_umap_tsne(features, labels, fold, prefix, results_dir):
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    
    embeddings = {
        "umap": umap.UMAP().fit_transform(X),
        "tsne": TSNE().fit_transform(X)
    }

    for name, emb in embeddings.items():
        plt.figure(figsize=(6, 5))
        for i, label in enumerate(CLASS_NAMES):
            idx = np.array(labels) == i
            plt.scatter(emb[idx, 0], emb[idx, 1], label=label, s=10)
        sil = silhouette_score(emb, labels)
        plt.title(f"{name.upper()} | Fold {fold} | Silhouette: {sil:.4f}")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{results_dir}/{prefix}_{name}_fold{fold}.png")
        plt.close()

def combine_umap_all_folds(results_dir, feature_paths, label_paths):
    all_feats, all_labels = [], []
    for f_path, l_path in zip(feature_paths, label_paths):
        feats = np.loadtxt(f_path, delimiter=",")
        labels = np.loadtxt(l_path, delimiter=",")
        all_feats.append(feats)
        all_labels.append(labels)

    X = np.vstack(all_feats)
    y = np.concatenate(all_labels)
    X_scaled = StandardScaler().fit_transform(X)
    embedding = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    for cls in np.unique(y):
        idx = y == cls
        plt.scatter(embedding[idx, 0], embedding[idx, 1], label=f"Class {int(cls)}", s=10)
    plt.title("UMAP: All Folds Combined"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(f"{results_dir}/umap_combined_all_folds.png")
    plt.close()
