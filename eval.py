# evaluation script
# External evaluation script for trained MoCoViT models on EXT1 and EXT2 datasets

import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import MoCoViT
from utils import plot_conf_matrix, plot_roc
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", DEVICE)

EXT1_PATH = "./EXT1"
EXT2_PATH = "./EXT2"
MODEL_DIR = "./results"
BATCH_SIZE = 32
CLASS_NAMES = ['Healthy', 'Unhealthy']

transform_val = transforms.Compose([
    transforms.ToTensor()
])

def evaluate_external_dataset(model, dataloader, fold, tag):
    model.eval()
    preds, probs, labels = [], [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=f"{tag} Fold {fold}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits_q, _, _, _, _ = model(x)
            p = F.softmax(logits_q, dim=1)[:, 1]
            preds += logits_q.argmax(1).cpu().numpy().tolist()
            probs += p.cpu().numpy().tolist()
            labels += y.cpu().numpy().tolist()
    return preds, probs, labels

def main():
    ext1_loader = DataLoader(ImageFolder(EXT1_PATH, transform=transform_val), batch_size=BATCH_SIZE, num_workers=2)
    ext2_loader = DataLoader(ImageFolder(EXT2_PATH, transform=transform_val), batch_size=BATCH_SIZE, num_workers=2)

    external_results = []

    for fold in range(1, 6):
        model = MoCoViT().to(DEVICE)
        model_path = f"{MODEL_DIR}/best_model_fold{fold}.pth"
        model.load_state_dict(torch.load(model_path))

        # EXT1
        ext1_preds, ext1_probs, ext1_labels = evaluate_external_dataset(model, ext1_loader, fold, "EXT1")
        plot_conf_matrix(ext1_labels, ext1_preds, f"{MODEL_DIR}/cm_ext1_fold{fold}.png", f"EXT1 CM Fold {fold}")
        plot_roc(ext1_labels, ext1_probs, f"{MODEL_DIR}/roc_ext1_fold{fold}.png", f"EXT1 ROC Fold {fold}")

        # EXT2
        ext2_preds, ext2_probs, ext2_labels = evaluate_external_dataset(model, ext2_loader, fold, "EXT2")
        plot_conf_matrix(ext2_labels, ext2_preds, f"{MODEL_DIR}/cm_ext2_fold{fold}.png", f"EXT2 CM Fold {fold}")
        plot_roc(ext2_labels, ext2_probs, f"{MODEL_DIR}/roc_ext2_fold{fold}.png", f"EXT2 ROC Fold {fold}")

        external_results.append({
            "fold": fold,
            "ext1_acc": accuracy_score(ext1_labels, ext1_preds),
            "ext1_auc": roc_auc_score(ext1_labels, ext1_probs),
            "ext2_acc": accuracy_score(ext2_labels, ext2_preds),
            "ext2_auc": roc_auc_score(ext2_labels, ext2_probs),
        })

    # Save results
    import pandas as pd
    pd.DataFrame(external_results).to_csv(f"{MODEL_DIR}/external_results.csv", index=False)
    print("External evaluation results saved.")

if __name__ == "__main__":
    main()
