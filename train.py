import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from tqdm import tqdm

from model import MoCoViT, MoCoNTXentLoss
from utils import plot_conf_matrix, plot_roc, plot_umap_tsne


def train_model(
    data_path, results_dir, batch_size=32, epochs=50, patience=7,
    queue_size=2048, use_enhancement=True, use_contrastive=True, use_mse=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    transform_train=None, transform_val=None
):
    os.makedirs(results_dir, exist_ok=True)
    dataset = ImageFolder(data_path, transform=transform_train)
    labels = [s[1] for s in dataset.samples]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n Fold {fold+1}")

        trainval_labels = [labels[i] for i in trainval_idx]
        train_idx, val_idx = train_test_split(trainval_idx, test_size=0.176, stratify=trainval_labels, random_state=42)

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, num_workers=4)

        model = MoCoViT(use_enhancement, use_contrastive, use_mse).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion_contrastive = MoCoNTXentLoss(temperature=0.07)

        memory_queue = torch.randn(queue_size, model.encoder_q.num_features).to(device)
        memory_queue = F.normalize(memory_queue, dim=1)
        queue_ptr = 0

        best_val_acc = 0
        patience_counter = 0
        train_log = []

        for epoch in range(epochs):
            model.train()
            total_loss, correct, total = 0, 0, 0

            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                x, y = x.to(device), y.to(device)
                logits_q, logits_k, f_q, f_k, x_enh = model(x)

                ce_q = F.cross_entropy(logits_q, y)
                ce_k = F.cross_entropy(logits_k, y)
                mse = F.mse_loss(x_enh, x) if use_mse else 0
                contrastive = criterion_contrastive(f_q, f_k, memory_queue) if use_contrastive else 0

                loss = ce_q + ce_k + 0.2 * mse + 0.5 * contrastive

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.momentum_update()

                total_loss += loss.item()
                correct += (logits_q.argmax(1) == y).sum().item()
                total += y.size(0)

                actual_B = f_k.shape[0]
                end_ptr = queue_ptr + actual_B
                if end_ptr <= queue_size:
                    memory_queue[queue_ptr:end_ptr] = f_k
                else:
                    first_part = queue_size - queue_ptr
                    memory_queue[queue_ptr:] = f_k[:first_part]
                    memory_queue[:actual_B - first_part] = f_k[first_part:]
                queue_ptr = (queue_ptr + actual_B) % queue_size

            train_acc = correct / total

            model.eval()
            all_val_labels, val_preds, val_probs, all_val_features = [], [], [], []
            val_loss_total = 0

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits_q, _, f_q, _, _ = model(x)
                    val_loss_total += F.cross_entropy(logits_q, y).item()
                    probs = F.softmax(logits_q, dim=1)[:, 1]
                    preds = logits_q.argmax(1)
                    val_preds += preds.cpu().numpy().tolist()
                    val_probs += probs.cpu().numpy().tolist()
                    all_val_labels += y.cpu().numpy().tolist()
                    all_val_features.append(f_q.cpu().numpy())

            val_acc = accuracy_score(all_val_labels, val_preds)
            val_loss_avg = val_loss_total / len(val_loader)

            print(f" Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            train_log.append({
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": total_loss / len(train_loader),
                "train_acc": train_acc,
                "val_loss": val_loss_avg,
                "val_acc": val_acc
            })

            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), f"{results_dir}/best_model_fold{fold+1}.pth")
                print(" Model improved and saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(" Early stopping triggered!")
                    break

        features = np.vstack(all_val_features)
        pd.DataFrame(train_log).to_csv(f"{results_dir}/train_log_fold{fold+1}.csv", index=False)
        np.savetxt(f"{results_dir}/features_fold{fold+1}.csv", features, delimiter=",")
        np.savetxt(f"{results_dir}/labels_fold{fold+1}.csv", np.array(all_val_labels), delimiter=",")
        np.savetxt(f"{results_dir}/preds_fold{fold+1}.csv", np.array(val_preds), delimiter=",")

        plot_conf_matrix(all_val_labels, val_preds, f"{results_dir}/cm_fold{fold+1}.png", f"Confusion Matrix Fold {fold+1}")
        plot_roc(all_val_labels, val_probs, f"{results_dir}/roc_fold{fold+1}.png", f"ROC Curve Fold {fold+1}")
        plot_umap_tsne(features, all_val_labels, fold+1, prefix="internal")


if __name__ == "__main__":
    from torchvision import transforms

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomAffine(degrees=15, shear=10),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
        transforms.ToTensor()
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor()
    ])

    train_model(
        data_path="/InternalData",
        results_dir="/moCE_vit",
        transform_train=transform_train,
        transform_val=transform_val
    )
