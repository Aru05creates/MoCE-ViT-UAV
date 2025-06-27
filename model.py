### model

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

USE_ENHANCEMENT = True
MOMENTUM = 0.999

class EnhancementBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class MoCoViT(nn.Module):
    def __init__(self, backbone_name='vit_base_patch16_224.dino', num_classes=2):
        super().__init__()
        self.use_enh = USE_ENHANCEMENT
        self.enhancer = EnhancementBlock() if self.use_enh else nn.Identity()
        self.encoder_q = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.encoder_k = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.classifier = nn.Linear(self.encoder_q.num_features, num_classes)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def momentum_update(self, m=MOMENTUM):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    def forward(self, x):
        x_enh = self.enhancer(x)
        f_q = self.encoder_q(x)
        f_k = self.encoder_k(x_enh).detach()
        logits_q = self.classifier(f_q)
        logits_k = self.classifier(f_k)
        return logits_q, logits_k, f_q, f_k, x_enh

class MoCoNTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cosine = nn.CosineSimilarity(dim=2)

    def forward(self, q, k, queue):
        N, D = q.size()
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        queue = F.normalize(queue, dim=1)

        l_pos = torch.bmm(q.view(N, 1, D), k.view(N, D, 1)).squeeze(-1).squeeze(-1)
        l_neg = torch.mm(q, queue.T)

        logits = torch.cat([l_pos.unsqueeze(1), l_neg], dim=1) / self.temperature
        labels = torch.zeros(N, dtype=torch.long).to(logits.device)
        return F.cross_entropy(logits, labels)
