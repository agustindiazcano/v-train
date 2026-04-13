#!/usr/bin/env python3
# =============================================================================
# MONOLITO LVM-TITAN V11.1: RUNPOD EDITION (BLINDADO CONTRA OOM)
# =============================================================================
import os
import sys
import time
import math
import subprocess

# -----------------------------------------------------------------------------
# 0. BOOTSTRAP
# -----------------------------------------------------------------------------
def install_deps():
    print("📦 Verificando infraestructura...")
    paquetes = ["pytorch_lightning", "datasets", "spacy", "faiss-cpu"]
    for p in paquetes:
        try:
            __import__(p.replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except OSError:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

install_deps()

import spacy
import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

# --- CONFIGURACIÓN DE GRADO INDUSTRIAL (H100 - 80GB) ---
CONFIG = {
    "dim_sem": 512,
    "epochs": 40,
    "batch_size": 4096,      # Reducido de 16k a 8k para evitar picos de VRAM
    "lr": 0.005,
    "num_negativos": 127,    # Presión termodinámica extrema
    "temp_train": 2.0,
    "temp_eval": 1.5,
    "precision": "bf16-mixed", # TensorCores de la H100 al máximo
    "train_limit": 50000    # Experimento Pesado: 100k párrafos
}

pl.seed_everything(42)

# -----------------------------------------------------------------------------
# 1. INGESTA Y TOPOLOGÍA
# -----------------------------------------------------------------------------
print(f"\n🚜 [FASE 1] Extrayendo Topología SE(3) de WikiText-103 ({CONFIG['train_limit']} párrafos)...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
textos = [txt for txt in dataset['text'] if len(txt.strip()) > 50][:CONFIG["train_limit"]]

nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
vocabulario = {"[PAD]": 0, "[UNK]": 1}
enlaces = []
SOCKET_MAP = {"nsubj": 0, "csubj": 0, "dobj": 1, "pobj": 1, "amod": 2, "advmod": 2}

for doc in nlp.pipe(textos, batch_size=1000, n_process=2):
    for token in doc:
        if token.is_punct or token.is_space or token.dep_ == "ROOT": continue
        lem, h_lem = token.lemma_.lower(), token.head.lemma_.lower()
        for l in [lem, h_lem]:
            if l not in vocabulario: vocabulario[l] = len(vocabulario)
        enlaces.append((vocabulario[lem], vocabulario[h_lem], SOCKET_MAP.get(token.dep_, 3)))

t_p = torch.tensor([e[0] for e in enlaces])
t_s = torch.tensor([e[1] for e in enlaces])
t_t = torch.tensor([e[2] for e in enlaces])
V_SIZE = len(vocabulario)

# FIX CLAVE: num_workers=0 evita el crasheo silencioso de Docker por RAM
loader = DataLoader(TensorDataset(t_p, t_s, t_t), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
print(f"📊 Vocabulario Forjado: {V_SIZE} palabras. Total Encastres: {len(t_p)}")

# -----------------------------------------------------------------------------
# 2. MOTOR FÍSICO SE(3) + INFONCE
# -----------------------------------------------------------------------------
def apply_rotation_batch(vectors, q):
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    v_x, v_y, v_z = vectors[:,0], vectors[:,1], vectors[:,2]
    tx, ty, tz = 2*(y*v_z - z*v_y), 2*(z*v_x - x*v_z), 2*(x*v_y - y*v_x)
    return vectors + w.unsqueeze(1)*torch.stack([tx, ty, tz], dim=1) + torch.stack([y*tz - z*ty, z*tx - x*tz, x*ty - y*tx], dim=1)

class LVM_Titan(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.plugs_pos = nn.Embedding(V_SIZE, 3)
        self.plugs_quat = nn.Embedding(V_SIZE, 4)
        self.plugs_sem = nn.Embedding(V_SIZE, CONFIG["dim_sem"])
        self.polaridad = nn.Embedding(V_SIZE, 1)
        self.sockets_locales = nn.Parameter(torch.randn(V_SIZE, 4, 3) * 0.05)
        self.sockets_sem = nn.Parameter(torch.randn(V_SIZE, 4, CONFIG["dim_sem"]) * 0.05)
        
        nn.init.orthogonal_(self.plugs_pos.weight)
        nn.init.orthogonal_(self.plugs_sem.weight)
        nn.init.normal_(self.polaridad.weight, mean=0.0, std=0.5)
        with torch.no_grad():
            self.plugs_quat.weight.fill_(0); self.plugs_quat.weight[:, 0] = 1.0 

    def forward(self, id_ent, id_madre, s_tipo):
        p_pos, p_sem = F.normalize(self.plugs_pos(id_ent), dim=1), F.normalize(self.plugs_sem(id_ent), dim=1)
        m_pos, m_q = F.normalize(self.plugs_pos(id_madre), dim=1), F.normalize(self.plugs_quat(id_madre), dim=1)
        s_global = m_pos + apply_rotation_batch(self.sockets_locales[id_madre, s_tipo, :], m_q)
        e_mec = torch.sum((p_pos - s_global)**2, dim=1)
        s_target_sem = F.normalize(self.sockets_sem[id_madre, s_tipo, :], dim=1)
        e_qui = torch.sum((p_sem - s_target_sem)**2, dim=1)
        e_coulomb = torch.relu(1.0 + torch.tanh(self.polaridad(id_ent)) * torch.tanh(self.polaridad(id_madre))) * 15.0
        return e_mec + e_qui + e_coulomb.squeeze()

    def training_step(self, batch, batch_idx):
        p, s, t = batch
        bs = p.shape[0]
        e_pos = self(p, s, t) 
        
        s_exp = s.repeat_interleave(CONFIG["num_negativos"])
        t_exp = t.repeat_interleave(CONFIG["num_negativos"])
        fake_p = torch.randint(0, V_SIZE, (bs * CONFIG["num_negativos"],), device=self.device)
        
        e_neg = self(fake_p, s_exp, t_exp).view(bs, CONFIG["num_negativos"])
        e_total = torch.cat([e_pos.unsqueeze(1), e_neg], dim=1)
        logits = -e_total / CONFIG["temp_train"]
        
        loss = F.cross_entropy(logits, torch.zeros(bs, dtype=torch.long, device=self.device))
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=CONFIG["lr"], weight_decay=0.01)

# -----------------------------------------------------------------------------
# 3. ENTRENAMIENTO (H100)
# -----------------------------------------------------------------------------
print(f"\n🔥 [FASE 2] Encendiendo Reactor H100 (Precision: {CONFIG['precision']})")
model = LVM_Titan()
trainer = pl.Trainer(max_epochs=CONFIG["epochs"], accelerator="gpu", devices=1, precision=CONFIG["precision"], enable_checkpointing=False)
trainer.fit(model, loader)

# -----------------------------------------------------------------------------
# 4. BENCHMARK ACADÉMICO DE PERPLEJIDAD
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("🔬 [FASE 3] BENCHMARK DE PERPLEJIDAD (EVALUACIÓN GLOBAL)")
print("=" * 60)

val_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
val_textos = [txt for txt in val_dataset['text'] if len(txt.strip()) > 50]
val_enlaces = []

for doc in nlp.pipe(val_textos, batch_size=500, n_process=2):
    for token in doc:
        if token.is_punct or token.is_space or token.dep_ == "ROOT": continue
        lem, h_lem = token.lemma_.lower(), token.head.lemma_.lower()
        if lem in vocabulario and h_lem in vocabulario:
            val_enlaces.append((vocabulario[lem], vocabulario[h_lem], SOCKET_MAP.get(token.dep_, 3)))

val_p = torch.tensor([e[0] for e in val_enlaces])
val_s = torch.tensor([e[1] for e in val_enlaces])
val_t = torch.tensor([e[2] for e in val_enlaces])
val_loader = DataLoader(TensorDataset(val_p, val_s, val_t), batch_size=1024, shuffle=False)

@torch.no_grad()
def evaluate_perplexity(modelo, loader, temp):
    modelo.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelo.to(device)
    total_loss = 0.0
    
    all_pos = F.normalize(modelo.plugs_pos.weight, dim=1)
    all_sem = F.normalize(modelo.plugs_sem.weight, dim=1)
    all_cargas = torch.tanh(modelo.polaridad.weight).squeeze()
    
    for p, s, t in loader:
        p, s, t = p.to(device), s.to(device), t.to(device)
        m_pos, m_q = F.normalize(modelo.plugs_pos(s), dim=1), F.normalize(modelo.plugs_quat(s), dim=1)
        s_global = m_pos + apply_rotation_batch(modelo.sockets_locales[s, t, :], m_q)
        s_target_sem = F.normalize(modelo.sockets_sem[s, t, :], dim=1)
        c_mad = torch.tanh(modelo.polaridad(s)).squeeze()
        
        e_mec = torch.cdist(s_global, all_pos, p=2)**2
        e_qui = torch.cdist(s_target_sem, all_sem, p=2)**2
        e_coulomb = torch.relu(1.0 + (c_mad.unsqueeze(1) * all_cargas.unsqueeze(0))) * 15.0
        
        logits = -(e_mec + e_qui + e_coulomb) / temp
        total_loss += F.cross_entropy(logits, p).item()

    mean_loss = total_loss / len(loader)
    return mean_loss, math.exp(mean_loss)

loss_val, ppl_val = evaluate_perplexity(model, val_loader, temp=CONFIG["temp_eval"])

print("\n" + "=" * 60)
print("🏆 VEREDICTO FINAL: LVM-TITAN VS TRANSFORMER")
print("=" * 60)
print(f"📉 Validation Loss (Energía Libre): {loss_val:.4f}")
print(f"🤯 Perplejidad Termodinámica (PPL): {ppl_val:.2f}")
print("=" * 60)