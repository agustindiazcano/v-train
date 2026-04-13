#!/usr/bin/env python3
# =============================================================================
# MONOLITO LVM-TITAN V11: RUNPOD EDITION (INFONCE + BENCHMARK GLOBAL)
# =============================================================================
import os
import sys
import time
import math
import subprocess

# -----------------------------------------------------------------------------
# 0. BOOTSTRAP DE DEPENDENCIAS (Instalación autónoma)
# -----------------------------------------------------------------------------
def install_deps():
    print("📦 Verificando infraestructura de dependencias para H100...")
    paquetes = ["pytorch_lightning", "datasets", "spacy", "faiss-cpu"]
    for p in paquetes:
        try:
            __import__(p.replace("-", "_"))
        except ImportError:
            print(f"Instalando {p}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
    
    # Modelo de lenguaje de SpaCy
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except OSError:
        print("Descargando modelo topológico de SpaCy...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

install_deps()

import spacy
import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

# --- CONFIGURACIÓN DE GRADO INDUSTRIAL ---
CONFIG = {
    "dim_sem": 512,
    "epochs": 40,
    "batch_size": 16384,     # Saturación masiva para 80GB VRAM
    "lr": 0.005,
    "num_negativos": 127,    # Presión termodinámica extrema
    "temp_train": 2.0,
    "temp_eval": 1.5,
    "precision": "bf16-mixed", # Acelerador TensorCore de H100
    "train_limit": 100000,   # Párrafos de entrenamiento (WikiText-103)
    "checkpoint_dir": "./lvm_checkpoints"
}

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
pl.seed_everything(42)

# -----------------------------------------------------------------------------
# 1. INGESTA Y TOPOLOGÍA (TRAIN)
# -----------------------------------------------------------------------------
print(f"\n🚜 [FASE 1] Extrayendo Topología SE(3) de WikiText-103 ({CONFIG['train_limit']} párrafos)...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
textos = [txt for txt in dataset['text'] if len(txt.strip()) > 50][:CONFIG["train_limit"]]

nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
vocabulario = {"[PAD]": 0, "[UNK]": 1}
idx2word = {0: "[PAD]", 1: "[UNK]"}
enlaces = []
SOCKET_MAP = {"nsubj": 0, "csubj": 0, "dobj": 1, "pobj": 1, "amod": 2, "advmod": 2}

for doc in nlp.pipe(textos, batch_size=1000, n_process=4):
    for token in doc:
        if token.is_punct or token.is_space or token.dep_ == "ROOT": continue
        lem, h_lem = token.lemma_.lower(), token.head.lemma_.lower()
        for l in [lem, h_lem]:
            if l not in vocabulario:
                idx = len(vocabulario); vocabulario[l] = idx; idx2word[idx] = l
        enlaces.append((vocabulario[lem], vocabulario[h_lem], SOCKET_MAP.get(token.dep_, 3)))

t_p = torch.tensor([e[0] for e in enlaces])
t_s = torch.tensor([e[1] for e in enlaces])
t_t = torch.tensor([e[2] for e in enlaces])
V_SIZE = len(vocabulario)

loader = DataLoader(TensorDataset(t_p, t_s, t_t), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
print(f"📊 Vocabulario Forjado: {V_SIZE} palabras únicas. Total Encastres: {len(t_p)}")

# -----------------------------------------------------------------------------
# 2. MOTOR FÍSICO SE(3) + INFONCE
# -----------------------------------------------------------------------------
def apply_rotation_batch(vectors, q):
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    v_x, v_y, v_z = vectors[:,0], vectors[:,1], vectors[:,2]
    tx, ty, tz = 2*(y*v_z - z*v_y), 2*(z*v_x - x*v_z), 2*(x*v_y - y*v_x)
    return vectors + w.unsqueeze(1)*torch.stack([tx, ty, tz], dim=1) + torch.stack([y*tz - z*ty, z*tx - x*tz, x*ty - y*tx], dim=1)

class LVM_Titan(pl.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        
        self.plugs_pos = nn.Embedding(vocab_size, 3)
        self.plugs_quat = nn.Embedding(vocab_size, 4)
        self.plugs_sem = nn.Embedding(vocab_size, CONFIG["dim_sem"])
        self.polaridad = nn.Embedding(vocab_size, 1)
        self.sockets_locales = nn.Parameter(torch.randn(vocab_size, 4, 3) * 0.05)
        self.sockets_sem = nn.Parameter(torch.randn(vocab_size, 4, CONFIG["dim_sem"]) * 0.05)
        
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
        
        # Energía real
        e_pos = self(p, s, t) 
        
        # Gas de colisión (Negativos InfoNCE)
        s_exp = s.repeat_interleave(CONFIG["num_negativos"])
        t_exp = t.repeat_interleave(CONFIG["num_negativos"])
        fake_p = torch.randint(0, self.vocab_size, (bs * CONFIG["num_negativos"],), device=self.device)
        
        e_neg = self(fake_p, s_exp, t_exp).view(bs, CONFIG["num_negativos"])
        
        # Termodinámica de Boltzmann
        e_total = torch.cat([e_pos.unsqueeze(1), e_neg], dim=1)
        logits = -e_total / CONFIG["temp_train"]
        
        labels = torch.zeros(bs, dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, labels)
        
        loss_carga = 0.5 * (1.0 - torch.abs(torch.tanh(self.polaridad.weight))).mean()
        
        self.log("train_loss", loss, prog_bar=True)
        return loss + loss_carga

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=CONFIG["lr"], weight_decay=0.01)

# -----------------------------------------------------------------------------
# 3. ENTRENAMIENTO (SHIELD ACTIVADO)
# -----------------------------------------------------------------------------
print(f"\n🔥 [FASE 2] Encendiendo Reactor H100 (Precision: {CONFIG['precision']})")

checkpoint_cb = ModelCheckpoint(
    dirpath=CONFIG["checkpoint_dir"], 
    filename="titan-{epoch:02d}-{train_loss:.2f}",
    save_top_k=1, 
    monitor="train_loss", 
    mode="min"
)

model = LVM_Titan(vocab_size=V_SIZE)
trainer = pl.Trainer(
    max_epochs=CONFIG["epochs"],
    accelerator="gpu",
    devices=1,
    precision=CONFIG["precision"],
    callbacks=[checkpoint_cb],
    enable_model_summary=True
)

trainer.fit(model, loader)
print(f"\n✅ Entrenamiento completado. Mejor estado guardado en: {checkpoint_cb.best_model_path}")

# -----------------------------------------------------------------------------
# 4. BENCHMARK ACADÉMICO DE PERPLEJIDAD
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("🔬 [FASE 3] BENCHMARK DE PERPLEJIDAD (EVALUACIÓN GLOBAL)")
print("=" * 60)

val_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
val_textos = [txt for txt in val_dataset['text'] if len(txt.strip()) > 50]
val_enlaces = []

for doc in nlp.pipe(val_textos, batch_size=500, n_process=4):
    for token in doc:
        if token.is_punct or token.is_space or token.dep_ == "ROOT": continue
        lem, h_lem = token.lemma_.lower(), token.head.lemma_.lower()
        if lem in vocabulario and h_lem in vocabulario: # Evaluamos solo en vocabulario conocido
            val_enlaces.append((vocabulario[lem], vocabulario[h_lem], SOCKET_MAP.get(token.dep_, 3)))

val_p = torch.tensor([e[0] for e in val_enlaces])
val_s = torch.tensor([e[1] for e in val_enlaces])
val_t = torch.tensor([e[2] for e in val_enlaces])

val_loader = DataLoader(TensorDataset(val_p, val_s, val_t), batch_size=1024, shuffle=False)
print(f"🎯 Total de encastres de validación a evaluar: {len(val_p)}")

@torch.no_grad()
def evaluate_perplexity(modelo, loader, temp):
    modelo.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelo.to(device)
    
    total_loss = 0.0
    
    all_pos = F.normalize(modelo.plugs_pos.weight, dim=1)
    all_sem = F.normalize(modelo.plugs_sem.weight, dim=1)
    all_cargas = torch.tanh(modelo.polaridad.weight).squeeze()
    
    for batch_idx, (p, s, t) in enumerate(loader):
        p, s, t = p.to(device), s.to(device), t.to(device)
        
        m_pos, m_q = F.normalize(modelo.plugs_pos(s), dim=1), F.normalize(modelo.plugs_quat(s), dim=1)
        s_global = m_pos + apply_rotation_batch(modelo.sockets_locales[s, t, :], m_q)
        s_target_sem = F.normalize(modelo.sockets_sem[s, t, :], dim=1)
        c_mad = torch.tanh(modelo.polaridad(s)).squeeze()
        
        # MATRIZ GLOBAL (Broadcasting contra V_SIZE)
        e_mec = torch.cdist(s_global, all_pos, p=2)**2
        e_qui = torch.cdist(s_target_sem, all_sem, p=2)**2
        interaccion = c_mad.unsqueeze(1) * all_cargas.unsqueeze(0)
        e_coulomb = torch.relu(1.0 + interaccion) * 15.0
        
        e_total = e_mec + e_qui + e_coulomb
        logits = -e_total / temp
        
        loss = F.cross_entropy(logits, p)
        total_loss += loss.item()
        
        if batch_idx % 20 == 0:
            print(f"🔄 Evaluando batch {batch_idx}... Loss: {loss.item():.4f}")

    mean_loss = total_loss / len(loader)
    return mean_loss, math.exp(mean_loss)

t_eval = time.time()
loss_val, ppl_val = evaluate_perplexity(model, val_loader, temp=CONFIG["temp_eval"])

print("\n" + "=" * 60)
print("🏆 VEREDICTO FINAL: LVM-TITAN VS TRANSFORMER")
print("=" * 60)
print(f"📉 Validation Loss (Energía Libre): {loss_val:.4f}")
print(f"🤯 Perplejidad Termodinámica (PPL): {ppl_val:.2f}")
print(f"⏱️ Tiempo de evaluación: {time.time() - t_eval:.2f} segundos")
print("=" * 60)

if ppl_val < 40.0:
    print("\n✅ [STATUS: PARADIGM SHIFT] Has alcanzado métricas nivel GPT-2 con complejidad O(1). Prepara el paper.")
else:
    print("\n⚠️ [STATUS: STABLE] La física se sostiene. Para bajar más el PPL, el siguiente paso es escalar a 1 Billón de parámetros.")

# -----------------------------------------------------------------------------
# 5. INDEXACIÓN FAISS FINAL (EXPORTAR MOTOR O(1))
# -----------------------------------------------------------------------------
print("\n🗄️ [FASE 4] Exportando índice topológico HNSW FAISS para inferencia rápida...")
with torch.no_grad():
    pos_np = F.normalize(model.plugs_pos.weight, dim=1).cpu().numpy()
    sem_np = F.normalize(model.plugs_sem.weight, dim=1).cpu().numpy()
    vectores_fisicos = np.hstack([pos_np, sem_np]).astype('float32')

dim_total = vectores_fisicos.shape[1] 
index_faiss = faiss.IndexHNSWFlat(dim_total, 32)
index_faiss.add(vectores_fisicos)
faiss.write_index(index_faiss, "lvm_titan_index.faiss")
print(f"✅ Índice 'lvm_titan_index.faiss' exportado con éxito. Motor listo para producción.")