#!/usr/bin/env python3

"""
Treina rede neural MLP em GPU (PyTorch) com métricas completas e formato de output 
idêntico ao script de Random Forest/XGBoost.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from itertools import product as iterproduct
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, label_binarize

# ---------------------------------------------------------------------------
# Colunas de features (mesmo padrão do script de XGBoost/RF)
# ---------------------------------------------------------------------------

FEATURE_COLS_BASE = [
    "idade", "tempo_ate_consulta", "tempo_ate_tratamento", "tipo_caso", "sexo",
    "historico_familiar_cancer", "mais_de_um_tumor_primario", "escolaridade",
    "t_tnm", "n_tnm", "m_tnm", "t_ptnm", "n_ptnm", "m_ptnm",
    "comportamento_histologico_tumor", "historico_tabagismo_info_ausente",
    "historico_alcoolismo_info_ausente", "tipo_histologico_tumor_te",
    "subcat_localizacao_primaria_te", "cat_localizacao_primaria_te",
    "ocupacao_principal_gap",
]

OHE_PREFIXES = [
    "raca_cor_", "uf_procedencia_regiao_", "uf_hospital_regiao_",
    "origem_encaminhamento_", "exames_relevantes_diagnostico_",
    "diagnostico_tratamento_anteriores_", "base_diagnostico_mais_importante_",
    "base_diagnostico_microscopica_", "primeiro_tratamento_hospital_",
    "razao_nao_tratamento_hospital_", "historico_tabagismo_clinico_",
    "historico_alcoolismo_clinico_",
]

# ---------------------------------------------------------------------------
# Definição da MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], output_dim: int, dropout: float = 0.0, activation: str = "relu"):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}[activation]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---------------------------------------------------------------------------
# Funções de Métricas (Igual ao XGBoost)
# ---------------------------------------------------------------------------

def compute_ks_gain_lift(y_true, proba_pos, n_deciles=10):
    n = len(y_true)
    total_pos = y_true.sum()
    total_neg = n - total_pos
    if total_pos == 0 or total_neg == 0:
        return {"ks": float("nan"), "ks_decile": float("nan"), "gain_by_decile": [], "lift_by_decile": [], "decil_fracs": []}

    order = np.argsort(proba_pos)[::-1]
    y_sorted = y_true[order]
    gains, lifts, decil_fracs, ks_values = [], [], [], []
    for d in range(1, n_deciles + 1):
        cutoff = int(np.ceil(n * d / n_deciles))
        captured_pos = y_sorted[:cutoff].sum()
        captured_neg = cutoff - captured_pos
        frac_base = cutoff / n
        gain = captured_pos / total_pos
        lift = gain / frac_base
        ks_values.append(captured_pos / total_pos - captured_neg / total_neg)
        decil_fracs.append(round(frac_base, 2))
        gains.append(round(float(gain), 4))
        lifts.append(round(float(lift), 4))
    return {"ks": float(np.max(ks_values)), "ks_decile": int(np.argmax(ks_values) + 1), "gain_by_decile": gains, "lift_by_decile": lifts, "decil_fracs": decil_fracs}

def compute_all_metrics(y_true, y_proba, auc_mode, threshold=0.5, split_name="teste"):
    is_binary = auc_mode == "binary"
    metrics = {}
    if is_binary:
        proba_pos = y_proba[:, 1]
        y_pred = (proba_pos >= threshold).astype(int)
    else:
        y_pred = np.argmax(y_proba, axis=1)
        proba_pos = None

    if is_binary:
        metrics["auc_roc"] = roc_auc_score(y_true, proba_pos)
        metrics["auc_pr"] = average_precision_score(y_true, proba_pos)
    else:
        metrics["auc_roc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        classes = np.unique(y_true)
        ap_scores, weights = [], []
        for cls in classes:
            y_bin = (y_true == cls).astype(int)
            ap_scores.append(average_precision_score(y_bin, y_proba[:, cls]))
            weights.append(y_bin.sum())
        metrics["auc_pr"] = float(np.average(ap_scores, weights=weights))

    metrics["gini"] = 2 * metrics["auc_roc"] - 1
    avg = "binary" if is_binary else "weighted"
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    
    if is_binary:
        metrics["brier_score"] = brier_score_loss(y_true, proba_pos)
    else:
        y_bin_matrix = label_binarize(y_true, classes=np.unique(y_true))
        metrics["brier_score"] = float(np.mean((y_proba - y_bin_matrix) ** 2))
    metrics["log_loss"] = log_loss(y_true, y_proba)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metrics["classification_report"] = classification_report(y_true, y_pred, zero_division=0)

    if is_binary:
        kgl = compute_ks_gain_lift(y_true, proba_pos)
        metrics.update(kgl)
    else:
        metrics.update({"ks": float("nan"), "ks_decile": float("nan"), "gain_by_decile": [], "lift_by_decile": [], "decil_fracs": []})

    metrics.update({"_y_pred": y_pred, "_y_proba": y_proba, "_split": split_name, "_threshold": threshold})
    return metrics

# ---------------------------------------------------------------------------
# Formatação do Relatório (Igual ao XGBoost)
# ---------------------------------------------------------------------------

def format_metrics_block(m: dict, is_binary: bool) -> list[str]:
    split = m["_split"].upper()
    thr = m.get("_threshold", 0.5)
    ks_decil = m.get("ks_decile", "?")
    lines = [
        f"--- Metricas {split} ---", "",
        "  [Discriminacao]",
        f"  AUC-ROC       : {m['auc_roc']:.6f}",
        f"  AUC-PR        : {m['auc_pr']:.6f}   (Average Precision)",
        f"  Gini          : {m['gini']:.6f}", "",
        f"  [Predicao — threshold = {thr}]",
        f"  Accuracy      : {m['accuracy']:.6f}",
        f"  Precision     : {m['precision']:.6f}",
        f"  Recall        : {m['recall']:.6f}",
        f"  F1-Score      : {m['f1']:.6f}",
        f"  MCC           : {m['mcc']:.6f}", "",
        "  [Calibracao]",
        f"  Brier Score   : {m['brier_score']:.6f}   (menor e melhor)",
        f"  Log-Loss      : {m['log_loss']:.6f}", "",
        "  [Confusion Matrix]",
    ]
    for row in m["confusion_matrix"]:
        lines.append("  " + "  ".join(f"{v:6d}" for v in row))
    lines += ["", "  [Classification Report]"]
    for line in m["classification_report"].splitlines():
        lines.append("  " + line)
    if is_binary and m.get("gain_by_decile"):
        lines += ["", "  [KS / Ganho / Lift por Decil]", f"  KS estatistico : {m['ks']:.6f}   (maximo no decil {ks_decil})", "", f"  {'Decil':>5}  {'% base':>7}  {'Ganho':>7}  {'Lift':>7}", f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}"]
        for i, (frac, gain, lift) in enumerate(zip(m["decil_fracs"], m["gain_by_decile"], m["lift_by_decile"]), start=1):
            marker = " <- KS" if i == ks_decil else ""
            lines.append(f"  {i:>5}  {frac:>6.0%}  {gain:>7.4f}  {lift:>7.4f}{marker}")
    return lines

# ---------------------------------------------------------------------------
# Auxiliares de Carregamento e Treino
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Deep Learning PyTorch com Métricas Completas.")
    parser.add_argument("--train-path", type=Path, default=Path("data_train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data_test.csv"))
    parser.add_argument("--target", type=str, default="status_vital")
    parser.add_argument("--output", type=Path, default=Path("deep_learning_results.txt"))
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--perm-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def load_split_data(train_path: Path, test_path: Path, target: str):
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
    train_df, test_df = train_df.dropna(subset=[target]).copy(), test_df.dropna(subset=[target]).copy()
    
    ohe_cols = [c for c in train_df.columns if any(c.startswith(p) for p in OHE_PREFIXES)]
    feature_cols = [c for c in (FEATURE_COLS_BASE + ohe_cols) if c in train_df.columns and c in test_df.columns]
    
    x_train, y_train = train_df[feature_cols].astype(float), train_df[target].astype(int)
    x_test, y_test = test_df[feature_cols].astype(float), test_df[target].astype(int)
    w_train = train_df["class_weight"].astype(float) if "class_weight" in train_df.columns else None
    return x_train, y_train, x_test, y_test, w_train, feature_cols

def fit_mlp(x_train_np, y_train_np, params, output_dim, device, random_state, sample_weight=None):
    torch.manual_seed(random_state)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train_np).astype(np.float32)
    x_tensor, y_tensor = torch.tensor(x_scaled, device=device), torch.tensor(y_train_np.astype(np.int64), device=device)
    w_tensor = torch.tensor(sample_weight.astype(np.float32), device=device) if sample_weight is not None else None

    model = MLP(x_scaled.shape[1], params["hidden_sizes"], output_dim, params["dropout"], params["activation"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    model.train()
    for _ in range(params["epochs"]):
        perm = torch.randperm(len(x_tensor), device=device)
        x_tensor, y_tensor = x_tensor[perm], y_tensor[perm]
        if w_tensor is not None: w_tensor = w_tensor[perm]
        for s in range(0, len(x_tensor), params["batch_size"]):
            xb, yb = x_tensor[s:s+params["batch_size"]], y_tensor[s:s+params["batch_size"]]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            if w_tensor is not None: loss = (loss * w_tensor[s:s+params["batch_size"]]).mean()
            else: loss = loss.mean()
            loss.backward(); optimizer.step()
    return model, scaler

def predict_proba_mlp(model, x_np, scaler, device):
    model.eval()
    x_t = torch.tensor(scaler.transform(x_np).astype(np.float32), device=device)
    with torch.no_grad(): return torch.softmax(model(x_t), dim=-1).cpu().numpy()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Conforme seu script original
    x_train, y_train, x_test, y_test, w_train, feature_cols = load_split_data(args.train_path, args.test_path, args.target)
    auc_mode = "binary" if y_train.nunique() <= 2 else "multiclass"
    output_dim = int(y_train.nunique())

    # Grid simplificado para exemplo (ajuste conforme necessário)
    grid = {"hidden_sizes": [[256, 128]], "dropout": [0.3], "lr": [1e-3], "batch_size": [1024], "epochs": [100], "activation": ["relu"]}
    all_params = [dict(zip(grid.keys(), v)) for v in iterproduct(*grid.values())]

    # Grid Search manual
    best_score, best_params = -1.0, None
    cv_splitter = KFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    
    print(f"Iniciando Grid Search ({len(all_params)} combos)...")
    for params in all_params:
        scores = []
        for tr_idx, vl_idx in cv_splitter.split(x_train):
            m, s = fit_mlp(x_train.iloc[tr_idx].to_numpy(), y_train.iloc[tr_idx].to_numpy(), params, output_dim, args.device, args.random_state, w_train.iloc[tr_idx].to_numpy() if w_train is not None else None)
            p = predict_proba_mlp(m, x_train.iloc[vl_idx].to_numpy(), s, args.device)
            scores.append(roc_auc_score(y_train.iloc[vl_idx], p[:, 1] if auc_mode == "binary" else p, multi_class="ovr"))
        mean_s = np.mean(scores)
        if mean_s > best_score: best_score, best_params = mean_s, params
        print(f"AUC: {mean_s:.4f} com {params}")

    # Treino Final
    best_model, best_scaler = fit_mlp(x_train.to_numpy(), y_train.to_numpy(), best_params, output_dim, args.device, args.random_state, w_train.to_numpy() if w_train is not None else None)
    
    train_metrics = compute_all_metrics(y_train.to_numpy(), predict_proba_mlp(best_model, x_train.to_numpy(), best_scaler, args.device), auc_mode, args.threshold, "treino")
    test_metrics = compute_all_metrics(y_test.to_numpy(), predict_proba_mlp(best_model, x_test.to_numpy(), best_scaler, args.device), auc_mode, args.threshold, "teste")

    # Relatório TXT
    lines = ["="*70, "DEEP LEARNING PYTORCH - RESULTADOS COMPLETOS", "="*70,
             f"Data/hora: {datetime.now().isoformat()}", f"Alvo: {args.target}", f"Dispositivo: {args.device}",
             f"Features: {len(feature_cols)}", f"Melhores Params: {best_params}", ""]
    lines += format_metrics_block(train_metrics, auc_mode == "binary")
    lines += [""] + format_metrics_block(test_metrics, auc_mode == "binary")
    args.output.write_text("\n".join(lines))

    # Export Ensemble
    np.save(args.output.parent / "deep_learning_test_proba.npy", test_metrics["_y_proba"])
    print(f"\nFinalizado! Relatório em {args.output}")

if __name__ == "__main__":
    main()
