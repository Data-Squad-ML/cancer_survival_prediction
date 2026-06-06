#!/usr/bin/env python3

"""Treina DeepSurv (Cox MLP) em GPU (PyTorch) com Grid Search, KFold e metricas de survival.

Metricas adicionais incluidas (alem das originais):
  - Curvas de Kaplan-Meier por tercis de risco + log-rank test
  - KS de sobrevivência entre grupos de alto e baixo risco
  - Calibracao por decil (E/O ratio — analogo ao Lift/Ganho de classificacao)
  - D de Royston (poder discriminativo em escala de log-risco)
  - Distribuicao dos log-riscos por status de evento
  - Net Benefit simplificado (Decision Curve Analysis)
  - C-index por subgrupo (sexo, faixa etaria se disponiveis)
  - CSVs de curvas de sobrevivencia e calibracao para plotagem externa
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# ---------------------------------------------------------------------------
# Colunas de features (mesmo padrao dos outros scripts)
# ---------------------------------------------------------------------------

FEATURE_COLS_BASE = [
    "idade",
    "tempo_ate_consulta",
    "tempo_ate_tratamento",
    "tipo_caso",
    "sexo",
    "historico_familiar_cancer",
    "mais_de_um_tumor_primario",
    "escolaridade",
    "t_tnm",
    "n_tnm",
    "m_tnm",
    "t_ptnm",
    "n_ptnm",
    "m_ptnm",
    "comportamento_histologico_tumor",
    "historico_tabagismo_info_ausente",
    "historico_alcoolismo_info_ausente",
    "tipo_histologico_tumor_te",
    "subcat_localizacao_primaria_te",
    "cat_localizacao_primaria_te",
    "ocupacao_principal_gap",
]

OHE_PREFIXES = [
    "raca_cor_",
    "uf_procedencia_regiao_",
    "uf_hospital_regiao_",
    "origem_encaminhamento_",
    "exames_relevantes_diagnostico_",
    "diagnostico_tratamento_anteriores_",
    "base_diagnostico_mais_importante_",
    "base_diagnostico_microscopica_",
    "primeiro_tratamento_hospital_",
    "razao_nao_tratamento_hospital_",
    "historico_tabagismo_clinico_",
    "historico_alcoolismo_clinico_",
]


# ---------------------------------------------------------------------------
# Arquitetura DeepSurv
# ---------------------------------------------------------------------------

class DeepSurv(nn.Module):
    """MLP que aprende o log-risco parcial de Cox.

    Saida: escalar por amostra (log-risco relativo).
    Loss: negative partial log-likelihood de Cox (Breslow approximation).
    Quanto maior o valor de saida, maior o risco de evento.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        dropout: float = 0.3,
        activation: str = "selu",
        batch_norm: bool = True,
    ):
        super().__init__()

        act_map = {
            "relu": nn.ReLU,
            "selu": nn.SELU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
        }
        act_fn = act_map[activation]

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if batch_norm and activation != "selu":
                # BatchNorm e SELU sao incompativeis (SELU tem auto-normalizacao).
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            if dropout > 0.0:
                if activation == "selu":
                    layers.append(nn.AlphaDropout(dropout))
                else:
                    layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        # Inicializacao adequada para SELU (lecun_normal)
        if activation == "selu":
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Loss de Cox (negative partial log-likelihood, Breslow approximation)
# ---------------------------------------------------------------------------

def cox_partial_log_likelihood_loss(
    log_risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Negative partial log-likelihood de Cox com aproximacao de Breslow."""
    order = torch.argsort(time, descending=True)
    log_risk = log_risk[order]
    event = event[order]
    if sample_weight is not None:
        sample_weight = sample_weight[order]

    log_cumsum_hazard = torch.logcumsumexp(log_risk, dim=0)

    uncensored_mask = event.bool()
    event_log_risk = log_risk[uncensored_mask]
    event_log_cumsum = log_cumsum_hazard[uncensored_mask]

    per_event_ll = event_log_risk - event_log_cumsum

    if sample_weight is not None:
        w = sample_weight[uncensored_mask]
        loss = -(per_event_ll * w).sum() / (w.sum() + eps)
    else:
        n_events = uncensored_mask.sum().clamp(min=1)
        loss = -per_event_ll.sum() / n_events

    return loss


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------

def default_data_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino de DeepSurv (Cox MLP) em GPU com Grid Search, KFold e metricas survival."
    )
    parser.add_argument("--train-path", type=Path, default=default_data_path("data_train.csv"))
    parser.add_argument("--test-path", type=Path, default=default_data_path("data_test.csv"))
    parser.add_argument("--target", type=str, default="status_vital")
    parser.add_argument("--time-col", type=str, default="tempo_total_doenca")
    parser.add_argument("--event-time-col", type=str, default="tempo_ate_obito")
    parser.add_argument("--output", type=Path, default=Path("results/deepsurv_gpu_results.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                        help="Diretorio para CSVs auxiliares de curvas e calibracao.")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--perm-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-eval-times", type=int, default=8)
    parser.add_argument("--max-eval-times", type=int, default=25)
    parser.add_argument("--min-events-per-fold", type=int, default=5)
    parser.add_argument(
        "--weighting",
        type=str,
        choices=["ipcw", "class_weight", "none"],
        default="ipcw",
    )
    parser.add_argument("--ipcw-min-prob", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    # Coluna de subgrupo opcional para C-index estratificado
    parser.add_argument("--subgroup-col", type=str, default="sexo",
                        help="Coluna categorica para calculo de C-index por subgrupo.")
    return parser.parse_args()


def build_feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    feature_cols_ohe = [
        col for col in train_df.columns if any(col.startswith(p) for p in OHE_PREFIXES)
    ]
    selected = [col for col in (FEATURE_COLS_BASE + feature_cols_ohe) if col in train_df.columns]
    selected = [col for col in selected if col in test_df.columns]
    if not selected:
        raise ValueError("Nenhuma feature selecionada foi encontrada nos dados de treino/teste.")
    return selected


def build_survival_targets(
    df: pd.DataFrame,
    target: str,
    time_col: str,
    event_time_col: str,
) -> tuple:
    event = (df[target].astype(int) == 0).to_numpy(dtype=bool)
    time = np.where(event, df[event_time_col].to_numpy(), df[time_col].to_numpy()).astype(float)
    time = np.clip(time, 1e-6, None)
    y_surv = Surv.from_arrays(event=event, time=time)
    return y_surv, event, time


def compute_ipcw_weights(y_surv, min_prob: float = 1e-3) -> np.ndarray:
    event = np.asarray(y_surv["event"]).astype(bool)
    time = np.asarray(y_surv["time"]).astype(float)
    censor_event = ~event
    km_times, g_hat = kaplan_meier_estimator(censor_event, time)
    g_hat_interp = np.interp(time, km_times, g_hat, left=g_hat[0], right=g_hat[-1])
    g_hat_interp = np.clip(g_hat_interp, min_prob, 1.0)
    return (1.0 / g_hat_interp).astype(np.float32)


def load_split_data(
    train_path: Path,
    test_path: Path,
    target: str,
    time_col: str,
    event_time_col: str,
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    required = {target, time_col, event_time_col}
    for col in required:
        if col not in train_df.columns or col not in test_df.columns:
            raise ValueError(f"Coluna obrigatoria '{col}' ausente em treino ou teste.")

    train_df = train_df.dropna(subset=[target, time_col]).copy()
    test_df = test_df.dropna(subset=[target, time_col]).copy()

    feature_cols = build_feature_columns(train_df, test_df)

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    x_train = pd.DataFrame(
        imputer.fit_transform(train_df[feature_cols].astype(np.float32)),
        columns=feature_cols,
        index=train_df.index,
    )
    x_test = pd.DataFrame(
        imputer.transform(test_df[feature_cols].astype(np.float32)),
        columns=feature_cols,
        index=test_df.index,
    )

    y_train_surv, event_train, time_train = build_survival_targets(
        train_df, target, time_col, event_time_col
    )
    y_test_surv, event_test, time_test = build_survival_targets(
        test_df, target, time_col, event_time_col
    )

    w_class = train_df["class_weight"].astype(float).to_numpy() if "class_weight" in train_df.columns else None

    return (
        x_train, x_test,
        y_train_surv, y_test_surv,
        event_train, time_train,
        event_test, time_test,
        w_class, feature_cols,
        train_df, test_df,
    )


# ---------------------------------------------------------------------------
# Grid de hiperparametros
# ---------------------------------------------------------------------------

def param_grid():
    grid = {
        "hidden_sizes": [
            [128, 64],
            [256, 128],
            [256, 128, 64],
        ],
        "dropout": [0.1, 0.3],
        "lr": [1e-3, 3e-4],
        "batch_size": [512],
        "epochs": [75],
        "activation": ["selu", "relu"],
        "l2": [1e-4],
    }
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in iterproduct(*values):
        yield dict(zip(keys, combo))


# ---------------------------------------------------------------------------
# Treinamento de um DeepSurv
# ---------------------------------------------------------------------------

def fit_deepsurv(
    x_np: np.ndarray,
    time_np: np.ndarray,
    event_np: np.ndarray,
    params: dict,
    device: str,
    random_state: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[DeepSurv, StandardScaler]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_np).astype(np.float32)

    x_t = torch.tensor(x_scaled, device=device)
    time_t = torch.tensor(time_np.astype(np.float32), device=device)
    event_t = torch.tensor(event_np.astype(np.float32), device=device)
    w_t = (
        torch.tensor(sample_weight.astype(np.float32), device=device)
        if sample_weight is not None
        else None
    )

    model = DeepSurv(
        input_dim=x_scaled.shape[1],
        hidden_sizes=params["hidden_sizes"],
        dropout=params["dropout"],
        activation=params["activation"],
        batch_norm=True,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["l2"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    batch_size = params["batch_size"]
    n = len(x_t)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(random_state)

    for epoch in range(params["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=rng)
        x_t = x_t[perm]
        time_t = time_t[perm]
        event_t = event_t[perm]
        if w_t is not None:
            w_t = w_t[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            xb = x_t[start : start + batch_size]
            tb = time_t[start : start + batch_size]
            eb = event_t[start : start + batch_size]
            wb = w_t[start : start + batch_size] if w_t is not None else None

            if int(eb.sum().item()) < 2:
                continue

            optimizer.zero_grad()
            log_risk = model(xb)
            loss = cox_partial_log_likelihood_loss(log_risk, tb, eb, wb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if n_batches > 0:
            scheduler.step(epoch_loss / n_batches)

    return model, scaler


def predict_log_risk(
    model: DeepSurv,
    x_np: np.ndarray,
    scaler: StandardScaler,
    device: str,
) -> np.ndarray:
    model.eval()
    x_scaled = scaler.transform(x_np).astype(np.float32)
    x_t = torch.tensor(x_scaled, device=device)
    with torch.no_grad():
        log_risk = model(x_t).cpu().numpy()
    return log_risk


# ---------------------------------------------------------------------------
# Grid Search com StratifiedKFold (estratificado por evento)
# ---------------------------------------------------------------------------

def run_grid_search(
    x_train: pd.DataFrame,
    y_train_surv,
    event_train: np.ndarray,
    time_train: np.ndarray,
    w_train: np.ndarray | None,
    cv: int,
    random_state: int,
    device: str,
    min_events_per_fold: int,
) -> tuple:
    n_events = int(event_train.sum())

    if n_events < cv * min_events_per_fold:
        raise ValueError(
            f"Eventos insuficientes para CV: eventos={n_events}, cv={cv}, "
            f"min_events_per_fold={min_events_per_fold}."
        )

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_splits = list(cv_splitter.split(x_train.to_numpy(), event_train.astype(int)))

    all_params = list(param_grid())
    print(f"Total de combinacoes no grid: {len(all_params)}")
    print(f"CV: StratifiedKFold(n_splits={cv}) | Dispositivo: {device}")

    best_score = -1.0
    best_std = 0.0
    best_params = None

    x_np = x_train.to_numpy()

    for i, params in enumerate(all_params, start=1):
        fold_scores = []

        for fold_train_idx, fold_val_idx in cv_splits:
            x_fold_tr = x_np[fold_train_idx]
            time_fold_tr = time_train[fold_train_idx]
            event_fold_tr = event_train[fold_train_idx]
            w_fold_tr = w_train[fold_train_idx] if w_train is not None else None

            x_fold_val = x_np[fold_val_idx]
            event_fold_val = event_train[fold_val_idx]
            time_fold_val = time_train[fold_val_idx]

            if int(event_fold_tr.sum()) < 2:
                continue

            try:
                model, scaler = fit_deepsurv(
                    x_np=x_fold_tr,
                    time_np=time_fold_tr,
                    event_np=event_fold_tr,
                    params=params,
                    device=device,
                    random_state=random_state,
                    sample_weight=w_fold_tr,
                )

                log_risk_val = predict_log_risk(model, x_fold_val, scaler, device)

                y_fold_tr_surv = Surv.from_arrays(event=event_fold_tr.astype(bool), time=time_fold_tr)
                y_fold_val_surv = Surv.from_arrays(event=event_fold_val.astype(bool), time=time_fold_val)

                cindex = float(
                    concordance_index_ipcw(y_fold_tr_surv, y_fold_val_surv, log_risk_val)[0]
                )
                fold_scores.append(cindex)

            except Exception as e:
                print(f"  Fold falhou: {e}")
                continue

        if not fold_scores:
            continue

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))

        print(
            f"[{i}/{len(all_params)}] uno_cindex={mean_score:.6f} +/- {std_score:.6f} "
            f"params={params}"
        )

        if mean_score > best_score:
            best_score = mean_score
            best_std = std_score
            best_params = params

    return best_score, best_std, best_params


# ---------------------------------------------------------------------------
# Metricas de survival (originais)
# ---------------------------------------------------------------------------

def build_eval_times(
    y_train_surv,
    y_test_surv,
    min_eval_times: int,
    max_eval_times: int,
) -> np.ndarray:
    train_times = y_train_surv["time"].astype(float)
    test_times = y_test_surv["time"].astype(float)
    lower = max(float(np.min(train_times)), float(np.min(test_times)))
    upper = min(float(np.max(train_times)), float(np.max(test_times)))

    if upper <= lower:
        raise ValueError("Intervalo de tempos invalido para metricas dinamicas.")

    test_event_times = y_test_surv["time"][y_test_surv["event"]].astype(float)
    candidate = test_event_times[(test_event_times > lower) & (test_event_times < upper)]
    unique_candidate = np.unique(candidate)

    n_times = int(np.clip(len(unique_candidate), min_eval_times, max_eval_times))
    if len(unique_candidate) >= n_times:
        quantiles = np.linspace(0.1, 0.9, num=n_times)
        eval_times = np.quantile(unique_candidate, quantiles)
    else:
        eps = max((upper - lower) * 1e-3, 1e-6)
        eval_times = np.linspace(lower + eps, upper - eps, num=max(min_eval_times, 5))

    eval_times = np.unique(eval_times.astype(float))
    eval_times = eval_times[(eval_times > lower) & (eval_times < upper)]

    if len(eval_times) < 3:
        eps = max((upper - lower) * 1e-3, 1e-6)
        eval_times = np.linspace(lower + eps, upper - eps, num=5)

    return np.sort(eval_times)


def compute_breslow_survival(
    log_risk_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    log_risk_test: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    """Estimador de Breslow para funcao de sobrevivencia a partir do log-risco."""
    order = np.argsort(time_train)
    t_sorted = time_train[order]
    e_sorted = event_train[order]
    r_sorted = np.exp(log_risk_train[order])

    unique_times = np.unique(t_sorted[e_sorted])
    h0 = np.zeros(len(unique_times))

    for j, t in enumerate(unique_times):
        at_risk = np.sum(r_sorted[t_sorted >= t])
        n_events = np.sum(e_sorted[t_sorted == t])
        h0[j] = n_events / max(at_risk, 1e-10)

    H0_cumulative = np.cumsum(h0)

    H0_at_eval = np.interp(eval_times, unique_times, H0_cumulative, left=0.0, right=H0_cumulative[-1])

    risk_test = np.exp(log_risk_test)
    survival = np.exp(-np.outer(risk_test, H0_at_eval))  # shape (n_test, n_eval_times)

    return survival


def compute_survival_metrics(
    log_risk_train: np.ndarray,
    log_risk_test: np.ndarray,
    y_train_surv,
    y_test_surv,
    time_train: np.ndarray,
    event_train: np.ndarray,
    eval_times: np.ndarray,
) -> dict:
    cindex_train = float(
        concordance_index_censored(
            y_train_surv["event"], y_train_surv["time"], log_risk_train
        )[0]
    )
    cindex_test = float(
        concordance_index_censored(
            y_test_surv["event"], y_test_surv["time"], log_risk_test
        )[0]
    )
    uno_cindex_test = float(
        concordance_index_ipcw(y_train_surv, y_test_surv, log_risk_test)[0]
    )

    auc_by_time, mean_auc = cumulative_dynamic_auc(
        y_train_surv, y_test_surv, log_risk_test, eval_times
    )

    survival_test = compute_breslow_survival(
        log_risk_train=log_risk_train,
        time_train=time_train,
        event_train=event_train,
        log_risk_test=log_risk_test,
        eval_times=eval_times,
    )

    _, brier_scores = brier_score(y_train_surv, y_test_surv, survival_test, eval_times)
    mean_brier = float(np.mean(brier_scores))
    ibs = float(integrated_brier_score(y_train_surv, y_test_surv, survival_test, eval_times))

    return {
        "cindex_train": cindex_train,
        "cindex_test": cindex_test,
        "cindex_gap": cindex_train - cindex_test,
        "uno_cindex_test": uno_cindex_test,
        "mean_dynamic_auc": float(mean_auc),
        "dynamic_auc_by_time": auc_by_time,
        "mean_brier": mean_brier,
        "ibs": ibs,
        "eval_times": eval_times,
        "survival_test": survival_test,
        "brier_by_time": brier_scores,
    }


# ---------------------------------------------------------------------------
# Permutation Importance (C-index)
# ---------------------------------------------------------------------------

def permutation_importance_cindex(
    model: DeepSurv,
    scaler: StandardScaler,
    x_test: pd.DataFrame,
    y_train_surv,
    y_test_surv,
    n_repeats: int,
    random_state: int,
    device: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    x_np = x_test.to_numpy()

    base_risk = predict_log_risk(model, x_np, scaler, device)
    base_cindex = float(
        concordance_index_ipcw(y_train_surv, y_test_surv, base_risk)[0]
    )

    rows = []
    for j, col in enumerate(x_test.columns):
        scores = []
        for _ in range(n_repeats):
            x_perm = x_np.copy()
            x_perm[:, j] = rng.permutation(x_perm[:, j])
            perm_risk = predict_log_risk(model, x_perm, scaler, device)
            perm_cindex = float(
                concordance_index_ipcw(y_train_surv, y_test_surv, perm_risk)[0]
            )
            scores.append(base_cindex - perm_cindex)

        rows.append({
            "feature": col,
            "importance_mean": float(np.mean(scores)),
            "importance_std": float(np.std(scores)),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


# ===========================================================================
# METRICAS ADICIONAIS
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Kaplan-Meier por grupos de risco + log-rank test
# ---------------------------------------------------------------------------

def logrank_test_two_groups(
    time_a: np.ndarray, event_a: np.ndarray,
    time_b: np.ndarray, event_b: np.ndarray,
) -> tuple[float, float]:
    """Log-rank test manual (Mantel-Cox) entre dois grupos.

    Retorna (statistica_chi2, p_valor).
    """
    all_times = np.unique(np.concatenate([time_a[event_a], time_b[event_b]]))

    O_a_total, E_a_total = 0.0, 0.0
    V_total = 0.0

    for t in all_times:
        n_a = np.sum(time_a >= t)
        n_b = np.sum(time_b >= t)
        n = n_a + n_b
        if n == 0:
            continue

        d_a = np.sum((time_a == t) & event_a)
        d_b = np.sum((time_b == t) & event_b)
        d = d_a + d_b

        e_a = d * n_a / n
        O_a_total += d_a
        E_a_total += e_a

        if n > 1:
            v = d * n_a * n_b * (n - d) / (n**2 * (n - 1))
            V_total += v

    if V_total < 1e-10:
        return 0.0, 1.0

    chi2 = (O_a_total - E_a_total) ** 2 / V_total
    p_val = float(scipy_stats.chi2.sf(chi2, df=1))
    return float(chi2), p_val


def km_curve(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier simples. Retorna (times, survival_probs)."""
    order = np.argsort(time)
    t_s = time[order]
    e_s = event[order]

    unique_t = np.unique(t_s)
    n = len(t_s)
    s = 1.0
    times_out = [0.0]
    surv_out = [1.0]

    for t in unique_t:
        d = np.sum((t_s == t) & e_s)
        at_risk = np.sum(t_s >= t)
        if at_risk > 0:
            s *= (1 - d / at_risk)
        times_out.append(float(t))
        surv_out.append(float(s))

    return np.array(times_out), np.array(surv_out)


def compute_km_risk_groups(
    log_risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    n_groups: int = 3,
) -> dict:
    """Divide amostras em n_groups por quantil de log-risco e calcula KM + log-rank."""
    quantiles = np.linspace(0, 1, n_groups + 1)
    thresholds = np.quantile(log_risk, quantiles)
    labels = np.digitize(log_risk, thresholds[1:-1])  # 0, 1, ..., n_groups-1

    group_labels = []
    if n_groups == 3:
        group_labels = ["Baixo Risco", "Risco Intermediario", "Alto Risco"]
    else:
        group_labels = [f"Grupo {i+1}" for i in range(n_groups)]

    km_results = {}
    for g in range(n_groups):
        mask = labels == g
        if mask.sum() < 5:
            continue
        t_km, s_km = km_curve(time[mask], event[mask])
        km_results[group_labels[g]] = {
            "times": t_km,
            "survival": s_km,
            "n": int(mask.sum()),
            "n_events": int(event[mask].sum()),
        }

    # Log-rank: baixo vs alto
    mask_low = labels == 0
    mask_high = labels == (n_groups - 1)
    chi2, p_val = logrank_test_two_groups(
        time[mask_low], event[mask_low],
        time[mask_high], event[mask_high],
    )

    return {
        "groups": km_results,
        "logrank_chi2_low_vs_high": chi2,
        "logrank_pval_low_vs_high": p_val,
        "n_groups": n_groups,
    }


# ---------------------------------------------------------------------------
# 2. KS de sobrevivência (KS entre distribuicoes de log-risco por status)
# ---------------------------------------------------------------------------

def compute_survival_ks(
    log_risk: np.ndarray,
    event: np.ndarray,
) -> dict:
    """KS entre a distribuicao de log-risco de casos (evento=1) vs. censurados (evento=0).

    Analogo ao KS de classificacao, mede a separacao entre os dois grupos.
    """
    risk_event = np.sort(log_risk[event.astype(bool)])
    risk_censor = np.sort(log_risk[~event.astype(bool)])

    ks_stat, ks_pval = scipy_stats.ks_2samp(risk_event, risk_censor)

    # KS por decil (analogo ao KS de classificacao)
    n = len(log_risk)
    order = np.argsort(log_risk)[::-1]
    sorted_event = event[order].astype(float)

    total_events = event.sum()
    total_censored = (~event.astype(bool)).sum()

    decil_rows = []
    cumulative_events = 0.0
    cumulative_censored = 0.0
    best_ks = 0.0
    best_decil = 0

    for d in range(1, 11):
        idx_end = int(n * d / 10)
        idx_start = int(n * (d - 1) / 10)
        batch = sorted_event[idx_start:idx_end]
        cumulative_events += batch.sum()
        cumulative_censored += (1 - batch).sum()

        pct_events = cumulative_events / max(total_events, 1)
        pct_censored = cumulative_censored / max(total_censored, 1)
        ks_d = abs(pct_events - pct_censored)

        if ks_d > best_ks:
            best_ks = ks_d
            best_decil = d

        decil_rows.append({
            "decil": d,
            "pct_base": f"{d*10}%",
            "cum_pct_eventos": round(pct_events, 4),
            "cum_pct_censurados": round(pct_censored, 4),
            "ks_decil": round(ks_d, 4),
        })

    return {
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
        "ks_best_decil": best_decil,
        "ks_best_value": float(best_ks),
        "ks_by_decil": pd.DataFrame(decil_rows),
    }


# ---------------------------------------------------------------------------
# 3. Calibracao por decil de risco (E/O ratio — Observed vs Expected)
# ---------------------------------------------------------------------------

def compute_calibration_by_decil(
    log_risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    survival_at_t: np.ndarray,
    eval_time: float,
    n_decils: int = 10,
) -> pd.DataFrame:
    """Para um tempo de referencia eval_time, calcula para cada decil de risco:
      - Sobrevivencia esperada (media do modelo)
      - Sobrevivencia observada (Kaplan-Meier no decil)
      - Razao E/O

    Analogo ao Lift/Ganho de classificacao para survival.
    """
    order = np.argsort(log_risk)  # menor risco primeiro
    n = len(log_risk)

    rows = []
    for d in range(n_decils):
        idx_start = int(n * d / n_decils)
        idx_end = int(n * (d + 1) / n_decils)
        idx = order[idx_start:idx_end]

        expected_surv = float(np.mean(survival_at_t[idx]))  # media da curva do modelo
        expected_event_rate = 1.0 - expected_surv

        # KM observado no decil
        t_dec = time[idx]
        e_dec = event[idx]
        km_t, km_s = km_curve(t_dec, e_dec)
        obs_surv = float(np.interp(eval_time, km_t, km_s))
        obs_event_rate = 1.0 - obs_surv

        eo_ratio = obs_event_rate / max(expected_event_rate, 1e-8)

        rows.append({
            "decil": d + 1,
            "n": len(idx),
            "sobrev_esperada_modelo": round(expected_surv, 4),
            "taxa_evento_esperada": round(expected_event_rate, 4),
            "sobrev_observada_km": round(obs_surv, 4),
            "taxa_evento_observada": round(obs_event_rate, 4),
            "razao_E_O": round(eo_ratio, 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. D de Royston (poder discriminativo em escala de log-risco)
# ---------------------------------------------------------------------------

def compute_royston_d(
    log_risk: np.ndarray,
    event: np.ndarray,
) -> dict:
    """Calcula o D de Royston-Sauerbrei como medida de discriminacao.

    D = kappa * sqrt(8/pi) * (std dos prognostic scores entre eventos),
    onde os prognostic scores sao os quantis normais dos ranks do log-risco.

    Valores tipicos: D > 1 indica discriminacao razoavel, D > 2 e excelente.
    """
    n = len(log_risk)
    ranks = scipy_stats.rankdata(log_risk) / (n + 1)
    # Evita 0 e 1 para ppf
    ranks = np.clip(ranks, 1e-6, 1 - 1e-6)
    z = scipy_stats.norm.ppf(ranks)

    # Normaliza (media 0, std 1) apenas nos eventos
    z_events = z[event.astype(bool)]
    if len(z_events) < 5:
        return {"D_royston": np.nan, "R2_royston": np.nan}

    kappa = 1.0 / (np.sqrt(8.0 / np.pi))
    D = float(np.std(z_events, ddof=1) * np.sqrt(8.0 / np.pi))

    # R2 de Royston (fracao de variancia explicada)
    R2 = D**2 / (D**2 + (np.pi**2 / 6))

    return {"D_royston": round(D, 6), "R2_royston": round(R2, 6)}


# ---------------------------------------------------------------------------
# 5. Distribuicao dos log-riscos por status de evento
# ---------------------------------------------------------------------------

def compute_risk_score_distribution(
    log_risk: np.ndarray,
    event: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Estatisticas descritivas dos log-riscos separados por status de evento."""
    risk_ev = log_risk[event.astype(bool)]
    risk_cen = log_risk[~event.astype(bool)]

    def summary(arr: np.ndarray, label: str) -> dict:
        return {
            "grupo": label,
            "n": len(arr),
            "media": round(float(np.mean(arr)), 4),
            "mediana": round(float(np.median(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "p10": round(float(np.percentile(arr, 10)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    stats_df = pd.DataFrame([
        summary(risk_ev, "Evento (obito)"),
        summary(risk_cen, "Censurado (vivo)"),
    ])

    # Histograma por bins para exportacao
    all_bins = np.linspace(log_risk.min(), log_risk.max(), n_bins + 1)
    hist_rows = []
    for i in range(n_bins):
        lo, hi = all_bins[i], all_bins[i + 1]
        mask = (log_risk >= lo) & (log_risk < hi if i < n_bins - 1 else log_risk <= hi)
        hist_rows.append({
            "bin_inicio": round(lo, 4),
            "bin_fim": round(hi, 4),
            "n_evento": int(event[mask].sum()),
            "n_censurado": int((~event.astype(bool))[mask].sum()),
        })

    return {
        "stats_df": stats_df,
        "hist_df": pd.DataFrame(hist_rows),
    }


# ---------------------------------------------------------------------------
# 6. Net Benefit simplificado (Decision Curve Analysis)
# ---------------------------------------------------------------------------

def compute_net_benefit(
    log_risk: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    eval_time: float,
    survival_at_t: np.ndarray,
    n_thresholds: int = 20,
) -> pd.DataFrame:
    """Decision Curve Analysis simplificada para survival.

    Para cada threshold de probabilidade de evento pt, calcula:
      NB(pt) = (TP/n) - (FP/n) * (pt / (1 - pt))
    onde TP/FP sao baseados na predicao de evento ate eval_time vs. observado (KM).
    """
    n = len(log_risk)
    # Probabilidade de evento prevista = 1 - S(eval_time | x)
    prob_event = 1.0 - survival_at_t

    # Evento observado ate eval_time (usando o tempo real)
    obs_event = (event.astype(bool) & (time <= eval_time)).astype(float)

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    rows = []
    for pt in thresholds:
        predicted_pos = prob_event >= pt
        tp = float(np.sum(predicted_pos & obs_event.astype(bool)))
        fp = float(np.sum(predicted_pos & ~obs_event.astype(bool)))

        nb = (tp / n) - (fp / n) * (pt / max(1 - pt, 1e-8))

        # Estrategia "tratar todos"
        all_tp = float(obs_event.sum())
        all_fp = float((1 - obs_event).sum())
        nb_all = (all_tp / n) - (all_fp / n) * (pt / max(1 - pt, 1e-8))

        rows.append({
            "threshold_pt": round(float(pt), 3),
            "net_benefit_modelo": round(float(nb), 6),
            "net_benefit_tratar_todos": round(float(nb_all), 6),
            "net_benefit_nao_tratar": 0.0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. C-index por subgrupo
# ---------------------------------------------------------------------------

def compute_cindex_by_subgroup(
    log_risk: np.ndarray,
    y_train_surv,
    y_test_surv,
    event_test: np.ndarray,
    time_test: np.ndarray,
    subgroup_series: pd.Series | None,
) -> pd.DataFrame:
    """Calcula o Uno C-index para cada subgrupo de uma variavel categorica."""
    if subgroup_series is None or len(subgroup_series) == 0:
        return pd.DataFrame()

    rows = []
    for val in sorted(subgroup_series.unique()):
        mask = (subgroup_series == val).to_numpy()
        if mask.sum() < 10 or int(event_test[mask].sum()) < 3:
            continue

        lr_sub = log_risk[mask]
        y_sub = Surv.from_arrays(event=event_test[mask].astype(bool), time=time_test[mask])

        try:
            ci = float(concordance_index_ipcw(y_train_surv, y_sub, lr_sub)[0])
        except Exception:
            ci = float(
                concordance_index_censored(
                    event_test[mask].astype(bool), time_test[mask], lr_sub
                )[0]
            )

        rows.append({
            "subgrupo": val,
            "n": int(mask.sum()),
            "n_eventos": int(event_test[mask].sum()),
            "uno_cindex": round(ci, 6),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Utilitarios de exportacao de curvas KM para CSV
# ---------------------------------------------------------------------------

def km_groups_to_csv(km_result: dict, output_dir: Path, prefix: str = "km") -> None:
    """Exporta curvas KM por grupo para um CSV."""
    rows = []
    for group_name, gdata in km_result["groups"].items():
        for t, s in zip(gdata["times"], gdata["survival"]):
            rows.append({"grupo": group_name, "tempo": round(float(t), 4), "sobrevivencia": round(float(s), 4)})
    df = pd.DataFrame(rows)
    out = output_dir / f"{prefix}_curvas_km.csv"
    df.to_csv(out, index=False)
    print(f"  [CSV] Curvas KM salvas em: {out}")


def brier_by_time_to_csv(eval_times: np.ndarray, brier_by_time: np.ndarray, output_dir: Path) -> None:
    df = pd.DataFrame({"tempo": np.round(eval_times, 4), "brier_score": np.round(brier_by_time, 6)})
    out = output_dir / "brier_score_por_tempo.csv"
    df.to_csv(out, index=False)
    print(f"  [CSV] Brier por tempo salvo em: {out}")


# ===========================================================================
# Salvar resultados (expandido)
# ===========================================================================

def save_results(
    output_path: Path,
    output_dir: Path,
    target: str,
    train_size: int,
    test_size: int,
    feature_cols: list[str],
    w_train_used: bool,
    weighting_label: str,
    device: str,
    cv: int,
    best_score: float,
    best_std: float,
    best_params: dict,
    metrics: dict,
    perm_df: pd.DataFrame,
    # Metricas adicionais
    km_result: dict,
    ks_result: dict,
    calib_df: pd.DataFrame,
    calib_eval_time: float,
    royston: dict,
    risk_dist: dict,
    net_benefit_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_times = metrics["eval_times"]
    auc_by_time = metrics["dynamic_auc_by_time"]
    brier_by_time = metrics["brier_by_time"]

    auc_time_df = pd.DataFrame({
        "tempo": np.round(eval_times, 4),
        "auc_dinamica": np.round(auc_by_time, 6),
        "brier_score": np.round(brier_by_time, 6),
    })

    # Exporta CSVs auxiliares
    km_groups_to_csv(km_result, output_dir, prefix="test")
    brier_by_time_to_csv(eval_times, brier_by_time, output_dir)
    ks_result["ks_by_decil"].to_csv(output_dir / "ks_por_decil.csv", index=False)
    calib_df.to_csv(output_dir / "calibracao_por_decil.csv", index=False)
    net_benefit_df.to_csv(output_dir / "net_benefit_dca.csv", index=False)
    risk_dist["hist_df"].to_csv(output_dir / "distribuicao_log_risco.csv", index=False)
    if not subgroup_df.empty:
        subgroup_df.to_csv(output_dir / "cindex_por_subgrupo.csv", index=False)

    report_lines = [
        "=" * 70,
        "DEEPSURV (Cox MLP) GPU (PyTorch) - RESULTADOS",
        "=" * 70,
        f"Data/hora: {datetime.now().isoformat(timespec='seconds')}",
        f"Coluna alvo: {target}",
        f"Dispositivo: {device}",
        f"Amostras de treino: {train_size}",
        f"Amostras de teste: {test_size}",
        f"Total de features selecionadas: {len(feature_cols)}",
        f"Peso de amostras (treino): {weighting_label}",
        f"Uso de pesos nas amostras: {'sim' if w_train_used else 'nao'}",
        f"KFold (cv): {cv}",
        "Estrategia de CV: StratifiedKFold por evento",
        "Metrica de Grid/CV: Uno C-index (IPCW)",
        "Loss: Negative Partial Log-Likelihood de Cox (Breslow)",
        "",
        "--- Grid Search manual com KFold ---",
        f"Melhor score de CV (Uno C-index): {best_score:.6f}",
        f"Desvio padrao do melhor score: {best_std:.6f}",
        "Melhores hiperparametros:",
    ]

    for k, v in best_params.items():
        report_lines.append(f"  {k}: {v}")

    report_lines += [
        "",
        "=" * 70,
        "METRICAS DE DISCRIMINACAO",
        "=" * 70,
        f"C-index treino            : {metrics['cindex_train']:.6f}",
        f"C-index teste             : {metrics['cindex_test']:.6f}",
        f"Gap C-index (overfit)     : {metrics['cindex_gap']:.6f}",
        f"Uno C-index teste (IPCW)  : {metrics['uno_cindex_test']:.6f}",
        f"AUC dinamica media        : {metrics['mean_dynamic_auc']:.6f}",
        "",
        "[D de Royston]",
        f"  D de Royston            : {royston['D_royston']}",
        f"  R2 de Royston           : {royston['R2_royston']}",
        "  (D > 1 = discriminacao razoavel; D > 2 = excelente)",
        "",
        "=" * 70,
        "METRICAS DE CALIBRACAO",
        "=" * 70,
        f"Brier score medio         : {metrics['mean_brier']:.6f}",
        f"Integrated Brier (IBS)    : {metrics['ibs']:.6f}",
        "  (IBS < 0.25 e considerado bom; IBS = 0.25 equivale ao modelo nulo)",
        "",
        f"[Calibracao por decil de risco — tempo de referencia: {calib_eval_time:.1f}]",
        "(Decil 1 = menor risco, Decil 10 = maior risco)",
        "(Razao E/O: 1.0 = calibracao perfeita; > 1 = subestima evento; < 1 = superestima)",
        calib_df.to_string(index=False),
        "",
        "=" * 70,
        "KS DE SOBREVIVENCIA (separacao entre grupos Evento vs. Censurado)",
        "=" * 70,
        f"KS estatistico (2-amostras): {ks_result['ks_stat']:.6f}",
        f"KS p-valor                 : {ks_result['ks_pval']:.6e}",
        f"KS maximo no decil         : {ks_result['ks_best_decil']} (KS = {ks_result['ks_best_value']:.6f})",
        "",
        "[KS por Decil de Risco — log-risco decrescente]",
        "(Analoga a tabela de KS / Ganho / Lift de classificacao)",
        ks_result["ks_by_decil"].to_string(index=False),
        "",
        "=" * 70,
        "CURVAS DE KAPLAN-MEIER POR GRUPO DE RISCO",
        "=" * 70,
        f"Log-rank test (baixo vs alto risco): chi2 = {km_result['logrank_chi2_low_vs_high']:.4f}, "
        f"p = {km_result['logrank_pval_low_vs_high']:.6e}",
        "",
    ]

    for gname, gdata in km_result["groups"].items():
        report_lines.append(
            f"  {gname}: n={gdata['n']}, eventos={gdata['n_events']}, "
            f"S(max_t) = {gdata['survival'][-1]:.4f}"
        )

    report_lines += [
        "(Curvas completas exportadas para: results/test_curvas_km.csv)",
        "",
        "=" * 70,
        "DISTRIBUICAO DOS LOG-RISCOS POR STATUS",
        "=" * 70,
        risk_dist["stats_df"].to_string(index=False),
        "",
        "=" * 70,
        "NET BENEFIT — DECISION CURVE ANALYSIS (simplificada)",
        "=" * 70,
        f"(Baseada no tempo de referencia: {calib_eval_time:.1f})",
        "(Arquivo completo: results/net_benefit_dca.csv)",
        net_benefit_df[
            net_benefit_df["threshold_pt"].isin(
                net_benefit_df["threshold_pt"].quantile(np.linspace(0, 1, 10)).values
            )
        ].to_string(index=False),
        "",
        "=" * 70,
        "AUC DINAMICA E BRIER POR TEMPO",
        "=" * 70,
        auc_time_df.to_string(index=False),
        "",
    ]

    if not subgroup_df.empty:
        report_lines += [
            "=" * 70,
            "C-INDEX POR SUBGRUPO",
            "=" * 70,
            subgroup_df.to_string(index=False),
            "",
        ]

    report_lines += [
        "=" * 70,
        "TOP 20 PERMUTATION IMPORTANCE (Uno C-index no teste)",
        "=" * 70,
        perm_df.head(20).to_string(index=False),
        "",
    ]

    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nRelatorio salvo em: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"Usando dispositivo: {args.device}")

    (
        x_train, x_test,
        y_train_surv, y_test_surv,
        event_train, time_train,
        event_test, time_test,
        w_class, feature_cols,
        train_df, test_df,
    ) = load_split_data(
        train_path=args.train_path,
        test_path=args.test_path,
        target=args.target,
        time_col=args.time_col,
        event_time_col=args.event_time_col,
    )

    # Selecao de pesos
    w_train = None
    weighting_label = "none"
    if args.weighting == "class_weight" and w_class is not None:
        w_train = w_class
        weighting_label = "class_weight"
    elif args.weighting == "ipcw":
        w_train = compute_ipcw_weights(y_train_surv, min_prob=args.ipcw_min_prob)
        weighting_label = f"ipcw(min_prob={args.ipcw_min_prob})"

    # Grid Search
    best_score, best_std, best_params = run_grid_search(
        x_train=x_train,
        y_train_surv=y_train_surv,
        event_train=event_train,
        time_train=time_train,
        w_train=w_train,
        cv=args.cv,
        random_state=args.random_state,
        device=args.device,
        min_events_per_fold=args.min_events_per_fold,
    )

    print(f"\nMelhor Uno C-index CV: {best_score:.6f} +/- {best_std:.6f}")
    print(f"Melhores params: {best_params}")

    # Treina modelo final com todos os dados de treino
    best_model, best_scaler = fit_deepsurv(
        x_np=x_train.to_numpy(),
        time_np=time_train,
        event_np=event_train,
        params=best_params,
        device=args.device,
        random_state=args.random_state,
        sample_weight=w_train,
    )

    log_risk_train = predict_log_risk(best_model, x_train.to_numpy(), best_scaler, args.device)
    log_risk_test = predict_log_risk(best_model, x_test.to_numpy(), best_scaler, args.device)

    eval_times = build_eval_times(
        y_train_surv=y_train_surv,
        y_test_surv=y_test_surv,
        min_eval_times=args.min_eval_times,
        max_eval_times=args.max_eval_times,
    )

    metrics = compute_survival_metrics(
        log_risk_train=log_risk_train,
        log_risk_test=log_risk_test,
        y_train_surv=y_train_surv,
        y_test_surv=y_test_surv,
        time_train=time_train,
        event_train=event_train,
        eval_times=eval_times,
    )

    print("\nCalculando metricas adicionais...")

    # --- Metricas adicionais ---

    # 1. KM por grupos de risco (no conjunto de teste)
    km_result = compute_km_risk_groups(log_risk_test, time_test, event_test, n_groups=3)
    print(
        f"  Log-rank (baixo vs alto risco): chi2={km_result['logrank_chi2_low_vs_high']:.4f}, "
        f"p={km_result['logrank_pval_low_vs_high']:.4e}"
    )

    # 2. KS de sobrevivência
    ks_result = compute_survival_ks(log_risk_test, event_test)
    print(f"  KS estatistico: {ks_result['ks_stat']:.6f} (decil {ks_result['ks_best_decil']})")

    # 3. Calibracao por decil — usa tempo mediano de avaliacao como referencia
    calib_eval_time = float(np.median(eval_times))
    calib_time_idx = int(np.argmin(np.abs(eval_times - calib_eval_time)))
    survival_at_ref = metrics["survival_test"][:, calib_time_idx]
    calib_df = compute_calibration_by_decil(
        log_risk=log_risk_test,
        time=time_test,
        event=event_test,
        survival_at_t=survival_at_ref,
        eval_time=calib_eval_time,
    )

    # 4. D de Royston
    royston = compute_royston_d(log_risk_test, event_test)
    print(f"  D de Royston: {royston['D_royston']}, R2: {royston['R2_royston']}")

    # 5. Distribuicao dos log-riscos
    risk_dist = compute_risk_score_distribution(log_risk_test, event_test)

    # 6. Net Benefit
    net_benefit_df = compute_net_benefit(
        log_risk=log_risk_test,
        event=event_test,
        time=time_test,
        eval_time=calib_eval_time,
        survival_at_t=survival_at_ref,
    )

    # 7. C-index por subgrupo
    subgroup_series = None
    if args.subgroup_col and args.subgroup_col in test_df.columns:
        subgroup_series = test_df[args.subgroup_col].reset_index(drop=True)
    subgroup_df = compute_cindex_by_subgroup(
        log_risk=log_risk_test,
        y_train_surv=y_train_surv,
        y_test_surv=y_test_surv,
        event_test=event_test,
        time_test=time_test,
        subgroup_series=subgroup_series,
    )

    # Permutation Importance
    perm_df = permutation_importance_cindex(
        model=best_model,
        scaler=best_scaler,
        x_test=x_test,
        y_train_surv=y_train_surv,
        y_test_surv=y_test_surv,
        n_repeats=args.perm_repeats,
        random_state=args.random_state,
        device=args.device,
    )

    save_results(
        output_path=args.output,
        output_dir=args.output_dir,
        target=args.target,
        train_size=len(x_train),
        test_size=len(x_test),
        feature_cols=feature_cols,
        w_train_used=w_train is not None,
        weighting_label=weighting_label,
        device=args.device,
        cv=args.cv,
        best_score=best_score,
        best_std=best_std,
        best_params=best_params,
        metrics=metrics,
        perm_df=perm_df,
        km_result=km_result,
        ks_result=ks_result,
        calib_df=calib_df,
        calib_eval_time=calib_eval_time,
        royston=royston,
        risk_dist=risk_dist,
        net_benefit_df=net_benefit_df,
        subgroup_df=subgroup_df,
    )

    print(f"\nTreinamento DeepSurv concluido. Resultados salvos em: {args.output}")


if __name__ == "__main__":
    main()