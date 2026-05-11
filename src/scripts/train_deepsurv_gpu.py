#!/usr/bin/env python3

"""Treina DeepSurv (Cox MLP) em GPU (PyTorch) com Grid Search, KFold e metricas de survival."""

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
    """Negative partial log-likelihood de Cox com aproximacao de Breslow.

    Ordena por tempo decrescente. Para cada evento i, o risco acumulado e a
    soma dos log-riscos de todas as amostras ainda em risco (tempo >= t_i).
    """
    # Ordena por tempo decrescente
    order = torch.argsort(time, descending=True)
    log_risk = log_risk[order]
    event = event[order]
    if sample_weight is not None:
        sample_weight = sample_weight[order]

    # Log-sum-exp cumulativo (soma dos riscos acumulados)
    log_cumsum_hazard = torch.logcumsumexp(log_risk, dim=0)

    # Contribuicao de cada evento: log_risk_i - log(sum_risk_at_risk_i)
    uncensored_mask = event.bool()
    event_log_risk = log_risk[uncensored_mask]
    event_log_cumsum = log_cumsum_hazard[uncensored_mask]

    # log P(T_i, E_i | theta) por evento
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
    # Convencao: evento = obito (status_vital == 0), censura = vivo (status_vital == 1)
    event = (df[target].astype(int) == 0).to_numpy(dtype=bool)
    time = np.where(event, df[event_time_col].to_numpy(), df[time_col].to_numpy()).astype(float)
    # Garante tempos positivos
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

    # Imputa NaN com mediana antes de passar para o modelo
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    x_train = pd.DataFrame(
        imputer.fit_transform(train_df[feature_cols].astype(np.float32)),
        columns=feature_cols,
    )
    x_test = pd.DataFrame(
        imputer.transform(test_df[feature_cols].astype(np.float32)),
        columns=feature_cols,
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

    # Learning rate scheduler: reduz na platô
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

            # Pula batch com menos de 2 eventos (loss indefinida)
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
    n_samples = len(event_train)

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

            # Pula fold sem eventos suficientes
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

                # Monta y_surv do fold de treino para Uno C-index
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
# Metricas de survival
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
    # Ordena por tempo
    order = np.argsort(time_train)
    t_sorted = time_train[order]
    e_sorted = event_train[order]
    r_sorted = np.exp(log_risk_train[order])

    # Calcula hazard baseline acumulada (Nelson-Aalen discreto com Breslow)
    unique_times = np.unique(t_sorted[e_sorted])
    h0 = np.zeros(len(unique_times))

    for j, t in enumerate(unique_times):
        at_risk = np.sum(r_sorted[t_sorted >= t])
        n_events = np.sum(e_sorted[t_sorted == t])
        h0[j] = n_events / max(at_risk, 1e-10)

    H0_cumulative = np.cumsum(h0)

    # Interpola H0 para eval_times
    H0_at_eval = np.interp(eval_times, unique_times, H0_cumulative, left=0.0, right=H0_cumulative[-1])

    # S(t | x) = exp(-H0(t) * exp(log_risk))
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

    # Brier score requer funcao de sobrevivencia estimada
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


# ---------------------------------------------------------------------------
# Salvar resultados
# ---------------------------------------------------------------------------

def save_results(
    output_path: Path,
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
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eval_times = metrics["eval_times"]
    auc_by_time = metrics["dynamic_auc_by_time"]
    auc_time_df = pd.DataFrame({
        "tempo": np.round(eval_times, 4),
        "auc_dinamica": np.round(auc_by_time, 6),
    })

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
        "--- Metricas de Qualidade (treino/teste) ---",
        f"C-index treino        : {metrics['cindex_train']:.6f}",
        f"C-index teste         : {metrics['cindex_test']:.6f}",
        f"Gap C-index           : {metrics['cindex_gap']:.6f}",
        f"Uno C-index teste     : {metrics['uno_cindex_test']:.6f}",
        f"AUC dinamica media    : {metrics['mean_dynamic_auc']:.6f}",
        f"Brier score medio     : {metrics['mean_brier']:.6f}",
        f"Integrated Brier (IBS): {metrics['ibs']:.6f}",
        "",
        "--- AUC Dinamica por Tempo ---",
        auc_time_df.to_string(index=False),
        "",
        "--- Top 20 Permutation Importance (Uno C-index no teste) ---",
        perm_df.head(20).to_string(index=False),
        "",
    ]

    output_path.write_text("\n".join(report_lines), encoding="utf-8")


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
    )

    print(f"\nTreinamento DeepSurv concluido. Resultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
