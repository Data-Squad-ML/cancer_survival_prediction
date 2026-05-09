#!/usr/bin/env python3

"""Treina rede neural MLP em GPU (PyTorch) seguindo a logica do notebook e salva relatorio TXT."""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import product as iterproduct
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Colunas de features (mesmo padrao do script de Random Forest)
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
# Definicao da MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Rede neural totalmente conectada (Multi-Layer Perceptron)."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        output_dim: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
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
# Auxiliares
# ---------------------------------------------------------------------------

def default_data_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino de MLP em GPU com Grid Search manual, KFold e ROC-AUC."
    )
    parser.add_argument("--train-path", type=Path, default=default_data_path("data_train.csv"))
    parser.add_argument("--test-path", type=Path, default=default_data_path("data_test.csv"))
    parser.add_argument("--target", type=str, default="status_vital")
    parser.add_argument("--output", type=Path, default=Path("neural_network_gpu_results.txt"))
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--perm-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo PyTorch: 'cuda' ou 'cpu'.",
    )
    return parser.parse_args()


def get_auc_mode(y_train: pd.Series) -> str:
    return "binary" if int(y_train.nunique()) <= 2 else "multiclass"


def compute_auc(y_true: np.ndarray, y_proba: np.ndarray, auc_mode: str) -> float:
    if auc_mode == "binary":
        return roc_auc_score(y_true, y_proba[:, 1])
    return roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")


def build_feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    feature_cols_ohe = [
        col for col in train_df.columns if any(col.startswith(p) for p in OHE_PREFIXES)
    ]
    selected = [col for col in (FEATURE_COLS_BASE + feature_cols_ohe) if col in train_df.columns]
    selected = [col for col in selected if col in test_df.columns]

    if not selected:
        raise ValueError("Nenhuma feature selecionada foi encontrada nos dados de treino/teste.")

    return selected


def load_split_data(train_path: Path, test_path: Path, target: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target not in train_df.columns or target not in test_df.columns:
        raise ValueError(f"A coluna alvo '{target}' nao existe em ambos os arquivos.")

    train_df = train_df.dropna(subset=[target]).copy()
    test_df = test_df.dropna(subset=[target]).copy()

    feature_cols = build_feature_columns(train_df, test_df)

    x_train = train_df[feature_cols].astype(float)
    y_train = train_df[target].astype(int)
    x_test = test_df[feature_cols].astype(float)
    y_test = test_df[target].astype(int)

    w_train = train_df["class_weight"].astype(float) if "class_weight" in train_df.columns else None

    return x_train, y_train, x_test, y_test, w_train, feature_cols


# ---------------------------------------------------------------------------
# Grid de hiperparametros
# ---------------------------------------------------------------------------

def param_grid():
    grid = {
        "hidden_sizes": [[256, 128], [512, 256, 128], [128, 64]],
        "dropout": [0.0, 0.3],
        "lr": [1e-3, 3e-4],
        "batch_size": [512, 1024],
        "epochs": [50, 100],
        "activation": ["relu", "leaky_relu"],
    }
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in iterproduct(*values):
        yield dict(zip(keys, combo))


# ---------------------------------------------------------------------------
# Treinamento de uma MLP
# ---------------------------------------------------------------------------

def fit_mlp(
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    params: dict,
    output_dim: int,
    device: str,
    random_state: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[MLP, StandardScaler]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train_np).astype(np.float32)
    y_arr = y_train_np.astype(np.int64)

    x_tensor = torch.tensor(x_scaled, device=device)
    y_tensor = torch.tensor(y_arr, device=device)
    w_tensor = (
        torch.tensor(sample_weight.astype(np.float32), device=device)
        if sample_weight is not None
        else None
    )

    model = MLP(
        input_dim=x_scaled.shape[1],
        hidden_sizes=params["hidden_sizes"],
        output_dim=output_dim,
        dropout=params["dropout"],
        activation=params["activation"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    batch_size = params["batch_size"]
    n = len(x_tensor)

    model.train()
    for _ in range(params["epochs"]):
        # shuffle
        perm = torch.randperm(n, device=device)
        x_tensor = x_tensor[perm]
        y_tensor = y_tensor[perm]
        if w_tensor is not None:
            w_tensor = w_tensor[perm]

        for start in range(0, n, batch_size):
            xb = x_tensor[start : start + batch_size]
            yb = y_tensor[start : start + batch_size]

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)

            if w_tensor is not None:
                wb = w_tensor[start : start + batch_size]
                loss = (loss * wb).mean()
            else:
                loss = loss.mean()

            loss.backward()
            optimizer.step()

    return model, scaler


def predict_proba_mlp(
    model: MLP,
    x_np: np.ndarray,
    scaler: StandardScaler,
    device: str,
) -> np.ndarray:
    model.eval()
    x_scaled = scaler.transform(x_np).astype(np.float32)
    x_tensor = torch.tensor(x_scaled, device=device)

    with torch.no_grad():
        logits = model(x_tensor)
        proba = torch.softmax(logits, dim=-1).cpu().numpy()

    return proba


# ---------------------------------------------------------------------------
# Grid Search com KFold
# ---------------------------------------------------------------------------

def run_grid_search(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series | None,
    cv: int,
    random_state: int,
    device: str,
):
    auc_mode = get_auc_mode(y_train)
    output_dim = int(y_train.nunique())
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    all_params = list(param_grid())
    print(f"Total de combinacoes no grid: {len(all_params)}")

    best_score = -1.0
    best_std = 0.0
    best_params = None

    x_np = x_train.to_numpy()
    y_np = y_train.to_numpy()
    w_np = w_train.to_numpy() if w_train is not None else None

    for i, params in enumerate(all_params, start=1):
        fold_scores = []

        for fold_train_idx, fold_val_idx in cv_splitter.split(x_np):
            x_fold_train = x_np[fold_train_idx]
            y_fold_train = y_np[fold_train_idx]
            x_fold_val = x_np[fold_val_idx]
            y_fold_val = y_np[fold_val_idx]
            w_fold_train = w_np[fold_train_idx] if w_np is not None else None

            model, scaler = fit_mlp(
                x_train_np=x_fold_train,
                y_train_np=y_fold_train,
                params=params,
                output_dim=output_dim,
                device=device,
                random_state=random_state,
                sample_weight=w_fold_train,
            )

            y_proba_fold = predict_proba_mlp(model, x_fold_val, scaler, device)
            fold_auc = compute_auc(y_fold_val, y_proba_fold, auc_mode)
            fold_scores.append(fold_auc)

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))

        print(f"[{i}/{len(all_params)}] cv_auc={mean_score:.6f} +/- {std_score:.6f} params={params}")

        if mean_score > best_score:
            best_score = mean_score
            best_std = std_score
            best_params = params

    return best_score, best_std, best_params, auc_mode, output_dim


# ---------------------------------------------------------------------------
# Permutation Importance
# ---------------------------------------------------------------------------

def permutation_importance_auc(
    model: MLP,
    scaler: StandardScaler,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    auc_mode: str,
    n_repeats: int,
    random_state: int,
    device: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    x_np = x_test.to_numpy()
    y_np = y_test.to_numpy()

    base_proba = predict_proba_mlp(model, x_np, scaler, device)
    base_auc = compute_auc(y_np, base_proba, auc_mode)

    rows = []
    for j, col in enumerate(x_test.columns):
        scores = []
        for _ in range(n_repeats):
            x_perm = x_np.copy()
            x_perm[:, j] = rng.permutation(x_perm[:, j])
            perm_proba = predict_proba_mlp(model, x_perm, scaler, device)
            perm_auc = compute_auc(y_np, perm_proba, auc_mode)
            scores.append(base_auc - perm_auc)

        rows.append(
            {
                "feature": col,
                "importance_mean": float(np.mean(scores)),
                "importance_std": float(np.std(scores)),
            }
        )

    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


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
    device: str,
    best_score: float,
    best_std: float,
    best_params: dict,
    train_auc: float,
    test_auc: float,
    perm_df: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "=" * 70,
        "REDE NEURAL MLP GPU (PyTorch) - RESULTADOS",
        "=" * 70,
        f"Data/hora: {datetime.now().isoformat(timespec='seconds')}",
        f"Coluna alvo: {target}",
        f"Dispositivo: {device}",
        f"Amostras de treino: {train_size}",
        f"Amostras de teste: {test_size}",
        f"Total de features selecionadas: {len(feature_cols)}",
        f"Uso de class_weight como sample_weight: {'sim' if w_train_used else 'nao'}",
        "Metrica de Grid/CV: roc_auc (KFold)",
        "",
        "--- Grid Search manual com KFold ---",
        f"Melhor score de CV (AUC): {best_score:.6f}",
        f"Desvio padrao do melhor score: {best_std:.6f}",
        f"Melhores hiperparametros:",
    ]

    for k, v in best_params.items():
        report_lines.append(f"  {k}: {v}")

    report_lines += [
        "",
        "--- Avaliacao AUC ---",
        f"AUC treino: {train_auc:.6f}",
        f"AUC teste : {test_auc:.6f}",
        f"Gap       : {train_auc - test_auc:.6f}",
        "",
        "--- Top 20 Permutation Importance (teste) ---",
        perm_df.head(20).to_string(index=False),
        "",
    ]

    output_path.write_text("\n".join(report_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    args = parse_args()

    print(f"Usando dispositivo: {args.device}")

    x_train, y_train, x_test, y_test, w_train, feature_cols = load_split_data(
        train_path=args.train_path,
        test_path=args.test_path,
        target=args.target,
    )

    best_score, best_std, best_params, auc_mode, output_dim = run_grid_search(
        x_train=x_train,
        y_train=y_train,
        w_train=w_train,
        cv=args.cv,
        random_state=args.random_state,
        device=args.device,
    )

    # Treina modelo final com todos os dados de treino
    best_model, best_scaler = fit_mlp(
        x_train_np=x_train.to_numpy(),
        y_train_np=y_train.to_numpy(),
        params=best_params,
        output_dim=output_dim,
        device=args.device,
        random_state=args.random_state,
        sample_weight=w_train.to_numpy() if w_train is not None else None,
    )

    train_proba = predict_proba_mlp(best_model, x_train.to_numpy(), best_scaler, args.device)
    test_proba = predict_proba_mlp(best_model, x_test.to_numpy(), best_scaler, args.device)

    train_auc = compute_auc(y_train.to_numpy(), train_proba, auc_mode)
    test_auc = compute_auc(y_test.to_numpy(), test_proba, auc_mode)

    perm_df = permutation_importance_auc(
        model=best_model,
        scaler=best_scaler,
        x_test=x_test,
        y_test=y_test,
        auc_mode=auc_mode,
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
        device=args.device,
        best_score=best_score,
        best_std=best_std,
        best_params=best_params,
        train_auc=train_auc,
        test_auc=test_auc,
        perm_df=perm_df,
    )

    print(f"Treinamento concluido. Resultados salvos em: {args.output}")


if __name__ == "__main__":
    main()