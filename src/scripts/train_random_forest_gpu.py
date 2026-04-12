#!/usr/bin/env python3

"""Treina Random Forest em GPU (cuML) seguindo a logica do notebook e salva relatorio TXT."""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import product
from pathlib import Path

import cupy as cp
import cudf
import numpy as np
import pandas as pd
from cuml.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


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


def default_data_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino de Random Forest em GPU com Grid Search manual, KFold e ROC-AUC."
    )
    parser.add_argument("--train-path", type=Path, default=default_data_path("data_train.csv"))
    parser.add_argument("--test-path", type=Path, default=default_data_path("data_test.csv"))
    parser.add_argument("--target", type=str, default="status_vital")
    parser.add_argument("--output", type=Path, default=Path("random_forest_gpu_results.txt"))
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--perm-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _to_numpy(x):
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    return np.asarray(x)


def build_feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    feature_cols_ohe = [
        col for col in train_df.columns if any(col.startswith(prefix) for prefix in OHE_PREFIXES)
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


def get_auc_mode(y_train: pd.Series) -> str:
    return "binary" if int(y_train.nunique()) <= 2 else "multiclass"


def compute_auc(y_true, y_proba, auc_mode: str) -> float:
    if auc_mode == "binary":
        return roc_auc_score(y_true, y_proba[:, 1])
    return roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")


def param_grid():
    grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def _fit_gpu_rf(
    x_train_df: pd.DataFrame,
    y_train_sr: pd.Series,
    params: dict,
    random_state: int,
    sample_weight: pd.Series | None,
):
    model = RandomForestClassifier(
        random_state=random_state,
        n_streams=1,
        **params,
    )

    x_gpu = cudf.DataFrame.from_pandas(x_train_df)
    y_gpu = cudf.Series(y_train_sr.reset_index(drop=True))

    if sample_weight is not None:
        w_gpu = cudf.Series(sample_weight.reset_index(drop=True))
        try:
            model.fit(x_gpu, y_gpu, sample_weight=w_gpu)
        except TypeError:
            model.fit(x_gpu, y_gpu)
    else:
        model.fit(x_gpu, y_gpu)

    return model


def run_grid_search(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series | None,
    cv: int,
    random_state: int,
):
    auc_mode = get_auc_mode(y_train)
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    all_params = list(param_grid())
    print(f"Total de combinacoes no grid: {len(all_params)}")

    best_score = -1.0
    best_std = 0.0
    best_params = None

    x_np = x_train.to_numpy()
    y_np = y_train.to_numpy()

    for i, params in enumerate(all_params, start=1):
        fold_scores = []

        for fold_train_idx, fold_val_idx in cv_splitter.split(x_np):
            x_fold_train = x_train.iloc[fold_train_idx]
            y_fold_train = y_train.iloc[fold_train_idx]
            x_fold_val = x_train.iloc[fold_val_idx]
            y_fold_val = y_train.iloc[fold_val_idx].to_numpy()
            w_fold_train = w_train.iloc[fold_train_idx] if w_train is not None else None

            model = _fit_gpu_rf(
                x_train_df=x_fold_train,
                y_train_sr=y_fold_train,
                params=params,
                random_state=random_state,
                sample_weight=w_fold_train,
            )

            y_proba_fold = _to_numpy(model.predict_proba(cudf.DataFrame.from_pandas(x_fold_val)))
            fold_auc = compute_auc(y_fold_val, y_proba_fold, auc_mode)
            fold_scores.append(fold_auc)

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))

        print(f"[{i}/{len(all_params)}] cv_auc={mean_score:.6f} +/- {std_score:.6f} params={params}")

        if mean_score > best_score:
            best_score = mean_score
            best_std = std_score
            best_params = params

    return best_score, best_std, best_params, auc_mode


def permutation_importance_auc(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    auc_mode: str,
    n_repeats: int,
    random_state: int,
):
    rng = np.random.default_rng(random_state)
    base_proba = _to_numpy(model.predict_proba(cudf.DataFrame.from_pandas(x_test)))
    base_auc = compute_auc(y_test.to_numpy(), base_proba, auc_mode)

    rows = []
    for col in x_test.columns:
        scores = []
        for _ in range(n_repeats):
            x_perm = x_test.copy()
            x_perm[col] = rng.permutation(x_perm[col].to_numpy())
            perm_proba = _to_numpy(model.predict_proba(cudf.DataFrame.from_pandas(x_perm)))
            perm_auc = compute_auc(y_test.to_numpy(), perm_proba, auc_mode)
            scores.append(base_auc - perm_auc)

        rows.append(
            {
                "feature": col,
                "importance_mean": float(np.mean(scores)),
                "importance_std": float(np.std(scores)),
            }
        )

    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def save_results(
    output_path: Path,
    target: str,
    train_size: int,
    test_size: int,
    feature_cols: list[str],
    w_train_used: bool,
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
        "RANDOM FOREST GPU (cuML) - RESULTADOS",
        "=" * 70,
        f"Data/hora: {datetime.now().isoformat(timespec='seconds')}",
        f"Coluna alvo: {target}",
        f"Amostras de treino: {train_size}",
        f"Amostras de teste: {test_size}",
        f"Total de features selecionadas: {len(feature_cols)}",
        f"Uso de class_weight como sample_weight: {'sim' if w_train_used else 'nao'}",
        "Metrica de Grid/CV: roc_auc (KFold)",
        "",
        "--- Grid Search manual com KFold ---",
        f"Melhor score de CV (AUC): {best_score:.6f}",
        f"Desvio padrao do melhor score: {best_std:.6f}",
        f"Melhores hiperparametros: {best_params}",
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


def main() -> None:
    args = parse_args()

    x_train, y_train, x_test, y_test, w_train, feature_cols = load_split_data(
        train_path=args.train_path,
        test_path=args.test_path,
        target=args.target,
    )

    best_score, best_std, best_params, auc_mode = run_grid_search(
        x_train=x_train,
        y_train=y_train,
        w_train=w_train,
        cv=args.cv,
        random_state=args.random_state,
    )

    best_model = _fit_gpu_rf(
        x_train_df=x_train,
        y_train_sr=y_train,
        params=best_params,
        random_state=args.random_state,
        sample_weight=w_train,
    )

    train_proba = _to_numpy(best_model.predict_proba(cudf.DataFrame.from_pandas(x_train)))
    test_proba = _to_numpy(best_model.predict_proba(cudf.DataFrame.from_pandas(x_test)))

    train_auc = compute_auc(y_train.to_numpy(), train_proba, auc_mode)
    test_auc = compute_auc(y_test.to_numpy(), test_proba, auc_mode)

    perm_df = permutation_importance_auc(
        model=best_model,
        x_test=x_test,
        y_test=y_test,
        auc_mode=auc_mode,
        n_repeats=args.perm_repeats,
        random_state=args.random_state,
    )

    save_results(
        output_path=args.output,
        target=args.target,
        train_size=len(x_train),
        test_size=len(x_test),
        feature_cols=feature_cols,
        w_train_used=w_train is not None,
        best_score=best_score,
        best_std=best_std,
        best_params=best_params,
        train_auc=train_auc,
        test_auc=test_auc,
        perm_df=perm_df,
    )

    print(f"Treinamento em GPU concluido. Resultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
