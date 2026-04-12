#!/usr/bin/env python3

"""Treina Random Forest em GPU (RAPIDS cuML) com busca de hiperparametros e salva relatorio TXT."""

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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino de Random Forest em GPU (cuML) com busca de hiperparametros."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data_train.csv"),
        help="Caminho do CSV de treino.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data_test.csv"),
        help="Caminho do CSV de teste.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="status_vital",
        help="Nome da coluna alvo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("random_forest_gpu_results.txt"),
        help="Arquivo TXT para salvar os resultados.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Quantidade de folds da validacao cruzada.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semente para reprodutibilidade.",
    )
    return parser.parse_args()


def _to_numpy(x):
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    return np.asarray(x)


def load_split_data(train_path: Path, test_path: Path, target: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target not in train_df.columns or target not in test_df.columns:
        raise ValueError(f"A coluna alvo '{target}' nao existe em ambos os arquivos.")

    train_df = train_df.dropna(subset=[target]).copy()
    test_df = test_df.dropna(subset=[target]).copy()

    x_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    x_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # Dados ja preparados no notebook; aqui so garantimos preenchimento simples para o modelo.
    medians = x_train.median(numeric_only=True)
    x_train = x_train.fillna(medians)
    x_test = x_test.fillna(medians)

    return x_train, y_train, x_test, y_test


def param_grid():
    grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [10, 20, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def cross_val_score_gpu(x_train: pd.DataFrame, y_train: pd.Series, params: dict, cv: int, random_state: int) -> float:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    fold_scores = []

    x_np = x_train.to_numpy()
    y_np = y_train.to_numpy()

    for train_idx, val_idx in skf.split(x_np, y_np):
        x_fold_train = cudf.DataFrame.from_pandas(x_train.iloc[train_idx])
        y_fold_train = cudf.Series(y_train.iloc[train_idx].reset_index(drop=True))
        x_fold_val = cudf.DataFrame.from_pandas(x_train.iloc[val_idx])
        y_fold_val = y_train.iloc[val_idx].to_numpy()

        model = RandomForestClassifier(
            random_state=random_state,
            n_streams=1,
            **params,
        )
        model.fit(x_fold_train, y_fold_train)

        y_pred = model.predict(x_fold_val)
        y_pred_np = _to_numpy(y_pred).astype(y_fold_val.dtype, copy=False)
        fold_scores.append(f1_score(y_fold_val, y_pred_np, average="weighted"))

    return float(np.mean(fold_scores))


def run_grid_search(x_train: pd.DataFrame, y_train: pd.Series, cv: int, random_state: int):
    best_score = -1.0
    best_params = None

    all_params = list(param_grid())
    print(f"Total de combinacoes no grid: {len(all_params)}")

    for i, params in enumerate(all_params, start=1):
        score = cross_val_score_gpu(
            x_train=x_train,
            y_train=y_train,
            params=params,
            cv=cv,
            random_state=random_state,
        )
        print(f"[{i}/{len(all_params)}] score={score:.6f} params={params}")
        if score > best_score:
            best_score = score
            best_params = params

    return best_score, best_params


def train_best_model(x_train: pd.DataFrame, y_train: pd.Series, best_params: dict, random_state: int):
    model = RandomForestClassifier(
        random_state=random_state,
        n_streams=1,
        **best_params,
    )
    model.fit(cudf.DataFrame.from_pandas(x_train), cudf.Series(y_train.reset_index(drop=True)))
    return model


def save_results(
    output_path: Path,
    target: str,
    train_size: int,
    test_size: int,
    best_score: float,
    best_params: dict,
    y_test,
    y_pred,
    feature_names,
    feature_importances,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    importance_series = pd.Series(feature_importances, index=feature_names)
    importance_series = importance_series.sort_values(ascending=False)
    top_importances = importance_series.head(20)

    report_lines = [
        "=" * 70,
        "RANDOM FOREST GPU (cuML) - RESULTADOS",
        "=" * 70,
        f"Data/hora: {datetime.now().isoformat(timespec='seconds')}",
        f"Coluna alvo: {target}",
        f"Amostras de treino: {train_size}",
        f"Amostras de teste: {test_size}",
        "",
        "--- Grid Search ---",
        f"Melhor score de CV (f1_weighted): {best_score:.6f}",
        f"Melhores hiperparametros: {best_params}",
        "",
        "--- Avaliacao no conjunto de teste ---",
        f"Accuracy: {accuracy_score(y_test, y_pred):.6f}",
        f"F1 weighted: {f1_score(y_test, y_pred, average='weighted'):.6f}",
        "",
        "Matriz de confusao:",
        str(confusion_matrix(y_test, y_pred)),
        "",
        "Classification report:",
        classification_report(y_test, y_pred, digits=4),
        "",
        "--- Feature Importance (Top 20) ---",
        top_importances.to_string(),
        "",
    ]

    output_path.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    x_train, y_train, x_test, y_test = load_split_data(
        train_path=args.train_path,
        test_path=args.test_path,
        target=args.target,
    )

    best_score, best_params = run_grid_search(
        x_train=x_train,
        y_train=y_train,
        cv=args.cv,
        random_state=args.random_state,
    )

    best_model = train_best_model(
        x_train=x_train,
        y_train=y_train,
        best_params=best_params,
        random_state=args.random_state,
    )

    x_test_gpu = cudf.DataFrame.from_pandas(x_test)
    y_pred = _to_numpy(best_model.predict(x_test_gpu)).astype(y_test.dtype, copy=False)

    feature_importances = _to_numpy(best_model.feature_importances_)

    save_results(
        output_path=args.output,
        target=args.target,
        train_size=len(x_train),
        test_size=len(x_test),
        best_score=best_score,
        best_params=best_params,
        y_test=y_test.to_numpy(),
        y_pred=y_pred,
        feature_names=x_train.columns,
        feature_importances=feature_importances,
    )

    print(f"Treinamento em GPU concluido. Resultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
