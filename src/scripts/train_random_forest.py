#!/usr/bin/env python3

"""Treina Random Forest com Grid Search e salva resultados em TXT."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino de Random Forest com Grid Search usando CSVs de treino e teste."
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
        default=Path("random_forest_results.txt"),
        help="Arquivo TXT para salvar os resultados.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Quantidade de folds da validação cruzada no Grid Search.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Quantidade de processos em paralelo no Grid Search (-1 usa todos).",
    )
    return parser.parse_args()


def load_split_data(train_path: Path, test_path: Path, target: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target not in train_df.columns or target not in test_df.columns:
        raise ValueError(f"A coluna alvo '{target}' nao existe em ambos os arquivos.")

    train_df = train_df.dropna(subset=[target])
    test_df = test_df.dropna(subset=[target])

    x_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    x_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    return x_train, y_train, x_test, y_test


def run_training(x_train, y_train, cv: int, n_jobs: int) -> GridSearchCV:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )

    param_grid = {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [None, 10, 20, 40],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
        "model__class_weight": [None, "balanced"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )
    grid_search.fit(x_train, y_train)
    return grid_search


def save_results(
    output_path: Path,
    target: str,
    train_size: int,
    test_size: int,
    grid_search: GridSearchCV,
    feature_names,
    y_test,
    y_pred,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rf_model = grid_search.best_estimator_.named_steps["model"]
    importance_series = pd.Series(rf_model.feature_importances_, index=feature_names)
    importance_series = importance_series.sort_values(ascending=False)
    top_importances = importance_series.head(20)

    report_lines = [
        "=" * 70,
        "RANDOM FOREST - RESULTADOS",
        "=" * 70,
        f"Data/hora: {datetime.now().isoformat(timespec='seconds')}",
        f"Coluna alvo: {target}",
        f"Amostras de treino: {train_size}",
        f"Amostras de teste: {test_size}",
        "",
        "--- Grid Search ---",
        f"Melhor score de CV (f1_weighted): {grid_search.best_score_:.6f}",
        f"Melhores hiperparametros: {grid_search.best_params_}",
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

    grid_search = run_training(x_train=x_train, y_train=y_train, cv=args.cv, n_jobs=args.n_jobs)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)

    save_results(
        output_path=args.output,
        target=args.target,
        train_size=len(x_train),
        test_size=len(x_test),
        grid_search=grid_search,
        feature_names=x_train.columns,
        y_test=y_test,
        y_pred=y_pred,
    )

    print(f"Treinamento concluido. Resultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
