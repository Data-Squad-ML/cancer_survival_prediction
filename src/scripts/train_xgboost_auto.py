#!/usr/bin/env python3

"""Treina XGBoost com Grid Search, usa GPU automaticamente quando disponivel e salva relatorio TXT."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino de XGBoost com Grid Search e selecao automatica de GPU/CPU."
    )
    parser.add_argument("--train-path", type=Path, default=Path("data_train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data_test.csv"))
    parser.add_argument("--target", type=str, default="status_vital")
    parser.add_argument("--output", type=Path, default=Path("xgboost_results.txt"))
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
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


def resolve_objective(y_train: pd.Series) -> tuple[str, int | None]:
    n_classes = int(y_train.nunique())
    if n_classes <= 2:
        return "binary:logistic", None
    return "multi:softprob", n_classes


def select_xgb_device(x_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> str:
    objective, num_class = resolve_objective(y_train)

    base_params = {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss" if objective == "binary:logistic" else "mlogloss",
        "objective": objective,
        "random_state": random_state,
        "verbosity": 0,
    }
    if num_class is not None:
        base_params["num_class"] = num_class

    try:
        gpu_model = XGBClassifier(
            **base_params,
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            device="cuda",
        )
        sample_x = x_train.head(min(512, len(x_train)))
        sample_y = y_train.head(min(512, len(y_train)))
        gpu_model.fit(sample_x, sample_y)
        return "gpu"
    except Exception:
        return "cpu"


def build_pipeline(y_train: pd.Series, device_mode: str, random_state: int) -> Pipeline:
    objective, num_class = resolve_objective(y_train)

    model_params = {
        "objective": objective,
        "random_state": random_state,
        "n_jobs": 1,
        "verbosity": 0,
        "eval_metric": "logloss" if objective == "binary:logistic" else "mlogloss",
    }
    if num_class is not None:
        model_params["num_class"] = num_class

    if device_mode == "gpu":
        model_params.update(
            {
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
                "device": "cuda",
            }
        )
    else:
        model_params.update(
            {
                "tree_method": "hist",
                "predictor": "auto",
            }
        )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(**model_params)),
        ]
    )


def run_training(x_train, y_train, cv: int, n_jobs: int, random_state: int):
    device_mode = select_xgb_device(x_train=x_train, y_train=y_train, random_state=random_state)
    pipeline = build_pipeline(y_train=y_train, device_mode=device_mode, random_state=random_state)

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__min_child_weight": [1, 3],
        "model__gamma": [0.0, 0.2],
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
    return grid_search, device_mode


def save_results(
    output_path: Path,
    target: str,
    train_size: int,
    test_size: int,
    grid_search: GridSearchCV,
    device_mode: str,
    feature_names,
    y_test,
    y_pred,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xgb_model = grid_search.best_estimator_.named_steps["model"]
    importance_series = pd.Series(xgb_model.feature_importances_, index=feature_names)
    importance_series = importance_series.sort_values(ascending=False)
    top_importances = importance_series.head(20)

    report_lines = [
        "=" * 70,
        "XGBOOST - RESULTADOS",
        "=" * 70,
        f"Data/hora: {datetime.now().isoformat(timespec='seconds')}",
        f"Coluna alvo: {target}",
        f"Modo de execucao: {device_mode.upper()}",
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

    grid_search, device_mode = run_training(
        x_train=x_train,
        y_train=y_train,
        cv=args.cv,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)

    save_results(
        output_path=args.output,
        target=args.target,
        train_size=len(x_train),
        test_size=len(x_test),
        grid_search=grid_search,
        device_mode=device_mode,
        feature_names=x_train.columns,
        y_test=y_test,
        y_pred=y_pred,
    )

    print(f"Treinamento concluido em modo {device_mode.upper()}. Resultados: {args.output}")


if __name__ == "__main__":
    main()
