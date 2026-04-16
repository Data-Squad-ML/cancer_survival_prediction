#!/usr/bin/env python3

"""Treina Random Survival Forest (sksurv), com Grid Search, KFold e permutation importance.

Observacao importante:
- O algoritmo RandomSurvivalForest do scikit-survival nao possui backend de treino em GPU.
- A execucao e paralelizada em CPU via n_jobs (modelo, GridSearchCV e permutation importance).
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

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
        description=(
            "Treino de Random Survival Forest com Grid Search manual via GridSearchCV, "
            "KFold, C-index e permutation importance."
        )
    )
    parser.add_argument("--train-path", type=Path, default=default_data_path("data_train.csv"))
    parser.add_argument("--test-path", type=Path, default=default_data_path("data_test.csv"))
    parser.add_argument("--target", type=str, default="status_vital")
    parser.add_argument("--time-col", type=str, default="tempo_total_doenca")
    parser.add_argument("--event-time-col", type=str, default="tempo_ate_obito")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/random_survival_forest_results.txt"),
    )
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--perm-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def build_feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    feature_cols_ohe = [
        col for col in train_df.columns if any(col.startswith(prefix) for prefix in OHE_PREFIXES)
    ]
    selected = [col for col in (FEATURE_COLS_BASE + feature_cols_ohe) if col in train_df.columns]
    selected = [col for col in selected if col in test_df.columns]

    if not selected:
        raise ValueError("Nenhuma feature selecionada foi encontrada nos dados de treino/teste.")

    return selected


def build_survival_array(df: pd.DataFrame, target: str, time_col: str, event_time_col: str):
    event = (df[target].astype(int) == 0).to_numpy(dtype=bool)
    time = np.where(event, df[event_time_col].to_numpy(), df[time_col].to_numpy()).astype(float)
    return Surv.from_arrays(event=event, time=time)


def load_split_data(
    train_path: Path,
    test_path: Path,
    target: str,
    time_col: str,
    event_time_col: str,
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    required_cols = {target, time_col, event_time_col}
    for col in required_cols:
        if col not in train_df.columns or col not in test_df.columns:
            raise ValueError(f"A coluna obrigatoria '{col}' nao existe em treino e teste.")

    train_df = train_df.dropna(subset=[target, time_col]).copy()
    test_df = test_df.dropna(subset=[target, time_col]).copy()

    feature_cols = build_feature_columns(train_df, test_df)

    x_train = train_df[feature_cols].astype(float)
    x_test = test_df[feature_cols].astype(float)

    y_train = build_survival_array(train_df, target, time_col, event_time_col)
    y_test = build_survival_array(test_df, target, time_col, event_time_col)

    return x_train, y_train, x_test, y_test, feature_cols


def rsf_cindex_scorer(estimator, x, y_surv) -> float:
    risk_scores = estimator.predict(x)
    return float(concordance_index_censored(y_surv["event"], y_surv["time"], risk_scores)[0])


def build_pipeline(random_state: int, n_jobs: int) -> Pipeline:
    model = RandomSurvivalForest(
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def build_param_grid() -> dict:
    return {
        "model__n_estimators": [100, 300, 500],
        "model__max_depth": [8, 12, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 3, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.5],
    }


def run_training(
    x_train: pd.DataFrame,
    y_train,
    cv: int,
    n_jobs: int,
    random_state: int,
    verbose: int,
):
    pipeline = build_pipeline(random_state=random_state, n_jobs=n_jobs)
    param_grid = build_param_grid()
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=rsf_cindex_scorer,
        cv=cv_strategy,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
    )

    grid_search.fit(x_train, y_train)
    return grid_search


def evaluate_cindex(model, x, y_surv) -> float:
    risk = model.predict(x)
    return float(concordance_index_censored(y_surv["event"], y_surv["time"], risk)[0])


def compute_permutation_importance(
    model,
    x_test: pd.DataFrame,
    y_test,
    n_repeats: int,
    random_state: int,
    n_jobs: int,
) -> pd.DataFrame:
    perm = permutation_importance(
        estimator=model,
        X=x_test,
        y=y_test,
        scoring=rsf_cindex_scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return (
        pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def save_results(
    output_path: Path,
    train_size: int,
    test_size: int,
    feature_cols: list[str],
    cv: int,
    n_jobs: int,
    best_cindex_cv: float,
    best_std_cv: float,
    best_params: dict,
    train_cindex: float,
    test_cindex: float,
    perm_df: pd.DataFrame,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "=" * 70,
        "RANDOM SURVIVAL FOREST - RESULTADOS",
        "=" * 70,
        f"Data/hora: {datetime.now().isoformat(timespec='seconds')}",
        "Modo de execucao: CPU paralela (RSF do scikit-survival nao suporta GPU)",
        f"Amostras de treino: {train_size}",
        f"Amostras de teste: {test_size}",
        f"Total de features selecionadas: {len(feature_cols)}",
        f"KFold (cv): {cv}",
        f"n_jobs: {n_jobs}",
        "Metrica de Grid/CV: C-index (concordance_index_censored)",
        "",
        "--- Grid Search com KFold ---",
        f"Melhor score de CV (C-index): {best_cindex_cv:.6f}",
        f"Desvio padrao do melhor score: {best_std_cv:.6f}",
        f"Melhores hiperparametros: {best_params}",
        "",
        "--- Avaliacao C-index ---",
        f"C-index treino: {train_cindex:.6f}",
        f"C-index teste : {test_cindex:.6f}",
        f"Gap           : {train_cindex - test_cindex:.6f}",
        "",
        "--- Top 20 Permutation Importance (teste) ---",
        perm_df.head(20).to_string(index=False),
        "",
    ]

    output_path.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    x_train, y_train, x_test, y_test, feature_cols = load_split_data(
        train_path=args.train_path,
        test_path=args.test_path,
        target=args.target,
        time_col=args.time_col,
        event_time_col=args.event_time_col,
    )

    grid_search = run_training(
        x_train=x_train,
        y_train=y_train,
        cv=args.cv,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbose=args.verbose,
    )

    best_model = grid_search.best_estimator_
    best_idx = grid_search.best_index_
    best_std = float(grid_search.cv_results_["std_test_score"][best_idx])

    train_cindex = evaluate_cindex(best_model, x_train, y_train)
    test_cindex = evaluate_cindex(best_model, x_test, y_test)

    perm_df = compute_permutation_importance(
        model=best_model,
        x_test=x_test,
        y_test=y_test,
        n_repeats=args.perm_repeats,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    save_results(
        output_path=args.output,
        train_size=len(x_train),
        test_size=len(x_test),
        feature_cols=feature_cols,
        cv=args.cv,
        n_jobs=args.n_jobs,
        best_cindex_cv=float(grid_search.best_score_),
        best_std_cv=best_std,
        best_params=grid_search.best_params_,
        train_cindex=train_cindex,
        test_cindex=test_cindex,
        perm_df=perm_df,
    )

    print(f"Treinamento RSF concluido. Resultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
