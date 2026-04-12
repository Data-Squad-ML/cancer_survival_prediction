#!/usr/bin/env python3

"""Treina XGBoost com logica alinhada ao notebook, com GPU NVIDIA automatica e fallback para CPU."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


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
        description="Treino de XGBoost com Grid Search, KFold e ROC-AUC, usando GPU automaticamente."
    )
    parser.add_argument("--train-path", type=Path, default=default_data_path("data_train.csv"))
    parser.add_argument("--test-path", type=Path, default=default_data_path("data_test.csv"))
    parser.add_argument("--target", type=str, default="status_vital")
    parser.add_argument("--output", type=Path, default=Path("xgboost_results.txt"))
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--perm-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def get_scoring_and_auc_mode(y_train: pd.Series) -> tuple[str, str, str, int | None]:
    n_classes = int(y_train.nunique())
    if n_classes <= 2:
        return "roc_auc", "binary", "binary:logistic", None
    return "roc_auc_ovr_weighted", "multiclass", "multi:softprob", n_classes


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


def make_xgb_model(objective: str, num_class: int | None, random_state: int, mode: str) -> XGBClassifier:
    params = {
        "objective": objective,
        "eval_metric": "logloss" if objective == "binary:logistic" else "mlogloss",
        "random_state": random_state,
        "verbosity": 0,
    }
    if num_class is not None:
        params["num_class"] = num_class

    if mode == "gpu_hist":
        params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
    elif mode == "cuda_hist":
        params.update({"tree_method": "hist", "device": "cuda", "predictor": "auto"})
    else:
        params.update({"tree_method": "hist", "predictor": "auto"})

    return XGBClassifier(**params)


def select_xgb_mode(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series | None,
    random_state: int,
    force_cpu: bool,
) -> str:
    if force_cpu:
        return "cpu"

    _, _, objective, num_class = get_scoring_and_auc_mode(y_train)
    sample_size = min(1024, len(x_train))
    x_sample = x_train.iloc[:sample_size]
    y_sample = y_train.iloc[:sample_size]
    w_sample = w_train.iloc[:sample_size] if w_train is not None else None

    for mode in ["cuda_hist", "gpu_hist"]:
        try:
            model = make_xgb_model(objective, num_class, random_state, mode)
            fit_kwargs = {"sample_weight": w_sample} if w_sample is not None else {}
            model.fit(x_sample, y_sample, **fit_kwargs)
            return mode
        except Exception:
            continue

    return "cpu"


def run_training(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series | None,
    cv: int,
    n_jobs: int,
    random_state: int,
    force_cpu: bool,
):
    scoring, _, objective, num_class = get_scoring_and_auc_mode(y_train)
    xgb_mode = select_xgb_mode(x_train, y_train, w_train, random_state, force_cpu)

    model = make_xgb_model(objective, num_class, random_state, xgb_mode)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__min_child_weight": [1, 3],
        "model__gamma": [0.0, 0.2],
    }

    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_strategy,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )

    fit_params = {"model__sample_weight": w_train} if w_train is not None else {}
    grid_search.fit(x_train, y_train, **fit_params)

    return grid_search, scoring, xgb_mode


def compute_auc(y_true, y_proba, auc_mode: str) -> float:
    if auc_mode == "binary":
        return roc_auc_score(y_true, y_proba[:, 1])
    return roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")


def save_results(
    output_path: Path,
    target: str,
    train_size: int,
    test_size: int,
    feature_cols: list[str],
    w_train_used: bool,
    scoring: str,
    xgb_mode: str,
    train_auc: float,
    test_auc: float,
    grid_search: GridSearchCV,
    perm_df: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_idx = grid_search.best_index_
    best_std = float(grid_search.cv_results_["std_test_score"][best_idx])

    report_lines = [
        "=" * 70,
        "XGBOOST - RESULTADOS",
        "=" * 70,
        f"Data/hora: {datetime.now().isoformat(timespec='seconds')}",
        f"Coluna alvo: {target}",
        f"Modo de execucao: {xgb_mode}",
        f"Amostras de treino: {train_size}",
        f"Amostras de teste: {test_size}",
        f"Total de features selecionadas: {len(feature_cols)}",
        f"Uso de class_weight como sample_weight: {'sim' if w_train_used else 'nao'}",
        f"Metrica de Grid/CV: {scoring}",
        "",
        "--- Grid Search com KFold ---",
        f"Melhor score de CV: {grid_search.best_score_:.6f}",
        f"Desvio padrao do melhor score: {best_std:.6f}",
        f"Melhores hiperparametros: {grid_search.best_params_}",
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

    grid_search, scoring, xgb_mode = run_training(
        x_train=x_train,
        y_train=y_train,
        w_train=w_train,
        cv=args.cv,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        force_cpu=args.force_cpu,
    )

    _, auc_mode, _, _ = get_scoring_and_auc_mode(y_train)
    best_model = grid_search.best_estimator_

    train_proba = best_model.predict_proba(x_train)
    test_proba = best_model.predict_proba(x_test)
    train_auc = compute_auc(y_train, train_proba, auc_mode)
    test_auc = compute_auc(y_test, test_proba, auc_mode)

    perm = permutation_importance(
        best_model,
        x_test,
        y_test,
        scoring=scoring,
        n_repeats=args.perm_repeats,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    perm_df = (
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

    save_results(
        output_path=args.output,
        target=args.target,
        train_size=len(x_train),
        test_size=len(x_test),
        feature_cols=feature_cols,
        w_train_used=w_train is not None,
        scoring=scoring,
        xgb_mode=xgb_mode,
        train_auc=train_auc,
        test_auc=test_auc,
        grid_search=grid_search,
        perm_df=perm_df,
    )

    print(f"Treinamento concluido em modo {xgb_mode}. Resultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
