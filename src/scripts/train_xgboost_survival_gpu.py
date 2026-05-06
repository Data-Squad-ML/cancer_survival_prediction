#!/usr/bin/env python3

"""Treina XGBoost Survival (Cox) com Grid Search, KFold, GPU automatica e metricas avancadas."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv
from xgboost import XGBRegressor


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
            "Treino de XGBoost Survival (objective=survival:cox) com Grid Search, "
            "KFold, C-index, AUC dinamica, Brier score, IBS e importancias."
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
        default=Path("results/xgboost_survival_gpu_results.txt"),
    )
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--xgb-n-jobs", type=int, default=1)
    parser.add_argument("--perm-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--min-events-per-fold", type=int, default=5)
    parser.add_argument("--min-eval-times", type=int, default=8)
    parser.add_argument("--max-eval-times", type=int, default=25)
    parser.add_argument(
        "--weighting",
        type=str,
        choices=["ipcw", "class_weight", "none"],
        default="ipcw",
        help=(
            "Peso de amostras. 'ipcw' usa pesos por censura (recomendado para survival), "
            "'class_weight' usa a coluna class_weight, 'none' desativa pesos."
        ),
    )
    parser.add_argument(
        "--ipcw-min-prob",
        type=float,
        default=1e-3,
        help="Limite minimo para G_hat(t) no IPCW para evitar pesos extremos.",
    )
    parser.add_argument(
        "--shap-csv",
        type=Path,
        default=Path("results/xgboost_survival_shap_global.csv"),
    )
    parser.add_argument(
        "--shap-plot",
        type=Path,
        default=Path("results/xgboost_survival_shap_summary.png"),
    )
    parser.add_argument(
        "--shap-beeswarm-plot",
        type=Path,
        default=Path("results/xgboost_survival_shap_beeswarm.png"),
    )
    parser.add_argument("--shap-max-samples", type=int, default=5000)
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


def build_survival_targets(
    df: pd.DataFrame,
    target: str,
    time_col: str,
    event_time_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event = (df[target].astype(int) == 0).to_numpy(dtype=bool)
    time = np.where(event, df[event_time_col].to_numpy(), df[time_col].to_numpy()).astype(float)
    y_surv = Surv.from_arrays(event=event, time=time)
    return y_surv, event, time


def build_xgb_label(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    label = time.copy()
    label[~event] *= -1.0
    return label


def compute_ipcw_weights(y_surv, min_prob: float = 1e-3) -> np.ndarray:
    """Calcula pesos IPCW com base na distribuicao de censura do treino."""
    event = np.asarray(y_surv["event"]).astype(bool)
    time = np.asarray(y_surv["time"]).astype(float)
    censor_event = ~event

    _, g_hat = kaplan_meier_estimator(censor_event, time)
    g_hat = np.interp(time, _, g_hat, left=g_hat[0], right=g_hat[-1])
    g_hat = np.clip(g_hat, min_prob, 1.0)
    return (1.0 / g_hat).astype(np.float32)


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
    x_train = train_df[feature_cols].astype(np.float32)
    x_test = test_df[feature_cols].astype(np.float32)

    y_train_surv, event_train, time_train = build_survival_targets(
        train_df, target, time_col, event_time_col
    )
    y_test_surv, event_test, time_test = build_survival_targets(
        test_df, target, time_col, event_time_col
    )

    y_train_xgb = build_xgb_label(time_train, event_train)
    y_test_xgb = build_xgb_label(time_test, event_test)

    w_class = train_df["class_weight"].astype(float) if "class_weight" in train_df.columns else None

    return (
        x_train,
        x_test,
        y_train_surv,
        y_test_surv,
        y_train_xgb,
        y_test_xgb,
        w_class,
        feature_cols,
    )


def make_xgb_model(random_state: int, xgb_n_jobs: int, mode: str) -> XGBRegressor:
    params = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "random_state": random_state,
        "verbosity": 0,
        "n_jobs": xgb_n_jobs,
        "max_delta_step": 1,
    }

    if mode == "gpu_hist":
        params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})
    elif mode == "cuda_hist":
        params.update({"tree_method": "hist", "device": "cuda", "predictor": "auto"})
    else:
        params.update({"tree_method": "hist", "predictor": "auto"})

    return XGBRegressor(**params)


class XGBSurvivalCoxEstimator(BaseEstimator, RegressorMixin):
    """Wrapper para permitir y estruturado de survival no GridSearchCV.

    O XGBoost survival:cox espera rótulo numérico com tempo positivo para evento
    e tempo negativo para censura. Este wrapper faz essa conversão internamente,
    preservando a API do scikit-learn para CV/scoring com y estruturado.
    """

    def __init__(
        self,
        random_state: int = 42,
        xgb_n_jobs: int = 1,
        mode: str = "cpu",
        n_estimators: int = 300,
        max_depth: int = 3,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        colsample_bynode: float = 1.0,
        min_child_weight: float = 1.0,
        gamma: float = 0.0,
        max_bin: int | None = None,
        grow_policy: str | None = None,
        max_leaves: int = 0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int | None = 50,
        eval_fraction: float = 0.1,
        eval_metric: str | None = None,
    ):
        self.random_state = random_state
        self.xgb_n_jobs = xgb_n_jobs
        self.mode = mode
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bynode = colsample_bynode
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.max_bin = max_bin
        self.grow_policy = grow_policy
        self.max_leaves = max_leaves
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_fraction = eval_fraction
        self.eval_metric = eval_metric

    def _make_model(self) -> XGBRegressor:
        model = make_xgb_model(
            random_state=self.random_state,
            xgb_n_jobs=self.xgb_n_jobs,
            mode=self.mode,
        )
        model.set_params(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            colsample_bynode=self.colsample_bynode,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            max_bin=self.max_bin,
            grow_policy=self.grow_policy,
            max_leaves=self.max_leaves,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
        )
        return model

    def fit(self, X, y, sample_weight=None):
        event = np.asarray(y["event"]).astype(bool)
        time = np.asarray(y["time"]).astype(float)
        y_xgb = build_xgb_label(time=time, event=event).astype(np.float32)
        X = np.asarray(X, dtype=np.float32)
        self.model_ = self._make_model()
        self.y_train_surv_ = y

        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        use_early_stopping = (
            self.early_stopping_rounds is not None
            and self.early_stopping_rounds > 0
            and self.eval_fraction is not None
            and 0.0 < self.eval_fraction < 0.5
        )

        if use_early_stopping and len(X) >= 10:
            stratify = event if event.sum() >= 2 else None
            X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
                X,
                y_xgb,
                sample_weight,
                test_size=self.eval_fraction,
                random_state=self.random_state,
                stratify=stratify,
            )
            es_kwargs = {
                "eval_set": [(X_val, y_val)],
                "early_stopping_rounds": self.early_stopping_rounds,
                "verbose": False,
            }
            if self.eval_metric:
                es_kwargs["eval_metric"] = self.eval_metric

            fit_kwargs_train = {}
            if w_tr is not None:
                fit_kwargs_train["sample_weight"] = w_tr
                es_kwargs["sample_weight_eval_set"] = [w_val]

            self.model_.fit(X_tr, y_tr, **fit_kwargs_train, **es_kwargs)
        else:
            self.model_.fit(X, y_xgb, **fit_kwargs)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model_.feature_importances_


def select_xgb_mode(
    x_train: pd.DataFrame,
    y_train_xgb: np.ndarray,
    w_train: pd.Series | None,
    random_state: int,
    xgb_n_jobs: int,
    force_cpu: bool,
) -> str:
    if force_cpu:
        return "cpu"

    sample_size = min(2048, len(x_train))
    x_sample = x_train.iloc[:sample_size]
    y_sample = y_train_xgb[:sample_size].astype(np.float32)
    if w_train is None:
        w_sample = None
    elif hasattr(w_train, "iloc"):
        w_sample = w_train.iloc[:sample_size]
    else:
        w_sample = np.asarray(w_train)[:sample_size]

    for mode in ["cuda_hist", "gpu_hist"]:
        try:
            model = make_xgb_model(random_state=random_state, xgb_n_jobs=xgb_n_jobs, mode=mode)
            fit_kwargs = {"sample_weight": w_sample} if w_sample is not None else {}
            model.fit(x_sample, y_sample, **fit_kwargs)
            return mode
        except Exception:
            continue

    return "cpu"


def rsf_like_cindex_scorer(estimator, x, y_surv) -> float:
    risk = estimator.predict(x)
    risk = np.clip(risk, -1e6, 1e6)

    return float(concordance_index_censored(y_surv["event"], y_surv["time"], risk)[0])


def uno_cindex_scorer(estimator, x, y_surv) -> float:
    risk = estimator.predict(x)
    risk = np.clip(risk, -1e6, 1e6)
    y_train = getattr(estimator, "y_train_surv_", y_surv)
    return float(concordance_index_ipcw(y_train, y_surv, risk)[0])


def run_training(
    x_train: pd.DataFrame,
    event_train: np.ndarray,
    y_train_surv,
    w_train: pd.Series | None,
    cv: int,
    n_jobs: int,
    xgb_n_jobs: int,
    random_state: int,
    verbose: int,
    force_cpu: bool,
    min_events_per_fold: int,
):
    xgb_mode = select_xgb_mode(
        x_train=x_train,
        y_train_xgb=build_xgb_label(
            time=np.asarray(y_train_surv["time"]).astype(float),
            event=np.asarray(y_train_surv["event"]).astype(bool),
        ),
        w_train=w_train,
        random_state=random_state,
        xgb_n_jobs=xgb_n_jobs,
        force_cpu=force_cpu,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBSurvivalCoxEstimator(
                    random_state=random_state,
                    xgb_n_jobs=xgb_n_jobs,
                    mode=xgb_mode,
                    early_stopping_rounds=75,
                    eval_fraction=0.15,
                    eval_metric="cox-nloglik",
                ),
            ),
        ]
    )

    param_grid = {
        "model__n_estimators": [600, 800, 1000],
        "model__max_depth": [3, 4, 5],
        "model__learning_rate": [0.01, 0.02, 0.03],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 5],
        "model__gamma": [0.0, 0.5],
        "model__max_bin": [256],
        "model__grow_policy": ["depthwise", "lossguide"],
        "model__max_leaves": [0, 64],
        "model__colsample_bynode": [1.0],
        "model__reg_alpha": [0.0, 0.5],
        "model__reg_lambda": [1.0, 3.0],
    }

    # Survival com evento raro pode gerar folds sem eventos com KFold comum.
    # Isso distorce C-index/seleciona hiperparametros instaveis.
    # Por isso, estratificamos pelo indicador de evento e validamos um minimo
    # de eventos por fold de validacao.
    event_train = np.asarray(event_train).astype(int)
    n_events = int(event_train.sum())
    n_samples = int(len(event_train))

    if n_events < cv:
        raise ValueError(
            "Numero de eventos insuficiente para StratifiedKFold: "
            f"eventos={n_events}, cv={cv}. Reduza cv ou aumente dados com evento."
        )

    if n_events < cv * min_events_per_fold:
        raise ValueError(
            "Distribuicao de eventos insuficiente para o minimo por fold: "
            f"eventos={n_events}, cv={cv}, min_events_per_fold={min_events_per_fold}. "
            "Reduza cv/min_events_per_fold ou reavalie o split treino/teste."
        )

    if (n_samples - n_events) < cv:
        raise ValueError(
            "Numero de censurados insuficiente para StratifiedKFold: "
            f"censurados={n_samples - n_events}, cv={cv}."
        )

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_splits = list(cv_strategy.split(x_train, event_train))

    min_events_found = min(int(event_train[val_idx].sum()) for _, val_idx in cv_splits)
    if min_events_found < min_events_per_fold:
        raise ValueError(
            "Fold de validacao com poucos eventos apos estratificacao: "
            f"min_encontrado={min_events_found}, minimo_exigido={min_events_per_fold}. "
            "Ajuste cv/min_events_per_fold."
        )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=uno_cindex_scorer,
        cv=cv_splits,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
    )

    fit_params = {"model__sample_weight": w_train} if w_train is not None else {}
    grid_search.fit(x_train, y_train_surv, **fit_params)
    return grid_search, xgb_mode


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
        raise ValueError("Nao foi possivel determinar intervalo de tempos para metricas dinamicas.")

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


def compute_survival_metrics(
    model,
    x_train: pd.DataFrame,
    y_train_surv,
    x_test: pd.DataFrame,
    y_test_surv,
    eval_times: np.ndarray,
) -> dict:
    risk_train = model.predict(x_train)
    risk_test = model.predict(x_test)

    cindex_train = float(
        concordance_index_censored(y_train_surv["event"], y_train_surv["time"], risk_train)[0]
    )
    cindex_test = float(
        concordance_index_censored(y_test_surv["event"], y_test_surv["time"], risk_test)[0]
    )
    uno_cindex_test = float(concordance_index_ipcw(y_train_surv, y_test_surv, risk_test)[0])

    auc_by_time, mean_auc = cumulative_dynamic_auc(
        y_train_surv,
        y_test_surv,
        risk_test,
        eval_times,
    )

    breslow = BreslowEstimator().fit(
        y_train_surv["event"],
        y_train_surv["time"],
        risk_train,
    )
    survival_functions_test = breslow.get_survival_function(risk_test)
    survival_prob_test = np.vstack([fn(eval_times) for fn in survival_functions_test])

    _, brier_scores = brier_score(y_train_surv, y_test_surv, survival_prob_test, eval_times)
    mean_brier = float(np.mean(brier_scores))
    ibs = float(integrated_brier_score(y_train_surv, y_test_surv, survival_prob_test, eval_times))

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


def compute_feature_importance_tables(
    best_model,
    x_test: pd.DataFrame,
    y_test_surv,
    n_repeats: int,
    random_state: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    xgb_model = best_model.named_steps["model"]
    xgb_gain_df = (
        pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance_gain": xgb_model.feature_importances_,
            }
        )
        .sort_values("importance_gain", ascending=False)
        .reset_index(drop=True)
    )

    perm = permutation_importance(
        estimator=best_model,
        X=x_test,
        y=y_test_surv,
        scoring=rsf_like_cindex_scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
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

    return xgb_gain_df, perm_df


def compute_and_save_global_shap(
    best_model,
    x_test: pd.DataFrame,
    shap_csv_path: Path,
    shap_plot_path: Path,
    shap_beeswarm_plot_path: Path,
    shap_max_samples: int,
) -> pd.DataFrame:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        raise ImportError(
            "Dependencias para SHAP ausentes. Instale com: pip install shap matplotlib"
        ) from exc

    shap_csv_path.parent.mkdir(parents=True, exist_ok=True)
    shap_plot_path.parent.mkdir(parents=True, exist_ok=True)
    shap_beeswarm_plot_path.parent.mkdir(parents=True, exist_ok=True)

    imputer = best_model.named_steps["imputer"]
    wrapped_model = best_model.named_steps["model"]
    xgb_model = wrapped_model.model_

    # SHAP no XGBoost deve usar a mesma representacao de entrada usada no treino.
    sample_size = min(len(x_test), int(shap_max_samples))
    x_sample = x_test.iloc[:sample_size].copy()
    x_imputed = imputer.transform(x_sample)
    x_imputed_df = pd.DataFrame(x_imputed, columns=x_test.columns, index=x_sample.index)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(x_imputed_df)

    if isinstance(shap_values, list):
        # Para manter compatibilidade com diferentes versoes do SHAP.
        shap_values = shap_values[0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_global_df = (
        pd.DataFrame(
            {
                "feature": x_test.columns,
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_global_df.to_csv(shap_csv_path, index=False)

    shap.summary_plot(
        shap_values,
        features=x_imputed_df,
        feature_names=list(x_test.columns),
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(shap_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        shap_values,
        features=x_imputed_df,
        feature_names=list(x_test.columns),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(shap_beeswarm_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    return shap_global_df


def save_results(
    output_path: Path,
    target: str,
    train_size: int,
    test_size: int,
    feature_cols: list[str],
    w_train_used: bool,
    weighting_label: str,
    cv: int,
    min_events_per_fold: int,
    observed_min_events_per_fold: int,
    n_jobs: int,
    xgb_n_jobs: int,
    xgb_mode: str,
    grid_search: GridSearchCV,
    metrics: dict,
    xgb_gain_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    shap_global_df: pd.DataFrame,
    shap_csv_path: Path,
    shap_plot_path: Path,
    shap_beeswarm_plot_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_idx = grid_search.best_index_
    best_std = float(grid_search.cv_results_["std_test_score"][best_idx])
    eval_times = metrics["eval_times"]
    auc_by_time = metrics["dynamic_auc_by_time"]

    auc_time_df = pd.DataFrame(
        {
            "tempo": np.round(eval_times, 4),
            "auc_dinamica": np.round(auc_by_time, 6),
        }
    )

    report_lines = [
        "=" * 70,
        "XGBOOST SURVIVAL (GPU/CPU) - RESULTADOS",
        "=" * 70,
        f"Data/hora: {datetime.now().isoformat(timespec='seconds')}",
        f"Coluna alvo: {target}",
        f"Modo de execucao: {xgb_mode}",
        f"Amostras de treino: {train_size}",
        f"Amostras de teste: {test_size}",
        f"Total de features selecionadas: {len(feature_cols)}",
        f"Peso de amostras (treino): {weighting_label}",
        f"Uso de pesos nas amostras: {'sim' if w_train_used else 'nao'}",
        f"KFold (cv): {cv}",
        "Estrategia de CV: StratifiedKFold por evento",
        f"Minimo exigido de eventos por fold (validacao): {min_events_per_fold}",
        f"Minimo observado de eventos por fold (validacao): {observed_min_events_per_fold}",
        f"n_jobs (GridSearch/permutation): {n_jobs}",
        f"xgb_n_jobs (modelo): {xgb_n_jobs}",
        "Metrica de Grid/CV: Uno C-index (IPCW)",
        "",
        "--- Grid Search com KFold ---",
        f"Melhor score de CV (C-index): {grid_search.best_score_:.6f}",
        f"Desvio padrao do melhor score: {best_std:.6f}",
        f"Melhores hiperparametros: {grid_search.best_params_}",
        "",
        "--- Metricas de Qualidade (treino/teste) ---",
        f"C-index treino       : {metrics['cindex_train']:.6f}",
        f"C-index teste        : {metrics['cindex_test']:.6f}",
        f"Gap C-index          : {metrics['cindex_gap']:.6f}",
        f"Uno C-index teste    : {metrics['uno_cindex_test']:.6f}",
        f"AUC dinamica media   : {metrics['mean_dynamic_auc']:.6f}",
        f"Brier medio          : {metrics['mean_brier']:.6f}",
        f"Integrated Brier (IBS): {metrics['ibs']:.6f}",
        "",
        "--- AUC Dinamica por Tempo ---",
        auc_time_df.to_string(index=False),
        "",
        "--- Top 20 Feature Importance (gain do XGBoost) ---",
        xgb_gain_df.head(20).to_string(index=False),
        "",
        "--- Top 20 Permutation Importance (C-index no teste) ---",
        perm_df.head(20).to_string(index=False),
        "",
        "--- Top 20 SHAP Global (mean(|SHAP|)) ---",
        shap_global_df.head(20).to_string(index=False),
        "",
        f"SHAP global CSV: {shap_csv_path}",
        f"SHAP summary plot: {shap_plot_path}",
        f"SHAP beeswarm plot: {shap_beeswarm_plot_path}",
        "",
    ]

    output_path.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    (
        x_train,
        x_test,
        y_train_surv,
        y_test_surv,
        _,
        _,
        w_class,
        feature_cols,
    ) = load_split_data(
        train_path=args.train_path,
        test_path=args.test_path,
        target=args.target,
        time_col=args.time_col,
        event_time_col=args.event_time_col,
    )

    w_train = None
    weighting_label = "none"
    if args.weighting == "class_weight":
        w_train = w_class
        weighting_label = "class_weight"
    elif args.weighting == "ipcw":
        # IPCW corrige vies de censura e costuma melhorar ranking (C-index).
        w_train = compute_ipcw_weights(y_train_surv, min_prob=args.ipcw_min_prob)
        weighting_label = f"ipcw(min_prob={args.ipcw_min_prob})"

    grid_search, xgb_mode = run_training(
        x_train=x_train,
        event_train=y_train_surv["event"],
        y_train_surv=y_train_surv,
        w_train=w_train,
        cv=args.cv,
        n_jobs=args.n_jobs,
        xgb_n_jobs=args.xgb_n_jobs,
        random_state=args.random_state,
        verbose=args.verbose,
        force_cpu=args.force_cpu,
        min_events_per_fold=args.min_events_per_fold,
    )

    best_model = grid_search.best_estimator_

    eval_times = build_eval_times(
        y_train_surv=y_train_surv,
        y_test_surv=y_test_surv,
        min_eval_times=args.min_eval_times,
        max_eval_times=args.max_eval_times,
    )
    metrics = compute_survival_metrics(
        model=best_model,
        x_train=x_train,
        y_train_surv=y_train_surv,
        x_test=x_test,
        y_test_surv=y_test_surv,
        eval_times=eval_times,
    )

    xgb_gain_df, perm_df = compute_feature_importance_tables(
        best_model=best_model,
        x_test=x_test,
        y_test_surv=y_test_surv,
        n_repeats=args.perm_repeats,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    shap_global_df = compute_and_save_global_shap(
        best_model=best_model,
        x_test=x_test,
        shap_csv_path=args.shap_csv,
        shap_plot_path=args.shap_plot,
        shap_beeswarm_plot_path=args.shap_beeswarm_plot,
        shap_max_samples=args.shap_max_samples,
    )

    event_train = y_train_surv["event"].astype(int)
    cv_probe = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    observed_min_events_per_fold = min(
        int(event_train[val_idx].sum()) for _, val_idx in cv_probe.split(x_train, event_train)
    )

    save_results(
        output_path=args.output,
        target=args.target,
        train_size=len(x_train),
        test_size=len(x_test),
        feature_cols=feature_cols,
        w_train_used=w_train is not None,
        weighting_label=weighting_label,
        cv=args.cv,
        min_events_per_fold=args.min_events_per_fold,
        observed_min_events_per_fold=observed_min_events_per_fold,
        n_jobs=args.n_jobs,
        xgb_n_jobs=args.xgb_n_jobs,
        xgb_mode=xgb_mode,
        grid_search=grid_search,
        metrics=metrics,
        xgb_gain_df=xgb_gain_df,
        perm_df=perm_df,
        shap_global_df=shap_global_df,
        shap_csv_path=args.shap_csv,
        shap_plot_path=args.shap_plot,
        shap_beeswarm_plot_path=args.shap_beeswarm_plot,
    )

    print(f"Treinamento XGBoost Survival concluido. Resultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
