#!/usr/bin/env python3

"""Treina XGBoost Survival (Cox) com Grid Search, KFold, GPU automatica e metricas avancadas.

Metricas adicionais incluidas (alem das originais):
  - Curvas de Kaplan-Meier por tercis de risco + log-rank test
  - KS de sobrevivência entre grupos de alto e baixo risco
  - Calibracao por decil (E/O ratio — analogo ao Lift/Ganho de classificacao)
  - D de Royston (poder discriminativo em escala de log-risco)
  - Distribuicao dos log-riscos por status de evento
  - Net Benefit simplificado (Decision Curve Analysis)
  - C-index por subgrupo (sexo, faixa etaria se disponiveis)
  - Brier score por tempo exportado como CSV
  - CSVs de curvas de sobrevivencia e calibracao para plotagem externa
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
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
    "historico_tabagismo_clinico_",
    "historico_alcoolismo_clinico_",
]


# ---------------------------------------------------------------------------
# Utilitarios gerais
# ---------------------------------------------------------------------------

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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Diretorio para CSVs auxiliares de curvas e calibracao.",
    )
    parser.add_argument("--cv", type=int, default=2)
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
    )
    parser.add_argument("--ipcw-min-prob", type=float, default=1e-3)
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
    # Coluna de subgrupo opcional para C-index estratificado
    parser.add_argument(
        "--subgroup-col",
        type=str,
        default="sexo",
        help="Coluna categorica para calculo de C-index por subgrupo.",
    )
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
    time = np.clip(time, 1e-6, None)
    y_surv = Surv.from_arrays(event=event, time=time)
    return y_surv, event, time


def build_xgb_label(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    label = time.copy()
    label[~event] *= -1.0
    return label


def compute_ipcw_weights(y_surv, min_prob: float = 1e-3) -> np.ndarray:
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
        train_df,
        test_df,
        event_train,
        time_train,
        event_test,
        time_test,
    )


# ---------------------------------------------------------------------------
# Modelo e treinamento
# ---------------------------------------------------------------------------

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
    """Wrapper para permitir y estruturado de survival no GridSearchCV."""

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

            try:
                self.model_.fit(X_tr, y_tr, **fit_kwargs_train, **es_kwargs)
            except TypeError as exc:
                msg = str(exc)
                if any(
                    key in msg
                    for key in ("early_stopping_rounds", "eval_set", "sample_weight_eval_set")
                ):
                    self.model_.fit(X, y_xgb, **fit_kwargs)
                else:
                    raise
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
    w_train,
    random_state: int,
    xgb_n_jobs: int,
    force_cpu: bool,
) -> str:
    if force_cpu:
        return "cpu"

    sample_size = min(2048, len(x_train))
    x_sample = x_train.iloc[:sample_size]
    y_sample = y_train_xgb[:sample_size].astype(np.float32)
    w_sample = None
    if w_train is not None:
        if hasattr(w_train, "iloc"):
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
    w_train,
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
        "model__n_estimators": [600, 800],
        "model__max_depth": [3, 4],
        "model__learning_rate": [0.01, 0.03],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__gamma": [0.0, 0.5],
        "model__max_leaves": [0, 64],
        "model__reg_lambda": [1.0, 3.0],
    }

    event_train = np.asarray(event_train).astype(int)
    n_events = int(event_train.sum())
    n_samples = int(len(event_train))

    if n_events < cv:
        raise ValueError(
            f"Numero de eventos insuficiente para StratifiedKFold: eventos={n_events}, cv={cv}."
        )
    if n_events < cv * min_events_per_fold:
        raise ValueError(
            f"Distribuicao de eventos insuficiente: eventos={n_events}, cv={cv}, "
            f"min_events_per_fold={min_events_per_fold}."
        )
    if (n_samples - n_events) < cv:
        raise ValueError(
            f"Numero de censurados insuficiente: censurados={n_samples - n_events}, cv={cv}."
        )

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_splits = list(cv_strategy.split(x_train, event_train))

    min_events_found = min(int(event_train[val_idx].sum()) for _, val_idx in cv_splits)
    if min_events_found < min_events_per_fold:
        raise ValueError(
            f"Fold de validacao com poucos eventos: min_encontrado={min_events_found}, "
            f"minimo_exigido={min_events_per_fold}."
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

    # Breslow para funcao de sobrevivencia — determina o dominio valido primeiro,
    # para garantir que AUC, Brier e survival_test usem EXATAMENTE os mesmos tempos.
    breslow = BreslowEstimator().fit(
        y_train_surv["event"],
        y_train_surv["time"],
        risk_train,
    )
    survival_functions_test = breslow.get_survival_function(risk_test)
    domain_min, domain_max = survival_functions_test[0].domain
    test_min = max(float(np.min(y_test_surv["time"])), 1e-6)
    test_max = float(np.max(y_test_surv["time"]))
    test_max = np.nextafter(test_max, float("-inf"))

    domain_min = max(float(domain_min), test_min)
    domain_max = min(float(domain_max), test_max)
    if domain_max <= domain_min:
        raise ValueError(
            "Nao foi possivel encontrar intervalo valido para metricas dinamicas dentro "
            "do follow-up do teste."
        )

    eps = max((domain_max - domain_min) * 1e-3, 1e-9)

    # clipped_times: subconjunto de eval_times dentro do dominio Breslow.
    # Todas as metricas (AUC, Brier, survival) usam este unico vetor.
    clipped_times = eval_times[(eval_times > domain_min + eps) & (eval_times < domain_max - eps)]
    if len(clipped_times) < 3:
        clipped_times = np.linspace(domain_min + eps, domain_max - eps, num=5)

    auc_by_time, mean_auc = cumulative_dynamic_auc(
        y_train_surv, y_test_surv, risk_test, clipped_times
    )

    # survival_prob_test: shape (n_test, n_clipped_times)
    survival_prob_test = np.vstack([fn(clipped_times) for fn in survival_functions_test])

    _, brier_scores = brier_score(y_train_surv, y_test_surv, survival_prob_test, clipped_times)
    mean_brier = float(np.mean(brier_scores))
    ibs = float(integrated_brier_score(y_train_surv, y_test_surv, survival_prob_test, clipped_times))

    return {
        "cindex_train": cindex_train,
        "cindex_test": cindex_test,
        "cindex_gap": cindex_train - cindex_test,
        "uno_cindex_test": uno_cindex_test,
        "mean_dynamic_auc": float(mean_auc),
        "dynamic_auc_by_time": auc_by_time,   # len == len(clipped_times)
        "mean_brier": mean_brier,
        "ibs": ibs,
        "eval_times": clipped_times,           # unica fonte de verdade para os tempos
        "survival_test": survival_prob_test,   # shape (n_test, len(clipped_times))
        "brier_by_time": brier_scores,         # len == len(clipped_times)
        "risk_train": risk_train,
        "risk_test": risk_test,
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

    sample_size = min(len(x_test), int(shap_max_samples))
    x_sample = x_test.iloc[:sample_size].copy()
    x_imputed = imputer.transform(x_sample)
    x_imputed_df = pd.DataFrame(x_imputed, columns=x_test.columns, index=x_sample.index)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(x_imputed_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_global_df = (
        pd.DataFrame({"feature": x_test.columns, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_global_df.to_csv(shap_csv_path, index=False)

    shap.summary_plot(
        shap_values, features=x_imputed_df,
        feature_names=list(x_test.columns), plot_type="bar", show=False,
    )
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(shap_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        shap_values, features=x_imputed_df,
        feature_names=list(x_test.columns), show=False,
    )
    plt.tight_layout()
    plt.savefig(shap_beeswarm_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    return shap_global_df


# ===========================================================================
# METRICAS ADICIONAIS (identicas as do DeepSurv, sem dependencia de PyTorch)
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Kaplan-Meier por grupos de risco + log-rank test
# ---------------------------------------------------------------------------

def logrank_test_two_groups(
    time_a: np.ndarray, event_a: np.ndarray,
    time_b: np.ndarray, event_b: np.ndarray,
) -> tuple[float, float]:
    """Log-rank test manual (Mantel-Cox) entre dois grupos."""
    all_times = np.unique(np.concatenate([time_a[event_a], time_b[event_b]]))
    O_a_total, E_a_total, V_total = 0.0, 0.0, 0.0

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
            V_total += d * n_a * n_b * (n - d) / (n**2 * (n - 1))

    if V_total < 1e-10:
        return 0.0, 1.0

    chi2 = (O_a_total - E_a_total) ** 2 / V_total
    p_val = float(scipy_stats.chi2.sf(chi2, df=1))
    return float(chi2), p_val


def km_curve(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier simples. Retorna (times, survival_probs)."""
    order = np.argsort(time)
    t_s, e_s = time[order], event[order]
    s = 1.0
    times_out, surv_out = [0.0], [1.0]
    for t in np.unique(t_s):
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
    thresholds = np.quantile(log_risk, np.linspace(0, 1, n_groups + 1))
    labels = np.digitize(log_risk, thresholds[1:-1])

    group_labels = (
        ["Baixo Risco", "Risco Intermediario", "Alto Risco"]
        if n_groups == 3
        else [f"Grupo {i+1}" for i in range(n_groups)]
    )

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
# 2. KS de sobrevivência
# ---------------------------------------------------------------------------

def compute_survival_ks(
    log_risk: np.ndarray,
    event: np.ndarray,
) -> dict:
    """KS entre a distribuicao de log-risco de eventos vs. censurados."""
    risk_event = np.sort(log_risk[event.astype(bool)])
    risk_censor = np.sort(log_risk[~event.astype(bool)])
    ks_stat, ks_pval = scipy_stats.ks_2samp(risk_event, risk_censor)

    n = len(log_risk)
    order = np.argsort(log_risk)[::-1]
    sorted_event = event[order].astype(float)
    total_events = event.sum()
    total_censored = (~event.astype(bool)).sum()

    decil_rows = []
    cumulative_events, cumulative_censored = 0.0, 0.0
    best_ks, best_decil = 0.0, 0

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
# 3. Calibracao por decil de risco (E/O ratio)
# ---------------------------------------------------------------------------

def compute_calibration_by_decil(
    log_risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    survival_at_t: np.ndarray,
    eval_time: float,
    n_decils: int = 10,
) -> pd.DataFrame:
    """Para um tempo de referencia eval_time, calcula E/O por decil de risco."""
    order = np.argsort(log_risk)
    n = len(log_risk)
    rows = []

    for d in range(n_decils):
        idx_start = int(n * d / n_decils)
        idx_end = int(n * (d + 1) / n_decils)
        idx = order[idx_start:idx_end]

        expected_surv = float(np.mean(survival_at_t[idx]))
        expected_event_rate = 1.0 - expected_surv

        t_dec, e_dec = time[idx], event[idx]
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
# 4. D de Royston
# ---------------------------------------------------------------------------

def compute_royston_d(
    log_risk: np.ndarray,
    event: np.ndarray,
) -> dict:
    """D de Royston-Sauerbrei como medida de discriminacao."""
    n = len(log_risk)
    ranks = scipy_stats.rankdata(log_risk) / (n + 1)
    ranks = np.clip(ranks, 1e-6, 1 - 1e-6)
    z = scipy_stats.norm.ppf(ranks)

    z_events = z[event.astype(bool)]
    if len(z_events) < 5:
        return {"D_royston": np.nan, "R2_royston": np.nan}

    D = float(np.std(z_events, ddof=1) * np.sqrt(8.0 / np.pi))
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
    """Estatisticas descritivas dos log-riscos separados por status."""
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
    """Decision Curve Analysis simplificada para survival."""
    n = len(log_risk)
    prob_event = 1.0 - survival_at_t
    obs_event = (event.astype(bool) & (time <= eval_time)).astype(float)

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    rows = []
    for pt in thresholds:
        predicted_pos = prob_event >= pt
        tp = float(np.sum(predicted_pos & obs_event.astype(bool)))
        fp = float(np.sum(predicted_pos & ~obs_event.astype(bool)))
        nb = (tp / n) - (fp / n) * (pt / max(1 - pt, 1e-8))

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
# Utilitarios de exportacao de curvas para CSV
# ---------------------------------------------------------------------------

def km_groups_to_csv(km_result: dict, output_dir: Path, prefix: str = "km") -> None:
    rows = []
    for group_name, gdata in km_result["groups"].items():
        for t, s in zip(gdata["times"], gdata["survival"]):
            rows.append({"grupo": group_name, "tempo": round(float(t), 4), "sobrevivencia": round(float(s), 4)})
    out = output_dir / f"{prefix}_curvas_km.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
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

    best_idx = grid_search.best_index_
    best_std = float(grid_search.cv_results_["std_test_score"][best_idx])
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
        "TOP 20 FEATURE IMPORTANCE (gain do XGBoost)",
        "=" * 70,
        xgb_gain_df.head(20).to_string(index=False),
        "",
        "=" * 70,
        "TOP 20 PERMUTATION IMPORTANCE (C-index no teste)",
        "=" * 70,
        perm_df.head(20).to_string(index=False),
        "",
        "=" * 70,
        "TOP 20 SHAP GLOBAL (mean(|SHAP|))",
        "=" * 70,
        shap_global_df.head(20).to_string(index=False),
        "",
        f"SHAP global CSV: {shap_csv_path}",
        f"SHAP summary plot: {shap_plot_path}",
        f"SHAP beeswarm plot: {shap_beeswarm_plot_path}",
        "",
    ]

    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nRelatorio salvo em: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        train_df,
        test_df,
        event_train,
        time_train,
        event_test,
        time_test,
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

    # Recupera os risk scores gerados internamente pelo compute_survival_metrics
    risk_test = metrics.pop("risk_test")

    print("\nCalculando metricas adicionais...")

    # 1. KM por grupos de risco (no conjunto de teste)
    km_result = compute_km_risk_groups(risk_test, time_test, event_test, n_groups=3)
    print(
        f"  Log-rank (baixo vs alto risco): chi2={km_result['logrank_chi2_low_vs_high']:.4f}, "
        f"p={km_result['logrank_pval_low_vs_high']:.4e}"
    )

    # 2. KS de sobrevivência
    ks_result = compute_survival_ks(risk_test, event_test)
    print(f"  KS estatistico: {ks_result['ks_stat']:.6f} (decil {ks_result['ks_best_decil']})")

    # 3. Calibracao por decil — tempo mediano de avaliacao como referencia
    calib_eval_time = float(np.median(metrics["eval_times"]))
    calib_time_idx = int(np.argmin(np.abs(metrics["eval_times"] - calib_eval_time)))
    survival_at_ref = metrics["survival_test"][:, calib_time_idx]
    calib_df = compute_calibration_by_decil(
        log_risk=risk_test,
        time=time_test,
        event=event_test,
        survival_at_t=survival_at_ref,
        eval_time=calib_eval_time,
    )

    # 4. D de Royston
    royston = compute_royston_d(risk_test, event_test)
    print(f"  D de Royston: {royston['D_royston']}, R2: {royston['R2_royston']}")

    # 5. Distribuicao dos log-riscos
    risk_dist = compute_risk_score_distribution(risk_test, event_test)

    # 6. Net Benefit
    net_benefit_df = compute_net_benefit(
        log_risk=risk_test,
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
        log_risk=risk_test,
        y_train_surv=y_train_surv,
        y_test_surv=y_test_surv,
        event_test=event_test,
        time_test=time_test,
        subgroup_series=subgroup_series,
    )

    # Importancias e SHAP
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

    # Minimo de eventos por fold observado
    event_train_int = y_train_surv["event"].astype(int)
    cv_probe = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    observed_min_events_per_fold = min(
        int(event_train_int[val_idx].sum())
        for _, val_idx in cv_probe.split(x_train, event_train_int)
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
        km_result=km_result,
        ks_result=ks_result,
        calib_df=calib_df,
        calib_eval_time=calib_eval_time,
        royston=royston,
        risk_dist=risk_dist,
        net_benefit_df=net_benefit_df,
        subgroup_df=subgroup_df,
    )

    print(f"Treinamento XGBoost Survival concluido. Resultados salvos em: {args.output}")


if __name__ == "__main__":
    main()