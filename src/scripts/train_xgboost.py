#!/usr/bin/env python3

"""Treina XGBoost com logica alinhada ao notebook, com GPU NVIDIA automatica e fallback para CPU."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
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
    "historico_tabagismo_clinico_",
    "historico_alcoolismo_clinico_",
]


def default_data_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino de XGBoost com Grid Search, KFold e metricas completas de classificacao."
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
    # Threshold customizavel para metricas binarias (default 0.5)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold de probabilidade para classificacao binaria (default: 0.5)")
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
        "model__n_estimators": [400],
        "model__max_depth": [6],
        "model__learning_rate": [0.03],
        "model__subsample": [0.8],
        "model__colsample_bytree": [0.8],
        "model__min_child_weight": [3],
        "model__gamma": [0.0],
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


# ---------------------------------------------------------------------------
# KS, Curva de Ganho e Curva de Lift (apenas classificacao binaria)
# ---------------------------------------------------------------------------
def compute_ks_gain_lift(
    y_true: np.ndarray,
    proba_pos: np.ndarray,
    n_deciles: int = 10,
) -> dict:
    """
    Calcula KS estatistico, curva de ganho e curva de lift por decil.

    - KS   : maxima separacao entre a CDF dos positivos e a dos negativos.
    - Gain : fracao acumulada dos positivos reais capturada em cada decil.
    - Lift : ganho / fracao da base abordada (vs modelo aleatorio).

    Retorna dict com chaves:
      ks, ks_decile, gain_by_decile, lift_by_decile, decil_fracs
    """
    n = len(y_true)
    total_pos = y_true.sum()
    total_neg = n - total_pos

    if total_pos == 0 or total_neg == 0:
        return {
            "ks": float("nan"),
            "ks_decile": float("nan"),
            "gain_by_decile": [],
            "lift_by_decile": [],
            "decil_fracs": [],
        }

    # Ordena por score decrescente
    order = np.argsort(proba_pos)[::-1]
    y_sorted = y_true[order]

    gains, lifts, decil_fracs = [], [], []
    ks_values = []

    for d in range(1, n_deciles + 1):
        cutoff = int(np.ceil(n * d / n_deciles))
        captured_pos = y_sorted[:cutoff].sum()
        captured_neg = cutoff - captured_pos

        frac_base = cutoff / n
        gain = captured_pos / total_pos          # % dos positivos capturados
        lift = gain / frac_base                  # vs modelo aleatorio

        # KS = max(TPR - FPR) ao longo dos decis
        tpr = captured_pos / total_pos
        fpr = captured_neg / total_neg
        ks_values.append(tpr - fpr)

        decil_fracs.append(round(frac_base, 2))
        gains.append(round(float(gain), 4))
        lifts.append(round(float(lift), 4))

    ks_max = float(np.max(ks_values))
    ks_decile = int(np.argmax(ks_values) + 1)

    return {
        "ks": ks_max,
        "ks_decile": ks_decile,
        "gain_by_decile": gains,
        "lift_by_decile": lifts,
        "decil_fracs": decil_fracs,
    }


# ---------------------------------------------------------------------------
# BLOCO CENTRAL: calculo de todas as metricas padronizadas
# ---------------------------------------------------------------------------
def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    auc_mode: str,
    threshold: float = 0.5,
    split_name: str = "teste",
) -> dict:
    """
    Calcula o conjunto completo de metricas de classificacao.

    Metricas retornadas:
    - Discriminacao : AUC-ROC, AUC-PR (Average Precision), Gini
    - Predicao      : Accuracy, Precision, Recall, F1, MCC
    - Calibracao    : Brier Score, Log-Loss
    """
    metrics: dict = {}
    is_binary = auc_mode == "binary"

    # --- Probabilidades e predicoes ---
    if is_binary:
        proba_pos = y_proba[:, 1]
        y_pred = (proba_pos >= threshold).astype(int)
    else:
        y_pred = np.argmax(y_proba, axis=1)
        proba_pos = None  # sem sentido unico em multiclasse

    # --- 1. Discriminacao ---
    if is_binary:
        metrics["auc_roc"] = roc_auc_score(y_true, proba_pos)
        metrics["auc_pr"] = average_precision_score(y_true, proba_pos)
    else:
        metrics["auc_roc"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="weighted"
        )
        # AUC-PR: media ponderada das APs por classe (OVR)
        classes = np.unique(y_true)
        ap_scores = []
        weights = []
        for cls in classes:
            y_bin = (y_true == cls).astype(int)
            ap_scores.append(average_precision_score(y_bin, y_proba[:, cls]))
            weights.append(y_bin.sum())
        metrics["auc_pr"] = float(np.average(ap_scores, weights=weights))

    metrics["gini"] = 2 * metrics["auc_roc"] - 1

    # --- 2. Predicao com threshold ---
    avg = "binary" if is_binary else "weighted"
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # --- 3. Calibracao ---
    if is_binary:
        metrics["brier_score"] = brier_score_loss(y_true, proba_pos)
    else:
        # Brier Score multiclasse: media dos erros quadraticos por classe
        from sklearn.preprocessing import label_binarize
        y_bin_matrix = label_binarize(y_true, classes=np.unique(y_true))
        metrics["brier_score"] = float(np.mean((y_proba - y_bin_matrix) ** 2))

    metrics["log_loss"] = log_loss(y_true, y_proba)

    # --- 4. Diagnostico ---
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metrics["classification_report"] = classification_report(
        y_true, y_pred, zero_division=0
    )

    # --- 5. KS, Ganho e Lift (binario apenas) ---
    if is_binary:
        kgl = compute_ks_gain_lift(y_true, proba_pos)
        metrics["ks"] = kgl["ks"]
        metrics["ks_decile"] = kgl["ks_decile"]
        metrics["gain_by_decile"] = kgl["gain_by_decile"]
        metrics["lift_by_decile"] = kgl["lift_by_decile"]
        metrics["decil_fracs"] = kgl["decil_fracs"]
    else:
        metrics["ks"] = float("nan")
        metrics["ks_decile"] = float("nan")
        metrics["gain_by_decile"] = []
        metrics["lift_by_decile"] = []
        metrics["decil_fracs"] = []

    # --- Dados para curvas (binario apenas) ---
    if is_binary:
        fpr, tpr, _ = roc_curve(y_true, proba_pos)
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, proba_pos)
        frac_pos, mean_pred = calibration_curve(y_true, proba_pos, n_bins=10)
        metrics["_roc_curve"] = (fpr, tpr)
        metrics["_pr_curve"] = (prec_curve, rec_curve)
        metrics["_calibration_curve"] = (frac_pos, mean_pred)

    metrics["_split"] = split_name
    metrics["_threshold"] = threshold
    return metrics


def format_metrics_block(m: dict, is_binary: bool) -> list[str]:
    """Formata o dicionario de metricas em linhas de texto para o relatorio."""
    split = m["_split"].upper()
    thr = m.get("_threshold", 0.5)
    lines = [
        f"--- Metricas {split} ---",
        "",
        "  [Discriminacao]",
        f"  AUC-ROC       : {m['auc_roc']:.6f}",
        f"  AUC-PR        : {m['auc_pr']:.6f}   (Average Precision)",
        f"  Gini          : {m['gini']:.6f}",
        "",
        f"  [Predicao — threshold = {thr}]",
        f"  Accuracy      : {m['accuracy']:.6f}",
        f"  Precision     : {m['precision']:.6f}",
        f"  Recall        : {m['recall']:.6f}",
        f"  F1-Score      : {m['f1']:.6f}",
        f"  MCC           : {m['mcc']:.6f}",
        "",
        "  [Calibracao]",
        f"  Brier Score   : {m['brier_score']:.6f}   (menor e melhor; baseline = prevalencia*(1-prevalencia))",
        f"  Log-Loss      : {m['log_loss']:.6f}",
        "",
        "  [Confusion Matrix]",
    ]
    cm = m["confusion_matrix"]
    for row in cm:
        lines.append("  " + "  ".join(f"{v:6d}" for v in row))
    lines += [
        "",
        "  [Classification Report]",
    ]
    for line in m["classification_report"].splitlines():
        lines.append("  " + line)

    # KS, Ganho e Lift — apenas binario
    if is_binary and m.get("gain_by_decile"):
        ks_decil = m.get("ks_decile", "?")
        lines += [
            "",
            f"  [KS / Ganho / Lift por Decil]",
            f"  KS estatistico : {m['ks']:.6f}   (maximo no decil {ks_decil})",
            "",
            f"  {'Decil':>5}  {'% base':>7}  {'Ganho':>7}  {'Lift':>7}",
            f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}",
        ]
        for i, (frac, gain, lift) in enumerate(
            zip(m["decil_fracs"], m["gain_by_decile"], m["lift_by_decile"]), start=1
        ):
            ks_marker = " ← KS" if i == ks_decil else ""
            lines.append(
                f"  {i:>5}  {frac:>6.0%}  {gain:>7.4f}  {lift:>7.4f}{ks_marker}"
            )

    return lines


def save_results(
    output_path: Path,
    target: str,
    train_size: int,
    test_size: int,
    feature_cols: list[str],
    w_train_used: bool,
    scoring: str,
    xgb_mode: str,
    train_metrics: dict,
    test_metrics: dict,
    auc_mode: str,
    grid_search: GridSearchCV,
    perm_df: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_idx = grid_search.best_index_
    best_std = float(grid_search.cv_results_["std_test_score"][best_idx])
    is_binary = auc_mode == "binary"

    # Gap das principais metricas
    gap_auc = train_metrics["auc_roc"] - test_metrics["auc_roc"]
    gap_f1 = train_metrics["f1"] - test_metrics["f1"]
    gap_brier = test_metrics["brier_score"] - train_metrics["brier_score"]  # positivo = pior no teste

    report_lines = [
        "=" * 70,
        "XGBOOST - RESULTADOS COMPLETOS",
        "=" * 70,
        f"Data/hora              : {datetime.now().isoformat(timespec='seconds')}",
        f"Coluna alvo            : {target}",
        f"Modo de execucao       : {xgb_mode}",
        f"Amostras de treino     : {train_size}",
        f"Amostras de teste      : {test_size}",
        f"Total de features      : {len(feature_cols)}",
        f"Uso de class_weight    : {'sim' if w_train_used else 'nao'}",
        f"Metrica de Grid/CV     : {scoring}",
        "",
        "--- Grid Search com KFold ---",
        f"Melhor score de CV     : {grid_search.best_score_:.6f}",
        f"Desvio padrao do score : {best_std:.6f}",
        f"Melhores hiperparametros: {grid_search.best_params_}",
        "",
    ]

    report_lines += format_metrics_block(train_metrics, is_binary)
    report_lines += [""]
    report_lines += format_metrics_block(test_metrics, is_binary)

    report_lines += [
        "",
        "--- Gaps treino → teste (sinal de overfitting) ---",
        f"  ΔAUC-ROC      : {gap_auc:+.6f}   (positivo = melhor no treino)",
        f"  ΔF1-Score     : {gap_f1:+.6f}   (positivo = melhor no treino)",
        f"  ΔBrier Score  : {gap_brier:+.6f}   (positivo = pior calibracao no teste)",
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

    # Calcula metricas completas para treino e teste
    train_metrics = compute_all_metrics(
        y_true=y_train.to_numpy(),
        y_proba=train_proba,
        auc_mode=auc_mode,
        threshold=args.threshold,
        split_name="treino",
    )
    test_metrics = compute_all_metrics(
        y_true=y_test.to_numpy(),
        y_proba=test_proba,
        auc_mode=auc_mode,
        threshold=args.threshold,
        split_name="teste",
    )

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
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        auc_mode=auc_mode,
        grid_search=grid_search,
        perm_df=perm_df,
    )

    print(f"\nTreinamento concluido em modo {xgb_mode}.")
    print(f"Resultados salvos em: {args.output}")
    print(f"\nResumo rapido — conjunto de TESTE:")
    print(f"  AUC-ROC    : {test_metrics['auc_roc']:.4f}")
    print(f"  AUC-PR     : {test_metrics['auc_pr']:.4f}")
    print(f"  KS         : {test_metrics['ks']:.4f}   (decil {test_metrics['ks_decile']})")
    print(f"  F1-Score   : {test_metrics['f1']:.4f}")
    print(f"  MCC        : {test_metrics['mcc']:.4f}")
    print(f"  Brier Score: {test_metrics['brier_score']:.4f}")
    print(f"  Log-Loss   : {test_metrics['log_loss']:.4f}")


if __name__ == "__main__":
    main()