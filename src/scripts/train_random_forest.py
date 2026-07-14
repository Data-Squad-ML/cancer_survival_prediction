#!/usr/bin/env python3

"""Treina Random Forest em CPU com metricas completas e exporta arrays para ensemble."""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
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
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize


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
# Utilitarios
# ---------------------------------------------------------------------------
def default_data_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / filename


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treino de Random Forest CPU com metricas completas e soft voting."
    )
    parser.add_argument("--train-path",     type=Path,  default=default_data_path("data_train.csv"))
    parser.add_argument("--test-path",      type=Path,  default=default_data_path("data_test.csv"))
    parser.add_argument("--target",         type=str,   default="status_vital")
    parser.add_argument("--output",         type=Path,  default=Path("random_forest_results.txt"))
    parser.add_argument("--cv",             type=int,   default=5)
    parser.add_argument("--perm-repeats",   type=int,   default=10)
    parser.add_argument("--random-state",   type=int,   default=42)
    parser.add_argument("--threshold",      type=float, default=0.5,
                        help="Threshold de probabilidade para classificacao binaria (default: 0.5)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def build_feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    ohe_cols = [
        col for col in train_df.columns
        if any(col.startswith(p) for p in OHE_PREFIXES)
    ]
    selected = [c for c in (FEATURE_COLS_BASE + ohe_cols) if c in train_df.columns]
    selected = [c for c in selected if c in test_df.columns]
    if not selected:
        raise ValueError("Nenhuma feature selecionada foi encontrada nos dados de treino/teste.")
    return selected


def load_split_data(train_path: Path, test_path: Path, target: str):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    if target not in train_df.columns or target not in test_df.columns:
        raise ValueError(f"A coluna alvo '{target}' nao existe em ambos os arquivos.")

    train_df = train_df.dropna(subset=[target]).copy()
    test_df  = test_df.dropna(subset=[target]).copy()

    feature_cols = build_feature_columns(train_df, test_df)

    x_train = train_df[feature_cols].astype(float)
    y_train = train_df[target].astype(int)
    x_test  = test_df[feature_cols].astype(float)
    y_test  = test_df[target].astype(int)

    w_train = (
        train_df["class_weight"].astype(float)
        if "class_weight" in train_df.columns else None
    )

    return x_train, y_train, x_test, y_test, w_train, feature_cols


def get_auc_mode(y: pd.Series) -> str:
    return "binary" if int(y.nunique()) <= 2 else "multiclass"


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def sanitize_params(params: dict) -> dict:
    return dict(params)


def _fit_rf(
    x_train_df: pd.DataFrame,
    y_train_sr: pd.Series,
    params: dict,
    random_state: int,
    sample_weight: pd.Series | None,
) -> RandomForestClassifier:
    params = sanitize_params(params)

    model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        **params,
    )

    if sample_weight is not None:
        model.fit(
            x_train_df,
            y_train_sr,
            sample_weight=sample_weight,
        )
    else:
        model.fit(
            x_train_df,
            y_train_sr,
        )

    return model


def _predict_proba_numpy(model: RandomForestClassifier, x_df: pd.DataFrame) -> np.ndarray:
    return _to_numpy(model.predict_proba(x_df))


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------
def _param_grid():
    grid = {
        "n_estimators":     [100, 200, 400],
        "max_depth":        [10, 20, None],
        "min_samples_split":[2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features":     ["sqrt", "log2"],
    }
    keys = list(grid.keys())
    for combo in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def _compute_auc_np(y_true: np.ndarray, y_proba: np.ndarray, auc_mode: str) -> float:
    if auc_mode == "binary":
        return roc_auc_score(y_true, y_proba[:, 1])
    return roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")


def run_grid_search(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series | None,
    cv: int,
    random_state: int,
    auc_mode: str,
) -> tuple[float, float, dict]:
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    all_params  = list(_param_grid())
    total       = len(all_params)
    print(f"Total de combinacoes no grid: {total}")

    best_score, best_std, best_params = -1.0, 0.0, None
    x_np = x_train.to_numpy()

    for i, params in enumerate(all_params, start=1):
        fold_scores = []

        for fold_train_idx, fold_val_idx in cv_splitter.split(x_np):
            x_ft = x_train.iloc[fold_train_idx]
            y_ft = y_train.iloc[fold_train_idx]
            x_fv = x_train.iloc[fold_val_idx]
            y_fv = y_train.iloc[fold_val_idx].to_numpy()
            w_ft = w_train.iloc[fold_train_idx] if w_train is not None else None

            model = _fit_rf(x_ft, y_ft, params, random_state, w_ft)
            proba = _predict_proba_numpy(model, x_fv)
            fold_scores.append(_compute_auc_np(y_fv, proba, auc_mode))

        mean_s = float(np.mean(fold_scores))
        std_s  = float(np.std(fold_scores))
        print(f"[{i}/{total}] cv_auc={mean_s:.6f} +/- {std_s:.6f} params={params}")

        if mean_s > best_score:
            best_score, best_std, best_params = mean_s, std_s, params

    return best_score, best_std, best_params


# ---------------------------------------------------------------------------
# KS, Ganho e Lift (binario apenas) — identico ao XGBoost
# ---------------------------------------------------------------------------
def compute_ks_gain_lift(
    y_true: np.ndarray,
    proba_pos: np.ndarray,
    n_deciles: int = 10,
) -> dict:
    n         = len(y_true)
    total_pos = y_true.sum()
    total_neg = n - total_pos

    if total_pos == 0 or total_neg == 0:
        return {"ks": float("nan"), "ks_decile": float("nan"),
                "gain_by_decile": [], "lift_by_decile": [], "decil_fracs": []}

    order    = np.argsort(proba_pos)[::-1]
    y_sorted = y_true[order]

    gains, lifts, decil_fracs, ks_values = [], [], [], []
    for d in range(1, n_deciles + 1):
        cutoff       = int(np.ceil(n * d / n_deciles))
        captured_pos = y_sorted[:cutoff].sum()
        captured_neg = cutoff - captured_pos
        frac_base    = cutoff / n
        gain         = captured_pos / total_pos
        lift         = gain / frac_base
        ks_values.append(captured_pos / total_pos - captured_neg / total_neg)
        decil_fracs.append(round(frac_base, 2))
        gains.append(round(float(gain), 4))
        lifts.append(round(float(lift), 4))

    return {
        "ks":              float(np.max(ks_values)),
        "ks_decile":       int(np.argmax(ks_values) + 1),
        "gain_by_decile":  gains,
        "lift_by_decile":  lifts,
        "decil_fracs":     decil_fracs,
    }


# ---------------------------------------------------------------------------
# Metricas completas padronizadas — mesmas chaves do XGBoost
# ---------------------------------------------------------------------------
def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    auc_mode: str,
    threshold: float = 0.5,
    split_name: str = "teste",
) -> dict:
    is_binary = auc_mode == "binary"
    metrics: dict = {}

    # Predicoes
    if is_binary:
        proba_pos = y_proba[:, 1]
        y_pred    = (proba_pos >= threshold).astype(int)
    else:
        y_pred    = np.argmax(y_proba, axis=1)
        proba_pos = None

    # 1. Discriminacao
    if is_binary:
        metrics["auc_roc"] = roc_auc_score(y_true, proba_pos)
        metrics["auc_pr"]  = average_precision_score(y_true, proba_pos)
    else:
        metrics["auc_roc"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="weighted"
        )
        classes    = np.unique(y_true)
        ap_scores, weights = [], []
        for cls in classes:
            y_bin = (y_true == cls).astype(int)
            ap_scores.append(average_precision_score(y_bin, y_proba[:, cls]))
            weights.append(y_bin.sum())
        metrics["auc_pr"] = float(np.average(ap_scores, weights=weights))

    metrics["gini"] = 2 * metrics["auc_roc"] - 1

    # 2. Predicao com threshold
    avg = "binary" if is_binary else "weighted"
    metrics["accuracy"]  = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["recall"]    = recall_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["f1"]        = f1_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["mcc"]       = matthews_corrcoef(y_true, y_pred)

    # 3. Calibracao
    if is_binary:
        metrics["brier_score"] = brier_score_loss(y_true, proba_pos)
    else:
        y_bin_matrix           = label_binarize(y_true, classes=np.unique(y_true))
        metrics["brier_score"] = float(np.mean((y_proba - y_bin_matrix) ** 2))

    metrics["log_loss"] = log_loss(y_true, y_proba)

    # 4. Diagnostico
    metrics["confusion_matrix"]      = confusion_matrix(y_true, y_pred)
    metrics["classification_report"] = classification_report(y_true, y_pred, zero_division=0)

    # 5. KS / Ganho / Lift
    if is_binary:
        kgl = compute_ks_gain_lift(y_true, proba_pos)
        metrics["ks"]             = kgl["ks"]
        metrics["ks_decile"]      = kgl["ks_decile"]
        metrics["gain_by_decile"] = kgl["gain_by_decile"]
        metrics["lift_by_decile"] = kgl["lift_by_decile"]
        metrics["decil_fracs"]    = kgl["decil_fracs"]
    else:
        metrics["ks"]             = float("nan")
        metrics["ks_decile"]      = float("nan")
        metrics["gain_by_decile"] = []
        metrics["lift_by_decile"] = []
        metrics["decil_fracs"]    = []

    # Dados para curvas (binario apenas)
    if is_binary:
        fpr, tpr, _         = roc_curve(y_true, proba_pos)
        prec_c, rec_c, _    = precision_recall_curve(y_true, proba_pos)
        frac_pos, mean_pred = calibration_curve(y_true, proba_pos, n_bins=10)
        metrics["_roc_curve"]         = (fpr, tpr)
        metrics["_pr_curve"]          = (prec_c, rec_c)
        metrics["_calibration_curve"] = (frac_pos, mean_pred)

    metrics["_y_pred"]   = y_pred
    metrics["_y_proba"]  = y_proba
    metrics["_split"]    = split_name
    metrics["_threshold"]= threshold
    return metrics


# ---------------------------------------------------------------------------
# Formatacao do relatorio
# ---------------------------------------------------------------------------
def format_metrics_block(m: dict, is_binary: bool) -> list[str]:
    split    = m["_split"].upper()
    thr      = m.get("_threshold", 0.5)
    ks_decil = m.get("ks_decile", "?")

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
        f"  Brier Score   : {m['brier_score']:.6f}   (menor e melhor)",
        f"  Log-Loss      : {m['log_loss']:.6f}",
        "",
        "  [Confusion Matrix]",
    ]
    for row in m["confusion_matrix"]:
        lines.append("  " + "  ".join(f"{v:6d}" for v in row))

    lines += ["", "  [Classification Report]"]
    for line in m["classification_report"].splitlines():
        lines.append("  " + line)

    if is_binary and m.get("gain_by_decile"):
        lines += [
            "",
            "  [KS / Ganho / Lift por Decil]",
            f"  KS estatistico : {m['ks']:.6f}   (maximo no decil {ks_decil})",
            "",
            f"  {'Decil':>5}  {'% base':>7}  {'Ganho':>7}  {'Lift':>7}",
            f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}",
        ]
        for i, (frac, gain, lift) in enumerate(
            zip(m["decil_fracs"], m["gain_by_decile"], m["lift_by_decile"]), start=1
        ):
            marker = " <- KS" if i == ks_decil else ""
            lines.append(
                f"  {i:>5}  {frac:>6.0%}  {gain:>7.4f}  {lift:>7.4f}{marker}"
            )

    return lines


# ---------------------------------------------------------------------------
# Permutation importance (usa AUC como metrica)
# ---------------------------------------------------------------------------
def permutation_importance_auc(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    auc_mode: str,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    rng      = np.random.default_rng(random_state)
    base_auc = _compute_auc_np(
        y_test.to_numpy(), _predict_proba_numpy(model, x_test), auc_mode
    )

    rows = []
    for col in x_test.columns:
        scores = []
        for _ in range(n_repeats):
            x_perm      = x_test.copy()
            x_perm[col] = rng.permutation(x_perm[col].to_numpy())
            perm_auc    = _compute_auc_np(
                y_test.to_numpy(), _predict_proba_numpy(model, x_perm), auc_mode
            )
            scores.append(base_auc - perm_auc)
        rows.append({
            "feature":         col,
            "importance_mean": float(np.mean(scores)),
            "importance_std":  float(np.std(scores)),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Salvar relatorio
# ---------------------------------------------------------------------------
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
    auc_mode: str,
    train_metrics: dict,
    test_metrics: dict,
    perm_df: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    is_binary = auc_mode == "binary"

    gap_auc   = train_metrics["auc_roc"]   - test_metrics["auc_roc"]
    gap_f1    = train_metrics["f1"]        - test_metrics["f1"]
    gap_brier = test_metrics["brier_score"]- train_metrics["brier_score"]

    lines = [
        "=" * 70,
        "RANDOM FOREST - RESULTADOS COMPLETOS",
        "=" * 70,
        f"Data/hora              : {datetime.now().isoformat(timespec='seconds')}",
        f"Coluna alvo            : {target}",
        f"Amostras de treino     : {train_size}",
        f"Amostras de teste      : {test_size}",
        f"Total de features      : {len(feature_cols)}",
        f"Uso de class_weight    : {'sim' if w_train_used else 'nao'}",
        "Metrica de Grid/CV     : roc_auc (KFold manual)",
        "",
        "--- Grid Search manual com KFold ---",
        f"Melhor score de CV     : {best_score:.6f}",
        f"Desvio padrao do score : {best_std:.6f}",
        f"Melhores hiperparametros: {best_params}",
        "",
    ]

    lines += format_metrics_block(train_metrics, is_binary)
    lines += [""]
    lines += format_metrics_block(test_metrics, is_binary)

    lines += [
        "",
        "--- Gaps treino → teste (sinal de overfitting) ---",
        f"  ΔAUC-ROC      : {gap_auc:+.6f}",
        f"  ΔF1-Score     : {gap_f1:+.6f}",
        f"  ΔBrier Score  : {gap_brier:+.6f}",
        "",
        "--- Top 20 Permutation Importance (teste) ---",
        perm_df.head(20).to_string(index=False),
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    x_train, y_train, x_test, y_test, w_train, feature_cols = load_split_data(
        train_path=args.train_path,
        test_path=args.test_path,
        target=args.target,
    )

    auc_mode = get_auc_mode(y_train)

    best_score, best_std, best_params = run_grid_search(
        x_train=x_train,
        y_train=y_train,
        w_train=w_train,
        cv=args.cv,
        random_state=args.random_state,
        auc_mode=auc_mode,
    )

    best_model = _fit_rf(
        x_train_df=x_train,
        y_train_sr=y_train,
        params=best_params,
        random_state=args.random_state,
        sample_weight=w_train,
    )

    train_proba = _predict_proba_numpy(best_model, x_train)
    test_proba  = _predict_proba_numpy(best_model, x_test)

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
        auc_mode=auc_mode,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        perm_df=perm_df,
    )

    print(f"\nTreinamento concluido. Resultados salvos em: {args.output}")
    print(f"\nResumo rapido — conjunto de TESTE:")
    print(f"  AUC-ROC    : {test_metrics['auc_roc']:.4f}")
    print(f"  AUC-PR     : {test_metrics['auc_pr']:.4f}")
    print(f"  KS         : {test_metrics['ks']:.4f}   (decil {test_metrics['ks_decile']})")
    print(f"  F1-Score   : {test_metrics['f1']:.4f}")
    print(f"  MCC        : {test_metrics['mcc']:.4f}")
    print(f"  Brier Score: {test_metrics['brier_score']:.4f}")
    print(f"  Log-Loss   : {test_metrics['log_loss']:.4f}")

    # --- Exporta arrays para uso no ensemble (soft voting) ---
    proba_path  = args.output.parent / "random_forest_test_proba.npy"
    y_test_path = args.output.parent / "y_test.npy"
    np.save(proba_path, test_metrics["_y_proba"])

    # y_test.npy: salva apenas se ainda nao existir (gerado pelo XGBoost)
    if not y_test_path.exists():
        np.save(y_test_path, y_test.to_numpy())
        print(f"  {y_test_path}  (criado)")
    else:
        print(f"  {y_test_path}  (ja existe — mantido)")

    print(f"\nArrays para ensemble salvos em:")
    print(f"  {proba_path}   (shape: {test_metrics['_y_proba'].shape})")


if __name__ == "__main__":
    main()
