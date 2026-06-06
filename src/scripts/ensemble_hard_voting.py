#!/usr/bin/env python3

"""
Ensemble por hard voting (votacao majoritaria) entre XGBoost, Random Forest e Deep Learning.

Reutiliza os arquivos .npy de probabilidade ja gerados pelos scripts de treinamento.
Nao e necessario re-treinar nenhum modelo.

Arquivos esperados (no mesmo diretorio ou em --proba-dir):
  xgboost_test_proba.npy
  random_forest_test_proba.npy
  deep_learning_test_proba.npy
  y_test.npy

Uso:
  python ensemble_hard_voting.py [--proba-dir DIR] [--output FILE] [--threshold T]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
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
from sklearn.preprocessing import label_binarize


MODELS: dict[str, str] = {
    "XGBoost":       "xgboost_test_proba.npy",
    "Random Forest": "random_forest_test_proba.npy",
    "Deep Learning": "deep_learning_test_proba.npy",
}

Y_TEST_FILE = "y_test.npy"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble por hard voting com metricas completas de classificacao."
    )
    parser.add_argument(
        "--proba-dir", type=Path, default=Path("."),
        help="Diretorio onde estao os arquivos .npy (default: diretorio atual)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("ensemble_hard_results.txt"),
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Threshold para metricas individuais binarias (default: 0.5)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Carregamento e validacao
# ---------------------------------------------------------------------------
def load_probas(proba_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    probas: dict[str, np.ndarray] = {}
    ref_shape: tuple | None = None

    for name, fname in MODELS.items():
        fpath = proba_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Arquivo nao encontrado: {fpath}\n"
                f"Execute primeiro o script de treinamento de '{name}'."
            )
        arr = np.load(fpath)
        if arr.ndim != 2:
            raise ValueError(f"'{fname}' deve ter shape (N, n_classes), mas tem shape {arr.shape}.")
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            raise ValueError(
                f"Shape de '{fname}' ({arr.shape}) difere do primeiro modelo ({ref_shape}). "
                "Todos os modelos precisam usar o mesmo conjunto de teste."
            )
        probas[name] = arr

    y_path = proba_dir / Y_TEST_FILE
    if not y_path.exists():
        raise FileNotFoundError(f"Gabarito nao encontrado: {y_path}")
    y_test = np.load(y_path)

    if y_test.shape[0] != ref_shape[0]:
        raise ValueError(
            f"y_test tem {y_test.shape[0]} amostras mas as probabilidades tem {ref_shape[0]}."
        )

    return probas, y_test


# ---------------------------------------------------------------------------
# Hard voting
# ---------------------------------------------------------------------------
def hard_vote(probas: dict[str, np.ndarray]) -> np.ndarray:
    """
    Converte cada array de probabilidades na classe predita (argmax),
    empilha os votos e retorna a classe com mais votos por amostra.
    Com 3 modelos nunca ha empate.
    """
    preds = np.stack(
        [np.argmax(arr, axis=1) for arr in probas.values()],
        axis=1,
    )  # shape (N, 3)

    voted, _ = stats.mode(preds, axis=1, keepdims=True)
    return voted.ravel().astype(int)


# ---------------------------------------------------------------------------
# Metricas padronizadas (identicas ao soft voting e aos scripts individuais)
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
        "ks":             float(np.max(ks_values)),
        "ks_decile":      int(np.argmax(ks_values) + 1),
        "gain_by_decile": gains,
        "lift_by_decile": lifts,
        "decil_fracs":    decil_fracs,
    }


def compute_metrics_from_proba(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Metricas completas a partir de probabilidades (para modelos individuais)."""
    is_binary = len(np.unique(y_true)) <= 2
    metrics: dict = {}

    if is_binary:
        proba_pos = y_proba[:, 1]
        y_pred    = (proba_pos >= threshold).astype(int)
    else:
        y_pred    = np.argmax(y_proba, axis=1)
        proba_pos = None

    _fill_metrics(metrics, y_true, y_pred, y_proba, proba_pos, is_binary, threshold)
    return metrics


def compute_metrics_from_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Metricas do ensemble hard voting: sem probabilidades agregadas,
    por isso AUC-ROC, AUC-PR, Brier e Log-Loss nao sao calculados.
    KS/Ganho/Lift tambem nao se aplicam (sem score continuo).
    """
    is_binary = len(np.unique(y_true)) <= 2
    avg       = "binary" if is_binary else "weighted"
    metrics: dict = {
        "auc_roc":   float("nan"),  # indisponivel no hard voting
        "auc_pr":    float("nan"),
        "gini":      float("nan"),
        "brier_score": float("nan"),
        "log_loss":  float("nan"),
        "ks":        float("nan"),
        "ks_decile": float("nan"),
        "gain_by_decile": [],
        "lift_by_decile": [],
        "decil_fracs":    [],
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall":    recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1":        f1_score(y_true, y_pred, average=avg, zero_division=0),
        "mcc":       matthews_corrcoef(y_true, y_pred),
        "confusion_matrix":      confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "_is_binary": is_binary,
        "_y_pred":    y_pred,
    }
    return metrics


def _fill_metrics(
    metrics: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    proba_pos,
    is_binary: bool,
    threshold: float,
) -> None:
    avg = "binary" if is_binary else "weighted"

    # Discriminacao
    if is_binary:
        metrics["auc_roc"] = roc_auc_score(y_true, proba_pos)
        metrics["auc_pr"]  = average_precision_score(y_true, proba_pos)
    else:
        metrics["auc_roc"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="weighted"
        )
        classes = np.unique(y_true)
        ap_scores, weights = [], []
        for cls in classes:
            y_bin = (y_true == cls).astype(int)
            ap_scores.append(average_precision_score(y_bin, y_proba[:, cls]))
            weights.append(y_bin.sum())
        metrics["auc_pr"] = float(np.average(ap_scores, weights=weights))

    metrics["gini"] = 2 * metrics["auc_roc"] - 1

    # Predicao
    metrics["accuracy"]  = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["recall"]    = recall_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["f1"]        = f1_score(y_true, y_pred, average=avg, zero_division=0)
    metrics["mcc"]       = matthews_corrcoef(y_true, y_pred)

    # Calibracao
    if is_binary:
        metrics["brier_score"] = brier_score_loss(y_true, proba_pos)
    else:
        y_bin_matrix           = label_binarize(y_true, classes=np.unique(y_true))
        metrics["brier_score"] = float(np.mean((y_proba - y_bin_matrix) ** 2))
    metrics["log_loss"] = log_loss(y_true, y_proba)

    # Diagnostico
    metrics["confusion_matrix"]      = confusion_matrix(y_true, y_pred)
    metrics["classification_report"] = classification_report(y_true, y_pred, zero_division=0)

    # KS / Ganho / Lift
    if is_binary:
        kgl = compute_ks_gain_lift(y_true, proba_pos)
        metrics.update(kgl)
    else:
        metrics.update({"ks": float("nan"), "ks_decile": float("nan"),
                        "gain_by_decile": [], "lift_by_decile": [], "decil_fracs": []})

    metrics["_is_binary"] = is_binary
    metrics["_threshold"] = threshold
    metrics["_y_pred"]    = y_pred


# ---------------------------------------------------------------------------
# Formatacao do relatorio
# ---------------------------------------------------------------------------
def format_metrics_block(m: dict, label: str, is_hard_ensemble: bool = False) -> list[str]:
    is_binary = m.get("_is_binary", True)
    thr       = m.get("_threshold", "—")
    ks_decil  = m.get("ks_decile", "?")
    nan_note  = "  (n/d no hard voting — sem score continuo agregado)"

    def fmt(v):
        return f"{v:.6f}" if not (isinstance(v, float) and np.isnan(v)) else "     n/d"

    lines = [
        f"--- {label} ---",
        "",
        "  [Discriminacao]",
        f"  AUC-ROC       : {fmt(m['auc_roc'])}" + (nan_note if is_hard_ensemble else ""),
        f"  AUC-PR        : {fmt(m['auc_pr'])}",
        f"  Gini          : {fmt(m['gini'])}",
        "",
        f"  [Predicao — threshold = {thr}]",
        f"  Accuracy      : {fmt(m['accuracy'])}",
        f"  Precision     : {fmt(m['precision'])}",
        f"  Recall        : {fmt(m['recall'])}",
        f"  F1-Score      : {fmt(m['f1'])}",
        f"  MCC           : {fmt(m['mcc'])}",
        "",
        "  [Calibracao]",
        f"  Brier Score   : {fmt(m['brier_score'])}" + (nan_note if is_hard_ensemble else ""),
        f"  Log-Loss      : {fmt(m['log_loss'])}",
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
            f"  KS estatistico : {fmt(m['ks'])}   (maximo no decil {ks_decil})",
            "",
            f"  {'Decil':>5}  {'% base':>7}  {'Ganho':>7}  {'Lift':>7}",
            f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}",
        ]
        for i, (frac, gain, lift) in enumerate(
            zip(m["decil_fracs"], m["gain_by_decile"], m["lift_by_decile"]), start=1
        ):
            marker = " <- KS" if i == ks_decil else ""
            lines.append(f"  {i:>5}  {frac:>6.0%}  {gain:>7.4f}  {lift:>7.4f}{marker}")
    elif is_hard_ensemble and is_binary:
        lines += ["", "  [KS / Ganho / Lift]  n/d no hard voting — sem score continuo agregado"]

    return lines


def save_report(
    output_path: Path,
    y_ensemble: np.ndarray,
    ensemble_metrics: dict,
    individual_metrics: dict[str, dict],
    threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 70,
        "ENSEMBLE HARD VOTING (VOTACAO MAJORITARIA) — RESULTADOS",
        "=" * 70,
        f"Data/hora   : {datetime.now().isoformat(timespec='seconds')}",
        f"Threshold   : {threshold}   (usado nas metricas individuais)",
        "",
        "--- Modelos participantes ---",
    ]
    for name in MODELS:
        lines.append(f"  {name}")

    lines += ["", "=" * 70, "ENSEMBLE (hard voting)", "=" * 70,
              "  Nota: AUC-ROC, AUC-PR, Brier, Log-Loss e KS/Lift nao se aplicam",
              "  ao hard voting pois nao ha score continuo agregado.",
              "  Use o ensemble_soft_voting.py para obter essas metricas.", ""]
    lines += format_metrics_block(ensemble_metrics, "Metricas do Ensemble", is_hard_ensemble=True)

    lines += ["", "=" * 70, "MODELOS INDIVIDUAIS (para comparacao)", "=" * 70]
    for name, m in individual_metrics.items():
        lines += [""]
        lines += format_metrics_block(m, f"Metricas — {name}")

    # Tabela comparativa
    all_results = {"Ensemble (hard)": ensemble_metrics, **individual_metrics}
    lines += [
        "",
        "=" * 70,
        "TABELA COMPARATIVA",
        "=" * 70,
        f"  {'Modelo':<22}  {'AUC-ROC':>8}  {'AUC-PR':>8}  {'KS':>8}  "
        f"{'F1':>8}  {'MCC':>8}  {'Brier':>8}",
        f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}",
    ]

    def fmtc(v):
        return f"{v:.6f}" if not (isinstance(v, float) and np.isnan(v)) else "     n/d"

    for nome, m in all_results.items():
        lines.append(
            f"  {nome:<22}  {fmtc(m['auc_roc']):>8}  {fmtc(m['auc_pr']):>8}  "
            f"{fmtc(m['ks']):>8}  {m['f1']:>8.6f}  {m['mcc']:>8.6f}  {fmtc(m['brier_score']):>8}"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    probas, y_test = load_probas(args.proba_dir)

    print(f"Modelos carregados : {list(probas.keys())}")
    print(f"Shape das probas   : {next(iter(probas.values())).shape}")

    # Hard voting
    y_ensemble = hard_vote(probas)

    # Metricas do ensemble (apenas a partir das predicoes — sem proba agregada)
    ensemble_metrics = compute_metrics_from_pred(y_test, y_ensemble)

    # Metricas individuais com probabilidades completas (para comparacao)
    individual_metrics: dict[str, dict] = {}
    for name, proba in probas.items():
        individual_metrics[name] = compute_metrics_from_proba(
            y_test, proba, threshold=args.threshold
        )

    save_report(
        output_path=args.output,
        y_ensemble=y_ensemble,
        ensemble_metrics=ensemble_metrics,
        individual_metrics=individual_metrics,
        threshold=args.threshold,
    )

    print(f"\nRelatorio salvo em: {args.output}")
    print(f"\nResumo — Ensemble hard voting vs Individuais:")
    all_results = {"Ensemble (hard)": ensemble_metrics, **individual_metrics}
    print(f"  {'Modelo':<22}  {'F1':>8}  {'MCC':>8}  {'AUC-ROC':>8}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}")
    for nome, m in all_results.items():
        auc = f"{m['auc_roc']:.4f}" if not np.isnan(m["auc_roc"]) else "    n/d"
        print(f"  {nome:<22}  {m['f1']:>8.4f}  {m['mcc']:>8.4f}  {auc:>8}")


if __name__ == "__main__":
    main()