#!/usr/bin/env python3

"""
Ensemble por soft voting (media de probabilidades) entre modelos de classificacao.

Cada modelo deve ter exportado previamente um arquivo .npy com shape (N, n_classes)
contendo as probabilidades do conjunto de teste. O gabarito y_test.npy deve ser
identico para todos os modelos (mesmo split de teste).

Convencao de nomes esperada (configuravel em MODELS abaixo):
  "xgboost_test_proba.npy",
  "random_forest_test_proba.npy",
  "deep_learning_test_proba.npy",
  "y_test.npy"

Uso:
  python ensemble_soft_voting.py [--proba-dir DIR] [--output FILE] [--threshold T] [--weights w1 w2 ...]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
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
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# Configuracao dos modelos participantes do ensemble
# Chave : nome legivel para o relatorio
# Valor : nome do arquivo .npy com as probabilidades de teste
# ---------------------------------------------------------------------------
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
        description="Ensemble por soft voting com metricas completas de classificacao."
    )
    parser.add_argument(
        "--proba-dir", type=Path, default=Path("."),
        help="Diretorio onde estao os arquivos .npy (default: diretorio atual)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("ensemble_results.txt"),
        help="Arquivo de saida com o relatorio (default: ensemble_results.txt)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Threshold para classificacao binaria (default: 0.5)",
    )
    parser.add_argument(
        "--weights", type=float, nargs="+", default=None,
        help=(
            "Pesos para cada modelo na ordem definida em MODELS "
            "(ex: --weights 2 1 1). Omitir = pesos iguais."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Carregamento e validacao
# ---------------------------------------------------------------------------
def load_probas(proba_dir: Path, weights: list[float] | None) -> tuple[dict, np.ndarray, list[float]]:
    """
    Carrega os arrays de probabilidade de cada modelo e o gabarito.
    Retorna (probas_dict, y_test, weights_normalized).
    """
    model_names = list(MODELS.keys())
    model_files = list(MODELS.values())

    if weights is not None and len(weights) != len(model_names):
        raise ValueError(
            f"--weights tem {len(weights)} valores mas ha {len(model_names)} modelos em MODELS."
        )

    probas: dict[str, np.ndarray] = {}
    ref_shape: tuple | None = None

    for name, fname in zip(model_names, model_files):
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

    # Normaliza pesos
    if weights is None:
        w_norm = [1.0 / len(probas)] * len(probas)
    else:
        total = sum(weights)
        w_norm = [w / total for w in weights]

    return probas, y_test, w_norm


# ---------------------------------------------------------------------------
# Soft voting
# ---------------------------------------------------------------------------
def soft_vote(probas: dict[str, np.ndarray], weights: list[float]) -> np.ndarray:
    """
    Media ponderada das probabilidades de cada modelo.
    Retorna array (N, n_classes) com as probabilidades do ensemble.
    """
    arrays = list(probas.values())
    # stack: (n_modelos, N, n_classes) → media ponderada ao longo do eixo 0
    stacked = np.stack(arrays, axis=0)                        # (M, N, C)
    w_arr = np.array(weights)[:, np.newaxis, np.newaxis]      # (M, 1, 1)
    return np.sum(stacked * w_arr, axis=0)                    # (N, C)


# ---------------------------------------------------------------------------
# Metricas — reusa a mesma logica padronizada dos scripts individuais
# ---------------------------------------------------------------------------
def _detect_mode(y_true: np.ndarray) -> str:
    return "binary" if len(np.unique(y_true)) <= 2 else "multiclass"


def compute_ks_gain_lift(y_true: np.ndarray, proba_pos: np.ndarray, n_deciles: int = 10) -> dict:
    n = len(y_true)
    total_pos = y_true.sum()
    total_neg = n - total_pos

    if total_pos == 0 or total_neg == 0:
        return {"ks": float("nan"), "ks_decile": float("nan"),
                "gain_by_decile": [], "lift_by_decile": [], "decil_fracs": []}

    order = np.argsort(proba_pos)[::-1]
    y_sorted = y_true[order]

    gains, lifts, decil_fracs, ks_values = [], [], [], []
    for d in range(1, n_deciles + 1):
        cutoff = int(np.ceil(n * d / n_deciles))
        captured_pos = y_sorted[:cutoff].sum()
        captured_neg = cutoff - captured_pos
        frac_base = cutoff / n
        gain = captured_pos / total_pos
        lift = gain / frac_base
        ks_values.append(captured_pos / total_pos - captured_neg / total_neg)
        decil_fracs.append(round(frac_base, 2))
        gains.append(round(float(gain), 4))
        lifts.append(round(float(lift), 4))

    return {
        "ks": float(np.max(ks_values)),
        "ks_decile": int(np.argmax(ks_values) + 1),
        "gain_by_decile": gains,
        "lift_by_decile": lifts,
        "decil_fracs": decil_fracs,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Mesmas metricas padronizadas dos scripts individuais."""
    auc_mode = _detect_mode(y_true)
    is_binary = auc_mode == "binary"
    metrics: dict = {}

    if is_binary:
        proba_pos = y_proba[:, 1]
        y_pred = (proba_pos >= threshold).astype(int)
    else:
        y_pred = np.argmax(y_proba, axis=1)
        proba_pos = None

    # 1. Discriminacao
    if is_binary:
        metrics["auc_roc"] = roc_auc_score(y_true, proba_pos)
        metrics["auc_pr"]  = average_precision_score(y_true, proba_pos)
    else:
        metrics["auc_roc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        classes = np.unique(y_true)
        ap_scores, w = [], []
        for cls in classes:
            y_bin = (y_true == cls).astype(int)
            ap_scores.append(average_precision_score(y_bin, y_proba[:, cls]))
            w.append(y_bin.sum())
        metrics["auc_pr"] = float(np.average(ap_scores, weights=w))

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
        from sklearn.preprocessing import label_binarize
        y_bin_matrix = label_binarize(y_true, classes=np.unique(y_true))
        metrics["brier_score"] = float(np.mean((y_proba - y_bin_matrix) ** 2))
    metrics["log_loss"] = log_loss(y_true, y_proba)

    # 4. Diagnostico
    metrics["confusion_matrix"]      = confusion_matrix(y_true, y_pred)
    metrics["classification_report"] = classification_report(y_true, y_pred, zero_division=0)

    # 5. KS / Ganho / Lift
    if is_binary:
        kgl = compute_ks_gain_lift(y_true, proba_pos)
        metrics.update(kgl)
    else:
        metrics.update({"ks": float("nan"), "ks_decile": float("nan"),
                        "gain_by_decile": [], "lift_by_decile": [], "decil_fracs": []})

    metrics["_is_binary"]  = is_binary
    metrics["_threshold"]  = threshold
    metrics["_y_pred"]     = y_pred
    metrics["_y_proba"]    = y_proba
    return metrics


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------
def format_metrics_block(m: dict, label: str) -> list[str]:
    is_binary = m["_is_binary"]
    thr = m["_threshold"]
    ks_decil = m.get("ks_decile", "?")

    lines = [
        f"--- {label} ---",
        "",
        "  [Discriminacao]",
        f"  AUC-ROC       : {m['auc_roc']:.6f}",
        f"  AUC-PR        : {m['auc_pr']:.6f}",
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
        f"  Brier Score   : {m['brier_score']:.6f}",
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
            f"  [KS / Ganho / Lift por Decil]",
            f"  KS estatistico : {m['ks']:.6f}   (maximo no decil {ks_decil})",
            "",
            f"  {'Decil':>5}  {'% base':>7}  {'Ganho':>7}  {'Lift':>7}",
            f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}",
        ]
        for i, (frac, gain, lift) in enumerate(
            zip(m["decil_fracs"], m["gain_by_decile"], m["lift_by_decile"]), start=1
        ):
            marker = " <- KS" if i == ks_decil else ""
            lines.append(f"  {i:>5}  {frac:>6.0%}  {gain:>7.4f}  {lift:>7.4f}{marker}")

    return lines


def save_report(
    output_path: Path,
    probas: dict[str, np.ndarray],
    weights: list[float],
    ensemble_metrics: dict,
    individual_metrics: dict[str, dict],
    threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 70,
        "ENSEMBLE SOFT VOTING — RESULTADOS",
        "=" * 70,
        f"Data/hora   : {datetime.now().isoformat(timespec='seconds')}",
        f"Threshold   : {threshold}",
        "",
        "--- Modelos e pesos ---",
    ]
    for (name, _), w in zip(MODELS.items(), weights):
        lines.append(f"  {name:<20} peso = {w:.4f}")

    lines += ["", "=" * 70, "ENSEMBLE (soft voting)", "=" * 70]
    lines += format_metrics_block(ensemble_metrics, "Metricas do Ensemble")

    lines += ["", "=" * 70, "MODELOS INDIVIDUAIS (para comparacao)", "=" * 70]
    for name, m in individual_metrics.items():
        lines += [""]
        lines += format_metrics_block(m, f"Metricas — {name}")

    # Tabela comparativa resumida
    all_results = {"Ensemble": ensemble_metrics, **individual_metrics}
    lines += [
        "",
        "=" * 70,
        "TABELA COMPARATIVA",
        "=" * 70,
        f"  {'Modelo':<20}  {'AUC-ROC':>8}  {'AUC-PR':>8}  {'KS':>8}  "
        f"{'F1':>8}  {'MCC':>8}  {'Brier':>8}  {'LogLoss':>8}",
        f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}",
    ]
    for nome, m in all_results.items():
        ks_val = m.get("ks", float("nan"))
        ks_str = f"{ks_val:.6f}" if not (isinstance(ks_val, float) and np.isnan(ks_val)) else "     n/a"
        lines.append(
            f"  {nome:<20}  {m['auc_roc']:>8.6f}  {m['auc_pr']:>8.6f}  {ks_str:>8}  "
            f"{m['f1']:>8.6f}  {m['mcc']:>8.6f}  {m['brier_score']:>8.6f}  {m['log_loss']:>8.6f}"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    probas, y_test, weights = load_probas(args.proba_dir, args.weights)

    print(f"Modelos carregados: {list(probas.keys())}")
    print(f"Pesos normalizados: {[round(w, 4) for w in weights]}")
    print(f"Shape das probabilidades: {next(iter(probas.values())).shape}")

    # Soft voting: media ponderada das probabilidades
    ensemble_proba = soft_vote(probas, weights)

    # Metricas do ensemble
    ensemble_metrics = compute_all_metrics(y_test, ensemble_proba, threshold=args.threshold)

    # Metricas individuais de cada modelo (para comparacao no relatorio)
    individual_metrics: dict[str, dict] = {}
    for name, proba in probas.items():
        individual_metrics[name] = compute_all_metrics(y_test, proba, threshold=args.threshold)

    save_report(
        output_path=args.output,
        probas=probas,
        weights=weights,
        ensemble_metrics=ensemble_metrics,
        individual_metrics=individual_metrics,
        threshold=args.threshold,
    )

    print(f"\nRelatorio salvo em: {args.output}")
    print(f"\nResumo rapido — Ensemble vs Individuais:")
    all_results = {"Ensemble": ensemble_metrics, **individual_metrics}
    print(f"  {'Modelo':<20}  {'AUC-ROC':>8}  {'KS':>8}  {'F1':>8}  {'MCC':>8}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for nome, m in all_results.items():
        ks_val = m.get("ks", float("nan"))
        ks_str = f"{ks_val:.4f}" if not (isinstance(ks_val, float) and np.isnan(ks_val)) else "   n/a"
        print(f"  {nome:<20}  {m['auc_roc']:>8.4f}  {ks_str:>8}  {m['f1']:>8.4f}  {m['mcc']:>8.4f}")


if __name__ == "__main__":
    main()
