#!/usr/bin/env python3

"""Treina AutoGluon Tabular com prioridade para modelos que aproveitam GPU."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score


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
            "Treino AutoML com AutoGluon Tabular para classificacao, "
            "sem limite de tempo e com prioridade para GPU."
        )
    )
    parser.add_argument("--train-path", type=Path, default=default_data_path("data_train.csv"))
    parser.add_argument("--test-path", type=Path, default=default_data_path("data_test.csv"))
    parser.add_argument("--target", type=str, default="status_vital")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("results/autogluon_tabular_models"),
        help="Pasta onde o AutoGluon salva modelos e artefatos.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("results/autogluon_tabular_results.txt"),
    )
    parser.add_argument(
        "--leaderboard-path",
        type=Path,
        default=Path("results/autogluon_tabular_leaderboard.csv"),
    )
    parser.add_argument(
        "--presets",
        type=str,
        default="high_quality",
        choices=["best_quality", "high_quality", "good_quality", "medium_quality"],
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--verbosity", type=int, default=2)
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


def load_split_data(train_path: Path, test_path: Path, target: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target not in train_df.columns or target not in test_df.columns:
        raise ValueError(f"A coluna alvo '{target}' nao existe em ambos os arquivos.")

    train_df = train_df.dropna(subset=[target]).copy()
    test_df = test_df.dropna(subset=[target]).copy()

    feature_cols = build_feature_columns(train_df, test_df)

    x_train = train_df[feature_cols].copy() #deixando o autogluon inferir os tipos
    y_train = train_df[target].astype(int)
    x_test = test_df[feature_cols].copy()
    y_test = test_df[target].astype(int)

    if y_train.nunique() < 2:
        raise ValueError("Treino invalido: a coluna alvo tem apenas uma classe.")

    train_data = x_train.copy()
    train_data[target] = y_train

    test_data = x_test.copy()
    test_data[target] = y_test

    return train_data, test_data, feature_cols


def build_hyperparameters(num_gpus: int) -> dict:
    gpu_fit = {"ag_args_fit": {"num_gpus": max(0, int(num_gpus))}}

    return {
        "GBM": [gpu_fit],
        "XGB": [gpu_fit],
        "CAT": [gpu_fit],
        "NN_TORCH": [gpu_fit],
        "RF": {},
        "XT": {},
    }


def save_report(
    report_path: Path,
    target: str,
    presets: str,
    num_gpus: int,
    train_rows: int,
    test_rows: int,
    feature_cols: list[str],
    best_model: str,
    test_auc: float,
    test_metrics: dict,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 70,
        "AUTOGLOUON TABULAR - RESULTADOS",
        "=" * 70,
        f"Data/hora: {datetime.now().isoformat(timespec='seconds')}",
        f"Coluna alvo: {target}",
        f"Presets: {presets}",
        f"Num GPUs configurado: {num_gpus}",
        f"Amostras de treino: {train_rows}",
        f"Amostras de teste: {test_rows}",
        f"Total de features selecionadas: {len(feature_cols)}",
        f"Melhor modelo (leaderboard): {best_model}",
        f"ROC-AUC no teste: {test_auc:.6f}",
        "",
        "--- Metricas de evaluate() no teste ---",
    ]

    for metric_name, metric_value in sorted(test_metrics.items()):
        try:
            lines.append(f"{metric_name}: {float(metric_value):.6f}")
        except Exception:
            lines.append(f"{metric_name}: {metric_value}")

    lines.append("")
    lines.append("--- Features usadas (anti-vazamento) ---")
    lines.extend(feature_cols)
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    train_data, test_data, feature_cols = load_split_data(
        train_path=args.train_path,
        test_path=args.test_path,
        target=args.target,
    )

    hyperparameters = build_hyperparameters(num_gpus=args.num_gpus)

    predictor = TabularPredictor(
        label=args.target,
        problem_type="binary",
        eval_metric="roc_auc",
        path=str(args.model_dir),
    )

    predictor.fit(
        train_data=train_data,
        presets=args.presets,
        hyperparameters=hyperparameters,
        verbosity=args.verbosity,
        time_limit=None,
        num_bag_folds=5,
        num_stack_levels=1,
        refit_full=False,
        set_best_to_refit_full=False,
    )

    leaderboard = predictor.leaderboard(test_data, silent=True)
    args.leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(args.leaderboard_path, index=False)

    leaderboard_train = predictor.leaderboard(train_data, silent=True)
    leaderboard_train.to_csv(args.leaderboard_path.parent / "leaderboard_train.csv", index=False)

    y_true = test_data[args.target]
    y_pred_proba = predictor.predict_proba(test_data)
    if isinstance(y_pred_proba, pd.DataFrame):
        if 1 in y_pred_proba.columns:
            y_score = y_pred_proba[1]
        else:
            y_score = y_pred_proba.iloc[:, -1]
    else:
        y_score = y_pred_proba

    test_auc = roc_auc_score(y_true, y_score)
    test_metrics = predictor.evaluate(test_data, silent=True)

    best_model = leaderboard.iloc[0]["model"] if not leaderboard.empty else "N/A"

    save_report(
        report_path=args.report_path,
        target=args.target,
        presets=args.presets,
        num_gpus=args.num_gpus,
        train_rows=len(train_data),
        test_rows=len(test_data),
        feature_cols=feature_cols,
        best_model=str(best_model),
        test_auc=float(test_auc),
        test_metrics=test_metrics,
    )

    print("Treino AutoGluon finalizado.")
    print(f"Melhor modelo: {best_model}")
    print(f"ROC-AUC teste: {test_auc:.6f}")
    print(f"Leaderboard salvo em: {args.leaderboard_path}")
    print(f"Relatorio salvo em: {args.report_path}")


if __name__ == "__main__":
    main()
