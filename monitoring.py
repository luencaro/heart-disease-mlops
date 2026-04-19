"""
Etapa 6: Monitoreo de deriva de datos con Evidently.
Archivo: monitoring.py
Proyecto: Heart Disease MLOps - Dr. Lihki Rubio

Requiere: evidently==0.4.33

Uso:
    python monitoring.py --train data/heart.csv --output drift_report.html
"""

import argparse
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
from evidently.pipeline.column_mapping import ColumnMapping

# Features numericas continuas
NUMERICAL_COLS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

# Features categoricas originales (antes del OHE)
CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]


def load_data(path):
    """Carga el CSV original sin transformar para el reporte de drift."""
    df = pd.read_csv(path)
    if "HeartDisease" in df.columns:
        df = df.drop("HeartDisease", axis=1)
    # Convertir a float64 para que Evidently muestre histogramas continuos
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def get_column_mapping():
    """Mapeo explícito: numericas con histogramas, categoricas con barras."""
    return ColumnMapping(
        numerical_features=NUMERICAL_COLS,
        categorical_features=CATEGORICAL_COLS,
    )


def simulate_production_data(df_train, drift_fraction=0.3, random_state=42):
    """
    Simula datos de produccion introduciendo deriva en features numericas.
    En un proyecto real estos datos vendrian de predicciones reales en produccion.
    """
    np.random.seed(random_state)
    df_prod = df_train.copy()
    n = len(df_prod)

    for col in NUMERICAL_COLS:
        if col not in df_prod.columns:
            continue
        df_prod[col] = df_prod[col].astype(float)
        mask = np.random.rand(n) < drift_fraction
        shift = df_prod[col].std() * 0.8
        df_prod.loc[mask, col] = df_prod.loc[mask, col] + shift

    return df_prod.sample(frac=0.4, random_state=random_state).reset_index(drop=True)


def generate_drift_report(df_reference, df_current, column_mapping, output_path="drift_report.html"):
    """Genera el reporte HTML de data drift con Evidently."""
    report = Report(metrics=[
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(
        reference_data=df_reference,
        current_data=df_current,
        column_mapping=column_mapping,
    )
    report.save_html(output_path)
    print(f"Reporte guardado en: {output_path}")
    return report


def print_summary(report):
    """Imprime resumen de resultados en consola."""
    result = report.as_dict()
    metrics = result.get("metrics", [])

    print("\n" + "=" * 50)
    print("  RESUMEN DE MONITOREO - Heart Disease MLOps")
    print("=" * 50)

    for m in metrics:
        name = m.get("metric", "")
        result_val = m.get("result", {})

        if "DatasetDriftMetric" in name:
            drifted = result_val.get("number_of_drifted_columns", "?")
            total = result_val.get("number_of_columns", "?")
            share = result_val.get("share_of_drifted_columns", 0)
            drift_detected = result_val.get("dataset_drift", False)
            status = "DRIFT DETECTADO" if drift_detected else "Sin drift significativo"
            print(f"\n  Estado : {status}")
            print(f"  Features con drift: {drifted} / {total} ({share * 100:.1f}%)")

        if "DatasetMissingValuesMetric" in name:
            missing = result_val.get("current", {}).get("number_of_missing_values", 0)
            print(f"  Valores nulos en produccion: {missing}")

    print("=" * 50)
    print("  Abre drift_report.html en el navegador para el reporte completo.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera reporte de data drift con Evidently.")
    parser.add_argument("--train", default="data/heart.csv", help="CSV de entrenamiento (referencia)")
    parser.add_argument("--output", default="drift_report.html", help="Ruta del reporte HTML")
    parser.add_argument("--drift", type=float, default=0.3, help="Fraccion de drift a simular (0-1)")
    args = parser.parse_args()

    print(f"Cargando datos de referencia desde: {args.train}")
    df_train = load_data(args.train)
    print(f"  Columnas: {df_train.columns.tolist()}")
    print(f"  Forma:    {df_train.shape}")

    print("Simulando datos de produccion con deriva...")
    df_prod = simulate_production_data(df_train, drift_fraction=args.drift)
    print(f"  Forma: {df_prod.shape}")

    print("Generando reporte Evidently...")
    column_mapping = get_column_mapping()
    report = generate_drift_report(df_train, df_prod, column_mapping, output_path=args.output)

    print_summary(report)