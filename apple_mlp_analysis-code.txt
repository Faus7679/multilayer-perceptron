from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load apple quality data and return processed features and labels."""
    data = pd.read_csv(csv_path)

    # Remove ID column and any rows with missing values.
    if "A_id" in data.columns:
        data = data.drop(columns=["A_id"])
    data = data.dropna(axis=0)

    # Separate predictors and target class.
    target = data["Quality"].astype(str).str.strip().str.lower()
    features = data.drop(columns=["Quality"])

    # Ensure all predictors are numeric.
    features = features.apply(pd.to_numeric, errors="coerce")
    valid_rows = features.notna().all(axis=1)
    features = features.loc[valid_rows]
    target = target.loc[valid_rows]

    return features, target


def build_classifiers(random_state: int = 42) -> dict[str, Pipeline]:
    """Create multiple MLP classifier pipelines for comparison."""
    return {
        "MLP_relu_adam": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=0.001,
                        max_iter=500,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "MLP_tanh_lbfgs": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(16, 8),
                        activation="tanh",
                        solver="lbfgs",
                        alpha=0.0005,
                        max_iter=800,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def evaluate_and_plot(
    model_name: str,
    model: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    label_encoder: LabelEncoder,
) -> dict[str, float | list[list[int]] | str]:
    """Train model, predict, compute metrics, and save confusion matrix plot."""
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    accuracy = accuracy_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.inverse_transform([0, 1]),
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{model_name} Confusion Matrix")
    fig.tight_layout()

    plot_path = output_dir / f"{model_name.lower()}_confusion_matrix.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.inverse_transform([0, 1]),
        digits=4,
        zero_division=0,
    )

    return {
        "classifier": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "plot_path": str(plot_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apple quality classification with MLP")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path.home() / "Downloads" / "apple_quality.csv",
        help="Path to apple_quality.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "figures",
        help="Directory where plots and metrics are saved",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data_path}. Place apple_quality.csv in your Downloads folder or pass --data-path."
        )

    sns.set_theme(style="whitegrid")

    x, y_raw = load_and_preprocess(args.data_path)

    # Label encode target for model compatibility.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Subset data to selected predictors (all numeric quality measurements).
    selected_features = [
        "Size",
        "Weight",
        "Sweetness",
        "Crunchiness",
        "Juiciness",
        "Ripeness",
        "Acidity",
    ]
    x_subset = x[selected_features].copy()

    # Split into training and test sets.
    x_train, x_test, y_train, y_test = train_test_split(
        x_subset,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    classifiers = build_classifiers(random_state=42)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, float | list[list[int]] | str]] = []
    for model_name, model in classifiers.items():
        results.append(
            evaluate_and_plot(
                model_name=model_name,
                model=model,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                output_dir=args.output_dir,
                label_encoder=label_encoder,
            )
        )

    metrics_df = pd.DataFrame(
        [
            {
                "classifier": r["classifier"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1_score": r["f1_score"],
            }
            for r in results
        ]
    ).sort_values("f1_score", ascending=False)

    metrics_output_path = args.output_dir.parent / "metrics_summary.csv"
    metrics_df.to_csv(metrics_output_path, index=False)

    print("\n=== Metrics Summary ===")
    print(metrics_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved metrics to: {metrics_output_path}")

    for result in results:
        print(f"\n=== {result['classifier']} ===")
        print(f"Confusion matrix: {result['confusion_matrix']}")
        print(result["classification_report"])
        print(f"Saved plot: {result['plot_path']}")


if __name__ == "__main__":
    main()
