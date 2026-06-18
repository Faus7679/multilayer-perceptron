# multilayer-perceptron

This repository contains a complete multilayer perceptron (MLP) classification workflow for the `apple_quality.csv` dataset.

## Files
- `apple_mlp_analysis.py`: end-to-end data loading, preprocessing, model training, prediction, and evaluation.
- `reports/apple_quality_mlp_report.md`: 500–750 word technical report with code, outputs, plots, analysis, and references.

## Run
```bash
python apple_mlp_analysis.py --data-path ~/Downloads/apple_quality.csv
```

Generated outputs:
- `reports/metrics_summary.csv`
- `reports/figures/*.png`
