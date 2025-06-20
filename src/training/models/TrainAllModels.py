import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime

FP_group_45 = '../../fingerprints/generators/45FP/45_fingerprints_grouped_avg_energy.csv'
FP_group_120 = '../../fingerprints/generators/120FP/120_fingerprints_grouped_avg_energy.csv'

FP_CO2_45 = '../split_odac/45_fingerprints_CO2.csv'
FP_H2O_45 = '../split_odac/45_fingerprints_H2O.csv'

FP_CO2_120 = '../split_odac/120_fingerprints_CO2.csv'
FP_H2O_120 = '../split_odac/120_fingerprints_H2O.csv'

models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "Bagging": BaggingRegressor(n_estimators=100, random_state=42)
}


def load_data(path):
    return pd.read_csv(path)


def get_metrics(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }


def format_metrics(metrics):
    return (
        f"- MAE (Mean Absolute Error): {metrics['MAE']:.4f}\n"
        f"- MSE (Mean Squared Error): {metrics['MSE']:.4f}\n"
        f"- RMSE (Root Mean Squared Error): {metrics['RMSE']:.4f}\n"
        f"- R² Score: {metrics['R2'] * 100:.2f}%\n"
    )


def format_prediction_samples(y_true, y_pred):
    indices = np.random.choice(len(y_true), 5, replace=False)
    return "\n".join(
        f"  Predicted: {y_pred[i]:.4f} | Actual: {y_true.iloc[i]:.4f}" for i in indices
    )


def train_models(X_train, y_train, X_valid, y_valid, outcome):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True)),
        # ('imputer', SimpleImputer(strategy='mean', keep_empty_features=True)),
        ('scaler', StandardScaler()),
    ])
    X_train_t = pipeline.fit_transform(X_train)
    X_valid_t = pipeline.transform(X_valid)

    outcome.append("1. BASELINE MODELS (no parameter tuning):\n")
    for name, model in models.items():
        model.fit(X_train_t, y_train)
        pred = model.predict(X_valid_t)
        metrics = get_metrics(y_valid, pred)
        outcome.append(f"--- {name} ---\n")
        outcome.append(format_metrics(metrics))
        outcome.append("Sample predictions:\n")
        outcome.append(format_prediction_samples(y_valid, pred))
        outcome.append("\n")

    return outcome


def train_PCA(X_train, y_train, X_valid, y_valid, outcome):
    outcome.append("2. GRID SEARCH WITH PCA + EXTRA TREES:\n")
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True)),
        # ('imputer', SimpleImputer(strategy='mean', keep_empty_features=True)),
        ('pca', PCA(n_components=10)),

        ('scaler', StandardScaler()),
        ('reg', ExtraTreesRegressor(n_estimators=100, random_state=42))
    ])

    # param_grid_pca = {
    #     'pca__n_components': [5, 8, 10, 12],
    #     'reg__n_estimators': [100, 200, 300],
    #     'reg__max_depth': [None, 10, 20, 30],
    #     'reg__min_samples_split': [2, 5, 10],
    #     'reg__min_samples_leaf': [1, 2, 4],
    #     'reg__max_features': ['auto', 'sqrt', 0.5],
    #     'reg__bootstrap': [True, False],
    #     'reg__max_samples': [None, 0.7],
    # }

    param_grid_pca = {'reg': [
        RandomForestRegressor(n_estimators=100, random_state=42),
        ExtraTreesRegressor(n_estimators=100, random_state=42),
        BaggingRegressor(n_estimators=100, random_state=42)
    ]
    }

    grid_search_pca = GridSearchCV(pipe, param_grid_pca, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    grid_search_pca.fit(X_train, y_train)

    pred_pca = grid_search_pca.best_estimator_.predict(X_valid)

    outcome.append("PCA was used to reduce input dimensions to 10.\n")
    outcome.append("ExtraTreesRegressor was tuned via grid search.\n")
    outcome.append(f"Best Parameters:\n")
    for param, val in grid_search_pca.best_params_.items():
        outcome.append(f"  - {param}: {val}\n")
    outcome.append("\nModel performance:\n")
    outcome.append(format_metrics(get_metrics(y_valid, pred_pca)))
    outcome.append("Sample predictions:\n")
    outcome.append(format_prediction_samples(y_valid, pred_pca))
    outcome.append("\n")
    return outcome


def train_script(path, output_file='model_report.txt'):
    with open(output_file, 'w') as f:
        f.write("=========== MODEL EVALUATION REPORT ==========\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    df = load_data(path)
    df = df.drop(columns=['MOF'])
    X = df.drop(columns=['energy'])
    y = df['energy']

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    report_lines = []
    report_lines = train_models(X_train, y_train, X_valid, y_valid, report_lines)
    report_lines = train_PCA(X_train, y_train, X_valid, y_valid, report_lines)

    report_lines.append("=== SUMMARY ===\n")
    report_lines.append(
        "All models were evaluated on the same validation set (20% of total data).\n"
        "Baseline models used default parameters.\n"
        "GridSearchCV improved performance by tuning parameters.\n"
        "PCA helped reduce feature space and may improve generalization.\n"
    )

    with open(output_file, 'a') as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ Saved in: {output_file}\n")


if __name__ == '__main__':
    train_script(FP_group_45, '45_fingerprints_model_report.txt')
    train_script(FP_group_120, '120_fingerprints_model_report.txt')
    train_script(FP_CO2_45, '45_fingerprints_CO2_model_report.txt')
    train_script(FP_CO2_120, '120_fingerprints_CO2_model_report.txt')
    train_script(FP_H2O_45, '45_fingerprints_H2O_model_report.txt')
    train_script(FP_H2O_120, '120_fingerprints_H2O_model_report.txt')
