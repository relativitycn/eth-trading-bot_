"""
Model Training Module
Implements LightGBM training with grid search and cross-validation
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import config
import os


class ModelTrainer:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_importance = None

    def split_data(self, df, feature_cols):
        """
        Split data into train/test/OOT chronologically

        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names

        Returns:
            X_train, X_test, X_oot, y_train, y_test, y_oot, train_idx, test_idx, oot_idx
        """
        n = len(df)
        train_size = int(n * config.TRAIN_RATIO)
        test_size = int(n * config.TEST_RATIO)

        # Split indices
        train_idx = df.index[:train_size]
        test_idx = df.index[train_size:train_size + test_size]
        oot_idx = df.index[train_size + test_size:]

        # Split data
        X_train = df.loc[train_idx, feature_cols]
        X_test = df.loc[test_idx, feature_cols]
        X_oot = df.loc[oot_idx, feature_cols]

        y_train = df.loc[train_idx, 'target']
        y_test = df.loc[test_idx, 'target']
        y_oot = df.loc[oot_idx, 'target']

        print("\n=== Data Split ===")
        print(f"Train: {len(X_train)} samples ({train_idx[0]} to {train_idx[-1]})")
        print(f"Test:  {len(X_test)} samples ({test_idx[0]} to {test_idx[-1]})")
        print(f"OOT:   {len(X_oot)} samples ({oot_idx[0]} to {oot_idx[-1]})")

        return X_train, X_test, X_oot, y_train, y_test, y_oot, train_idx, test_idx, oot_idx

    def train_with_grid_search(self, X_train, y_train, param_grid=None):
        """
        Train LightGBM with grid search using TimeSeriesSplit cross-validation

        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Grid search parameters (default from config)

        Returns:
            Best model
        """
        if param_grid is None:
            param_grid = config.LGBM_PARAM_GRID

        print("\n=== Grid Search ===")
        print(f"Parameter grid: {param_grid}")

        # Base model
        base_model = lgb.LGBMRegressor(
            objective='regression',
            random_state=config.RANDOM_STATE,
            n_jobs=config.N_JOBS,
            verbose=-1
        )

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        # Grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=config.N_JOBS,
            verbose=2
        )

        print("Starting grid search...")
        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        print("\n=== Best Parameters ===")
        for param, value in self.best_params.items():
            print(f"{param}: {value}")

        print(f"\nBest CV Score (MSE): {-grid_search.best_score_:.6f}")

        return self.model

    def train_simple(self, X_train, y_train, params=None):
        """
        Train LightGBM with fixed parameters (faster for testing)

        Args:
            X_train: Training features
            y_train: Training target
            params: Model parameters

        Returns:
            Trained model
        """
        if params is None:
            params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 5,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': config.RANDOM_STATE,
                'n_jobs': config.N_JOBS,
                'verbose': -1
            }

        print("\n=== Training LightGBM ===")
        print(f"Parameters: {params}")

        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(X_train, y_train)

        return self.model

    def evaluate(self, X, y, dataset_name="Dataset"):
        """
        Evaluate model on given dataset

        Args:
            X: Features
            y: True target values
            dataset_name: Name for printing

        Returns:
            Dictionary of metrics
        """
        y_pred = self.model.predict(X)

        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Directional accuracy (for signals)
        # Consider prediction > 0.5 as uptrend, < 0.5 as downtrend
        y_direction = (y > 0.5).astype(int)
        pred_direction = (y_pred > 0.5).astype(int)
        directional_accuracy = (y_direction == pred_direction).mean()

        print(f"\n=== {dataset_name} Evaluation ===")
        print(f"MSE:  {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE:  {mae:.6f}")
        print(f"R²:   {r2:.6f}")
        print(f"Directional Accuracy: {directional_accuracy:.4f}")

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'predictions': y_pred
        }

    def plot_feature_importance(self, feature_cols, top_n=30, save_path='outputs/feature_importance.png'):
        """
        Plot feature importance

        Args:
            feature_cols: List of feature names
            top_n: Number of top features to show
            save_path: Path to save plot
        """
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

        self.feature_importance = feature_importance

        # Plot top N
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(top_n), x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nFeature importance plot saved to {save_path}")

        print(f"\nTop {top_n} features:")
        for idx, row in feature_importance.head(top_n).iterrows():
            print(f"  {row['feature']}: {row['importance']:.2f}")

        return feature_importance

    def plot_predictions(self, y_true, y_pred, dataset_name="Dataset", save_path='outputs/predictions.png'):
        """
        Plot predicted vs actual values

        Args:
            y_true: True target values
            y_pred: Predicted values
            dataset_name: Name for title
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=1)
        axes[0, 0].plot([0, 1], [0, 1], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Target')
        axes[0, 0].set_ylabel('Predicted Target')
        axes[0, 0].set_title(f'{dataset_name}: Predicted vs True')
        axes[0, 0].grid(True, alpha=0.3)

        # Residuals
        residuals = y_pred - y_true
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=1)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted Target')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)

        # Distribution of predictions
        axes[1, 0].hist(y_true, bins=50, alpha=0.5, label='True', density=True)
        axes[1, 0].hist(y_pred, bins=50, alpha=0.5, label='Predicted', density=True)
        axes[1, 0].set_xlabel('Target Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Distribution Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Time series comparison (first 500 samples)
        n_samples = min(500, len(y_true))
        axes[1, 1].plot(range(n_samples), y_true[:n_samples], label='True', alpha=0.7)
        axes[1, 1].plot(range(n_samples), y_pred[:n_samples], label='Predicted', alpha=0.7)
        axes[1, 1].axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Long Threshold')
        axes[1, 1].axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='Short Threshold')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Target Value')
        axes[1, 1].set_title(f'Time Series (first {n_samples} samples)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Prediction plots saved to {save_path}")

    def save_model(self, path='models/lightgbm_model.pkl'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"\nModel saved to {path}")

    def load_model(self, path='models/lightgbm_model.pkl'):
        """Load trained model"""
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
        return self.model


if __name__ == "__main__":
    # Test model training
    from feature_engineering import get_feature_columns

    # Load data
    df = pd.read_csv('data/data_with_features.csv', index_col=0, parse_dates=True)

    # Get feature columns
    feature_cols = get_feature_columns(df)

    # Initialize trainer
    trainer = ModelTrainer()

    # Split data
    X_train, X_test, X_oot, y_train, y_test, y_oot, _, _, _ = trainer.split_data(df, feature_cols)

    # Train model (simple version for testing)
    trainer.train_simple(X_train, y_train)

    # Evaluate
    train_metrics = trainer.evaluate(X_train, y_train, "Train")
    test_metrics = trainer.evaluate(X_test, y_test, "Test")
    oot_metrics = trainer.evaluate(X_oot, y_oot, "OOT")

    # Feature importance
    os.makedirs('outputs', exist_ok=True)
    trainer.plot_feature_importance(feature_cols)

    # Prediction plots
    trainer.plot_predictions(y_oot, oot_metrics['predictions'], "OOT")

    # Save model
    trainer.save_model()
