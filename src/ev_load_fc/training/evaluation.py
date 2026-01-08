import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.inspection import permutation_importance
from typing import Any

class EvaluationPlots:


    def __init__(
            self, 
            X_train:pd.DataFrame, 
            y_train:pd.Series, 
            X_test:pd.DataFrame,
            y_test:pd.Series,
            model:Any,
            model_name:str, 
            run_num:int,
    ):
        """


        Args:
            X_train (pd.DataFrame): _description_
            y_train (pd.Series): _description_
            X_test (pd.DataFrame): _description_
            y_test (pd.Series): _description_
            model (Any): _description_
            model_name (str): _description_
            run (int): _description_
        """

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.run_name = f"{model_name}_{run_num}"


    def plot_correlation_with_target(self) -> plt.Figure:
        # Adapted from noteobok in https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs/
        """
        Plots the Pearson correlation of each input feature with the target.

        Returns:
            fig (plt.Figure): The matplotlib figure object.
        """

        df = self.X_train.copy()
        target_name = self.y_train.name
        df["target"] = self.y_train

        # Compute correlations between all input features and "target"
        correlations = df.corr()["target"].drop("target").sort_values()

        # Generate a color palette from red to green
        colors = sns.diverging_palette(10, 130, as_cmap=True)
        color_mapped = correlations.map(colors)

        # Set Seaborn style
        sns.set_style(
            "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
        )  # Light grey background and thicker grid lines

        # Create bar plot
        fig = plt.figure(figsize=(12, 8))
        plt.barh(correlations.index, correlations.values, color=color_mapped)

        # Set labels and title with increased font size
        plt.title(f"Correlation with {target_name}", fontsize=18)
        plt.xlabel("Correlation Coefficient", fontsize=16)
        plt.ylabel("Feature", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis="x")
        plt.tight_layout()

        # Save fig
        file_name = f"corr_{self.run_name}.png"
        save_path = self.model_plots / file_name
        plt.savefig(save_path, format="png", dpi=600)

        # prevent matplotlib from displaying the chart every time we call this function
        plt.close(fig)

        return fig, save_path


    def plot_feature_importance(self) -> plt.Figure:
        # Adapted from noteobok in https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs/
        """
        Plots the permutation importances of each input feature using the given model.

        Returns:
            fig (plt.Figure): The matplotlib figure object.
        """

        pi_set = permutation_importance(
            estimator=self.model,
            X=self.X_train,
            y=self.y_train,
            n_repeats=5,
            scoring="neg_root_mean_squared_error",
            random_state=42
        )

        pi_mean = pd.Series(data=pi_set.importances_mean, index=self.X_train.columns)

        # Generate a color palette from red to green
        colors = sns.diverging_palette(10, 130, as_cmap=True)
        color_mapped = pi_mean.map(colors)

        # Set Seaborn style
        sns.set_style(
            "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
        )  # Light grey background and thicker grid lines

        # Create bar plot
        fig = plt.figure(figsize=(12, 8))
        plt.barh(pi_mean.index, pi_mean.values, color=color_mapped)

        # Set labels and title with increased font size
        plt.title(f"Permutation importances of features predicting EV energy load", fontsize=18)
        plt.xlabel("Permutation Importance", fontsize=16)
        plt.ylabel("Feature", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis="x")
        plt.tight_layout()

        # Save fig
        file_name = f"feat_imp_{self.run_name}.png"
        save_path = self.model_plots / file_name
        plt.savefig(save_path, format="png", dpi=600)

        # prevent matplotlib from displaying the chart every time we call this function
        plt.close(fig)

        return fig


    def plot_residuals(self) -> plt.Figure:
        # Adapted from noteobok in https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs/
        """
        Plots the residuals of the model's predictions against the test set.

        Returns:
            fig (plt.Figure): The matplotlib figure object.
        """

        # Predict using the model
        preds = self.model.predict(self.X_test)

        # Calculate residuals
        residuals = self.y_test - preds

        # Set Seaborn style
        sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

        # Create scatter plot
        fig = plt.figure(figsize=(12, 8))
        plt.scatter(self.y_test, residuals, color="blue", alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="-")

        # Set labels, title and other plot properties
        plt.title("Residuals vs True Values", fontsize=18)
        plt.xlabel("True Values", fontsize=16)
        plt.ylabel("Residuals", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis="y")
        plt.tight_layout()

        # Save fig
        file_name = f"resid_{self.run_name}.png"
        save_path = self.model_plots / file_name
        plt.savefig(save_path, format="png", dpi=600)

        # Show the plot
        plt.close(fig)



