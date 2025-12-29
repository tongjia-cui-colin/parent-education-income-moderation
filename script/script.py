#---------------------------------------------------------------------
# Parent Education and Income Moderation Analysis
#---------------------------------------------------------------------


#---------------------------------------------------------------------
# Packages
#---------------------------------------------------------------------

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


# -------------------------------------------------------------------
# Data Cleaning and Feature Engineering
# -------------------------------------------------------------------

class DataCleaner:
    """
    Handles data loading, cleaning, and construction of analysis variables.
    """

    def __init__(self, filepath: Path):
        self.filepath = filepath

    def load_and_clean(self) -> pd.DataFrame:
        """
        Load dataset and construct analysis variables.

        Returns
        -------
        pd.DataFrame
            Cleaned dataset with constructed predictors.
        """
        df = pd.read_csv(self.filepath)

        # Drop observations with missing parental education
        df = df.dropna(subset=["Parent_Education_Level"])

        # Construct parental advanced education indicator
        df["Edu_Advanced"] = df["Parent_Education_Level"].apply(
            lambda x: 0 if x in ["High School", "None"] else 1
        )

        # Construct ordinal income category
        income_map = {"Low": 1, "Medium": 2, "High": 3}
        df["Income_Cat"] = df["Family_Income_Level"].map(income_map)

        return df




# -------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------

class Visualization:
    """
    Visualization utilities for exploratory analysis.
    """

    @staticmethod
    def boxplot_by_education(df: pd.DataFrame):
        df.boxplot(column="Total_Score", by="Edu_Advanced")
        plt.title("Total Score by Parental Advanced Education")
        plt.suptitle("")
        plt.xlabel("Edu_Advanced (0 = No, 1 = Yes)")
        plt.ylabel("Total Score")
        plt.show()

    @staticmethod
    def boxplot_moderation(df: pd.DataFrame):
        df.boxplot(column="Total_Score", by=["Income_Cat", "Edu_Advanced"])
        plt.title("Total Score by Income × Parental Education")
        plt.suptitle("")
        plt.xlabel("Income_Cat, Edu_Advanced")
        plt.ylabel("Total Score")
        plt.show()


# -------------------------------------------------------------------
# Regression Models
# -------------------------------------------------------------------

class SLR:
    """
    Simple Linear Regression model:
    Total_Score ~ Edu_Advanced
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = None

    def fit(self):
        X = sm.add_constant(self.df["Edu_Advanced"])
        y = self.df["Total_Score"]
        self.results = sm.OLS(y, X).fit()
        return self.results

    def predict(self):
        if self.results is None:
            raise RuntimeError("SLR model must be fitted before prediction.")
        return self.results.predict()


class Moderation:
    """
    Moderation model with interaction term:
    Total_Score ~ Edu_Advanced + Income_Cat + Edu_Advanced × Income_Cat
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = None

    def fit(self):
        self.df["Interaction"] = (
            self.df["Edu_Advanced"] * self.df["Income_Cat"]
        )

        X = sm.add_constant(
            self.df[["Edu_Advanced", "Income_Cat", "Interaction"]]
        )
        y = self.df["Total_Score"]

        self.results = sm.OLS(y, X).fit()
        return self.results

    def predict(self):
        if self.results is None:
            raise RuntimeError("Moderation model must be fitted before prediction.")
        return self.results.predict()


# -------------------------------------------------------------------
# Outlier Handling (Sensitivity Analysis)
# -------------------------------------------------------------------

class OutlierHandler:
    """
    Provides optional outlier handling utilities.
    Outlier removal is treated as a sensitivity check, not default preprocessing.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def remove_iqr(self, col: str = "Total_Score", k: float = 1.5) -> pd.DataFrame:
        q1, q3 = self.df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr

        return self.df[
            (self.df[col] >= lower) & (self.df[col] <= upper)
        ].copy()


# -------------------------------------------------------------------
# Statistical Tests
# -------------------------------------------------------------------

class StatisticalTests:
    """
    Statistical inference utilities.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def welch_ttest_edu(self) -> dict:
        """
        Welch’s unequal-variance t-test comparing academic performance
        by parental education status.
        """
        group0 = self.df[self.df["Edu_Advanced"] == 0]["Total_Score"]
        group1 = self.df[self.df["Edu_Advanced"] == 1]["Total_Score"]

        t_stat, p_val = stats.ttest_ind(group1, group0, equal_var=False)

        return {
            "mean_no_advanced": group0.mean(),
            "mean_advanced": group1.mean(),
            "t_statistic": t_stat,
            "p_value": p_val,
            "n_no_advanced": len(group0),
            "n_advanced": len(group1),
        }


# -------------------------------------------------------------------
# Main Execution Pipeline
# -------------------------------------------------------------------

cleaner = DataCleaner("data/Students_Grading_Dataset_Biased.csv")
df = cleaner.load_and_clean()

model1 = SLR(df)
res1 = model1.fit()
print(res1.summary())

model2 = Moderation(df)
res2 = model2.fit()
print(res2.summary())

viz = Visualization(df)
viz.boxplot_SLR()
viz.boxplot_Mod()

out_handler = OutlierHandler(df)

# remove outliers using IQR
df_no_outliers = out_handler.remove_iqr(col="Total_Score", k=1.5)

print("Original sample size:", df.shape[0])
print("After IQR outlier removal:", df_no_outliers.shape[0])

stats_test = StatisticalTests(df)
ttest_results = stats_test.welch_ttest_edu()

print("Welch t-test results:")
for k, v in ttest_results.items():
    print(f"{k}: {v}")
