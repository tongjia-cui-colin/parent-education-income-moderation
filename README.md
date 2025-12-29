# Parent Education and Income Moderation

## Overview
This project implements an end-to-end data science workflow to evaluate whether parental educational attainment holds explanatory power to student academic performance and whether family income moderates this relationship.

Using a dataset of approximately 5,000 students taken from Kaggle, this project uses a modular Python pipeline covering data cleaning, feature engineering, regression modeling, interaction analysis, visualization, and statistical inference. The project emphasizes reproducibility, clean software design, and responsible interpretation of results.

## Methods
- Ordinary Least Squares (OLS) regression
- Moderation analysis using interaction terms
- Welch’s unequal-variance t-test for group comparisons
- Exploratory visualization for distributional assessment
- Sensitivity checks across model specifications

## Key Findings
- Parental educational attainment shows minimal explanatory power when used as a standalone
  predictor of student academic performance.
- Family income does not meaningfully moderate the relationship between parental education
  and academic outcomes.
  - Socioeconomic predictors explain only a very small fraction of the variance in total
    academic scores (R² < 0.01).
  - Distributions of academic performance overlap substantially across education and income
    categories.
- Diagnostic inspection suggests that linear regression models with limited predictors are
  insufficient for capturing the complexity of academic performance outcomes.
- Results indicate that commonly assumed socioeconomic indicators may require augmentation
  with behavioral, institutional, or contextual variables to achieve meaningful explanatory
  power.

## Limitations
This analysis relies on observational, self-reported socioeconomic data and a composite academic performance measure. The absence of institutional, behavioral, and instructional variables limits explanatory capacity. Findings should be interpreted as exploratory rather than predictive.

## Collaboration Note

This project was completed as a team effort. Within the collaboration, I was responsible for the core analytical and programming components of the project.

My contributions include:
- Implementing the full Python analysis pipeline
- Designing and coding the regression and moderation models
- Proposing and implementing dummy coding and categorical variable transformations
  (e.g., construction of `Edu_Advanced` and `Income_Cat`)
- Structuring the workflow to emphasize model validity, diagnostics, and interpretability

Outlier handling and statistical significance testing were added subsequently by collaborators as extensions to the original pipeline. All remaining code reflects my direct technical contributions.

## Tools
Python; pandas, numpy, statsmodels, scipy, matplotlib
