# Dataset

The dataset is provided in CSV format and can be found in this folder under the file name  **`Students_Grading_Dataset_Biased.csv`**.

## Source
This project uses data from:

**Students Grading Dataset**  
Kaggle  
Author: Mahmoud Elhemaly  
Link: https://www.kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset?select=Students_Grading_Dataset_Biased.csv  

The original dataset contains student academic performance records along with demographic and socioeconomic attributes, including parental education level and family income level.

## Scope of Data Used
Only a **subset of variables** from the original dataset is used in this project. The analysis focuses on parental socioeconomic characteristics and student academic performance.

Specifically:
- Observations with missing values in `Parent_Education_Level` are removed
- Parental education and family income variables are recoded for regression analysis
- Only variables required for modeling and inference are retained

## Variables Used in This Project

### Outcome Variable
- **`Total_Score`**  
  Composite academic performance score across a semester, used as the dependent variable in all regression analyses.

### Source Variables
- **`Parent_Education_Level`**  
  Categorical variable indicating the highest level of education attained by the parent(s).

- **`Family_Income_Level`**  
  Categorical variable indicating household income level.

### Constructed Predictor Variables
- **`Edu_Advanced`**  
  Binary indicator constructed from `Parent_Education_Level` in the analysis code:
  - `0` if `Parent_Education_Level` is `"High School"` or `"None"`
  - `1` if `Parent_Education_Level` corresponds to any education level beyond high school (e.g., college degree or higher)

- **`Income_Cat`**  
  Ordinal income category constructed from `Family_Income_Level`:
  - `1` if `Family_Income_Level` is `"Low"`
  - `2` if `Family_Income_Level` is `"Medium"`
  - `3` if `Family_Income_Level` is `"High"`

### Interaction Term
- **`Edu_Advanced × Income_Cat`**  
  Interaction between parental advanced educational attainment and family income, included to test whether income moderates the association between parental education and student academic performance.

## Notes
- The dataset does **not include personally identifiable information**
- All records are anonymized student-level observations
- The dataset is observational and intended for exploratory analysis
- Users of the data should cite the original Kaggle source when referencing this dataset
