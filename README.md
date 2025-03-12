# Oil Pipeline Leakage Prediction and Analysis

**Project Overview**

This project aims to analyze historical oil pipeline leakage data and develop machine-learning models to predict:

1.  **Leak Size (Volume):** Predicting the amount of unintentional oil release (in barrels).
2.  **Leak Cause:** Classifying the cause of the leak into predefined categories.
3.  **Leak Costs:** Predicting the total financial costs associated with a leak.

These predictions can assist in risk assessment, resource allocation, and ultimately, help prevent or mitigate the impact of future pipeline incidents.

**Data Source**

The data used in this project is a collection of oil pipeline leak/spill reports submitted to the Pipeline and Hazardous Materials Safety Administration (PHMSA) since 2010.  The dataset is available in the `database.csv` file.

*   **Source:** Pipeline and Hazardous Materials Safety Administration (PHMSA)
*   **Time Period:** 2010 - Present
*   **File:** `database.csv`

**Workflow**

The project follows a structured workflow that includes data exploration, preprocessing, feature engineering, model building, and evaluation:

1.  **Data Acquisition and Loading:**
    *   The `database.csv` file is loaded using Pandas.  Error handling is included to ensure the file exists.

2.  **Exploratory Data Analysis (EDA):**
    *   Initial data inspection using `df.head()`, `df.info()`, and `df.isnull().sum()`.
    *   Visualizations to understand the data distribution and relationships:
        *   Accidents by Operator (barplot).
        *   Accidents by Location and Pipeline Type (countplot).
        *   Accidents by Pipeline Type (countplot).
        *   Cost by Pipeline Type (barplot).
        *   Geographical Distribution of Accidents (Folium heatmap).
        *   Cause Subcategory Analysis (Plotly bar chart).

3.  **Data Preprocessing:**
    *   **Handling Missing Values:**
        *   Columns with >50% missing values are dropped.
        *   `Shutdown Date/Time` and `Restart Date/Time` are filled with a future date ('2099-01-01') to represent "not applicable."
        *   Categorical missing values are filled using `SimpleImputer` with the 'most_frequent' strategy.
        *   Numerical missing values are filled using `IterativeImputer` with a `RandomForestRegressor` estimator.
    *   **Feature Engineering:**
        *   `Accident Date/Time` is converted to datetime and used to extract year, month, day, hour, and day of the week.
        *   Distance from the mean latitude/longitude is calculated (`Lat_Lon_Distance`).
        *   An interaction feature (`Net_Loss_Ignition`) is created.
    *   **Data Type Conversion:** Categorical features are converted to numerical representations using `LabelEncoder`.
    * **Winsorization and Log Transformation:** To address outliers, used winsorization to handle outliers and applied a log transformation.
    *   **Standardization:** Numerical features are standardized using `StandardScaler`.

4.  **Model Building and Evaluation:**
    *   **Models:** A variety of regression and classification models are trained and evaluated:
        *   **Regression (Leak Size and Costs):**
            *   `RandomForestRegressor`
            *   `GradientBoostingRegressor`
            *   `SVR`
            *   `KNeighborsRegressor`
            *   `DecisionTreeRegressor`
            *   `XGBRegressor`
            *   `HistGradientBoostingRegressor`
            *   `CatBoostRegressor`
        *   **Classification (Leak Cause):**
            *   `RandomForestClassifier`
            *   `GradientBoostingClassifier`
            *   `LogisticRegression`
            *    `SVC`
            *    `KNeighborsClassifier`
            *   `DecisionTreeClassifier`
            *   `GaussianNB`
        
    *   **Evaluation Metrics:**
        *   **Regression:** Root Mean Squared Error (RMSE) and R-squared (R2)
        *   **Classification:** Accuracy, Classification Report (precision, recall, F1-score), and Confusion Matrix.  Stratified K-Fold cross-validation is used for the classification models..

**Key Findings and Results:**

*   **Leak Size Prediction:**  GradientBoostingRegressor and DecisionTreeRegressor models performed best, with R-squared values of 0.90 and 0.89, respectively.  SVR performed poorly.
*   **Leak Cause Prediction:** The GradientBoostingClassifier achieved perfect accuracy (1.00) on the test set, while DecisionTreeClassifier also shows robust performance with a mean cross-validation accuracy of 0.96.  Logistic Regression, SVC, and GaussianNB performed relatively poorly.
*   **Leak Costs Prediction:** The CatBoostRegressor and HistGradientBoostingRegressor models achieved the best performance, both with an R-squared value of 0.97.
*   Overall, ensemble methods (Random Forest, Gradient Boosting, HistGradientBoosting, CatBoost) and DecisionTree generally outperformed other model types across all prediction tasks.

**Data Dictionary (database.csv):**

| Column Name                         | Description                                                                  | Data Type  |
| :---------------------------------- | :--------------------------------------------------------------------------- | :--------- |
| Report Number                       | Unique identifier for the incident report.                                   | int        |
| Supplemental Number                 |                                                                              | int        |
| Accident Year                       | Year of the accident.                                                        | int        |
| Accident Date/Time                  | Date and time of the accident.                                               | datetime   |
| Operator ID                         | Unique identifier for the pipeline operator.                                | int        |
| Operator Name                       | Name of the pipeline operator.                                               | object     |
| Pipeline/Facility Name              | Name of the pipeline or facility involved.                                  | object     |
| Pipeline Location                   | Location of the pipeline (Onshore, Offshore, etc.).                           | object     |
| Pipeline Type                       | Type of pipeline.                                                            | object     |
| Liquid Type                         | Type of hazardous liquid being transported.                                  | object     |
| Liquid Subtype                      | More specific subtype of the hazardous liquid.                               | object     |
| Liquid Name                         | Name of the specific liquid.                                                | object     |
| Accident City                       | City where the accident occurred.                                            | object     |
| Accident County                     | County where the accident occurred.                                          | object     |
| Accident State                      | State where the accident occurred.                                           | object     |
| Accident Latitude                   | Latitude of the accident location.                                           | float      |
| Accident Longitude                  | Longitude of the accident location.                                          | float      |
| Cause Category                      | Broad category of the accident cause.                                      | object     |
| Cause Subcategory                   | More specific subcategory of the accident cause.                             | object     |
| Unintentional Release (Barrels)     | Amount of hazardous liquid unintentionally released (in barrels).             | float      |
| Intentional Release (Barrels)       | Amount of hazardous liquid intentionally released (in barrels).               | float      |
| Liquid Recovery (Barrels)           | Amount of released liquid recovered (in barrels).                            | float      |
| Net Loss (Barrels)                  | Net loss of liquid (Unintentional Release - Liquid Recovery).              | float      |
| Liquid Ignition                     | Whether the released liquid ignited (YES/NO).                               | object     |
| Liquid Explosion                    | Whether the released liquid exploded (YES/NO).                              | object     |
| Pipeline Shutdown                   | Whether the pipeline was shut down due to the accident (YES/NO).             | object     |
| Shutdown Date/Time                  | Date and time of pipeline shutdown.                                         | datetime   |
| Restart Date/Time                   | Date and time of pipeline restart.                                          | datetime   |
| Public Evacuations                  | Number of people evacuated due to the accident.                             | float      |
| Operator Employee Injuries          | Number of operator employees injured.                                         | float      |
| Operator Contractor Injuries        | Number of operator contractors injured.                                       | float      |
| Emergency Responder Injuries        | Number of emergency responders injured.                                       | float      |
| Other Injuries                      | Number of other individuals injured.                                        | float      |
| Public Injuries                     | Number of members of the public injured.                                    | float      |
| All Injuries                        | Total number of injuries.                                                    | float      |
| Operator Employee Fatalities        | Number of operator employees killed.                                         | float      |
| Operator Contractor Fatalities      | Number of operator contractors killed.                                       | float      |
| Emergency Responder Fatalities      | Number of emergency responders killed.                                       | float      |
| Other Fatalities                    | Number of other individuals killed.                                        | float      |
| Public Fatalities                   | Number of members of the public killed.                                    | float      |
| All Fatalities                      | Total number of fatalities.                                                  | float      |
| Property Damage Costs               | Costs associated with property damage.                                      | float      |
| Lost Commodity Costs                | Costs of the lost hazardous liquid.                                          | float      |
| Public/Private Property Damage Costs| Costs associated with damage to public/private property.                     | float      |
| Emergency Response Costs            | Costs of emergency response efforts.                                         | float      |
| Environmental Remediation Costs     | Costs of environmental cleanup and remediation.                              | float      |
| Other Costs                         | Any other costs associated with the incident.                                 | float      |
| All Costs                           | Total costs associated with the incident.                                  | int        |
| Accident_Year                     | Extracted Year from 'Accident Date/Time'                                        | int        |
| Accident_Month                        | Extracted Month from 'Accident Date/Time'                                                        | int   |
| Accident_Day               | Extracted Day from 'Accident Date/Time'                                           | int   |
| Accident_Hour                         |Extracted Hour from 'Accident Date/Time'                                     | int   |
| Accident_DayOfWeek              | Extracted Day of Week from 'Accident Date/Time'                       | int      |
| Lat_Lon_Distance                 |  Calculated distance                                  | float      |
| Net_Loss_Ignition                     | Interaction Feature.                             | float  |

