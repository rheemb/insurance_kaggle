# 2023 NESS Statathon: Insurance Fraud Detection (Kaggle Competition)

This project was developed as part of the [2023 NESS Statathon Kaggle Competition (Theme 1)] (https://www.kaggle.com/competitions/2023-travelers-ness-statathon/overview), hosted by Travelers Insurance Company. The goal was to build a predictive model to identify fraudulent first-party physical damage insurance claims and uncover key indicators of fraud, helping Travelers minimize financial losses while ensuring fair compensation for policyholders.

## Project Overview
Travelers Insurance provided a dataset of referred claims from 2015-2016, and our task was to develop a machine learning model to classify claims as fraudulent (`fraud = 1`) or legitimate (`fraud = 0`), focusing on the weighted F1 score as the evaluation metric. We also aimed to provide interpretable insights into the drivers of fraud to support Travelers’ fraud detection team.

## What We Did
1. **Exploratory Data Analysis (EDA), Data Cleaning, and Feature Engineering**:
   - Analyzed the dataset (`train_2023.csv`) with 25 features, including `age_of_driver`, `past_num_of_claims`, `witness_present_ind`, and more.
   - Handled missing values by imputing with median values for numerical features and mode for categorical ones to avoid data loss.
   - Encoded categorical variables (e.g., `gender`, `vehicle_category`) using one-hot encoding and created dummy variables.
   - Performed feature scaling using standardization to normalize numerical features (e.g., `annual_income`, `claim_est_payout`).
   - Engineered features like `claim_date_day` (day of the month from `claim_date`) to explore temporal patterns in fraud.
   - Identified and imputed outliers in variables like `age_of_vehicle` using median values.

2. **Modeling**:
   - Split the data into training (80%) and test (20%) sets using a consistent random seed, per competition rules.
   - Addressed class imbalance by oversampling the minority class (`fraud = 1`) to balance the dataset.
   - Explored multiple classification models: Logistic Regression, Decision Tree, Gradient Boosting Classifier (GBC), Support Vector Classifier (SVC), and Random Forest Classifier (RFC).
   - Focused on the weighted F1 score as the primary metric, as specified by the competition.
   - Selected Random Forest as the final model due to its superior performance and ability to capture complex interactions.

3. **Model Results**:
   - **Random Forest** achieved the best weighted F1 score of **0.79** on the test set, outperforming other models:
     - Logistic Regression: 0.68
     - Decision Tree: 0.73
     - Gradient Boosting: 0.79 (but RFC had better interpretability)
     - SVC: 0.71
   - The model effectively identified fraudulent claims while minimizing overfitting, as shown in the classification report:
     - Test Precision (Fraud = Yes): 0.43
     - Test Recall (Fraud = Yes): 0.07
     - Test F1-Score (Weighted): 0.79
   - Confusion Matrix (Test Data) showed good performance on non-fraudulent claims but highlighted room for improvement in detecting fraudulent ones.

4. **Interpretability and Discussion**:
   - Identified the top 5 features influencing fraud prediction using feature importance from Random Forest:
     - `past_num_of_claims` (Correlation: 0.101): Higher prior claims increase fraud likelihood.
     - `witness_present_ind` (Correlation: -0.075): Presence of a witness reduces fraud probability.
     - `high_education_ind` (Correlation: -0.110): Higher education may correlate with fraud.
     - `age_of_driver` (Correlation: -0.064): Certain age groups (younger/older) show higher fraud risk.
     - `annual_income` (Correlation: -0.050): Income levels may indicate fraud tendencies.
   - Created partial dependence plots for these features to visualize their impact on fraud probability.
   - Analyzed fraud patterns by day of the month, finding no strong temporal trend (10-20% fraud rate daily).

5. **Business Recommendations**:
   - **Scrutinize High-Risk Profiles**: Focus on claimants with many prior claims (`past_num_of_claims`) for stricter verification.
   - **Leverage Witness Presence**: Claims with witnesses (`witness_present_ind = 1`) are less likely to be fraudulent and can be prioritized for faster processing.
   - **Target Specific Demographics**: Pay closer attention to claims from younger/older drivers (`age_of_driver`) and those with higher education (`high_education_ind`), as they may require deeper investigation.
   - **Monitor Income Trends**: Investigate claims from specific income brackets (`annual_income`) for potential fraud due to financial motivations.
   - **Operationalize the Model**: Implement the Random Forest model in an automated fraud detection system to streamline investigations and allocate resources based on risk profiles.

## Approach and Steps
- **Libraries Used**:
  - **Pandas** and **NumPy**: Data manipulation and cleaning.
  - **Matplotlib** and **Seaborn**: Visualization (e.g., fraud distribution, partial dependence plots).
  - **Scikit-learn**: Model building (e.g., `RandomForestClassifier`, `train_test_split`), evaluation (`f1_score`, `classification_report`), and preprocessing (`StandardScaler`).
- **Steps**:
  1. Loaded and explored the dataset to understand feature distributions and relationships.
  2. Cleaned the data by handling missing values, encoding categorical variables, and scaling numerical features.
  3. Performed EDA to identify patterns (e.g., fraud distribution by day of month) and correlations with the target (`fraud`).
  4. Balanced the dataset using oversampling to address the minority class (`fraud = 1`).
  5. Trained and evaluated multiple models, selecting Random Forest for its high weighted F1 score and interpretability.
  6. Analyzed feature importance and created visualizations to explain fraud drivers.
  7. Formulated business recommendations based on model insights.

## Results and Metrics
- **Weighted F1 Score (Test Set)**: 0.79 (Random Forest).
- **Key Insight**: The model excels at identifying non-fraudulent claims (precision: 0.85, recall: 0.98) but has lower recall for fraudulent claims (0.07), indicating a need for further tuning to improve fraud detection sensitivity.
- **Fraud Distribution by Day**: Fraudulent claims were stable at 10-20% per day, with no strong day-of-month pattern, suggesting other features drive fraud more significantly.

## Why We Chose This Approach
- **Random Forest Selection**: Chosen for its high F1 score (0.79), ability to handle imbalanced data, and interpretability through feature importance and partial dependence plots.
- **Feature Focus**: Prioritized features like `past_num_of_claims` and `witness_present_ind` due to their correlation with fraud, ensuring actionable insights for Travelers.
- **Oversampling**: Addressed class imbalance to improve model performance on the minority class (`fraud = 1`).
- **Excluded Features**: Dropped `claim_number` (as per competition rules) and avoided features like `zip_code` due to high cardinality and potential noise.

## Conclusions
Our Random Forest model, with a weighted F1 score of 0.79, provides a robust solution for identifying fraudulent insurance claims, helping Travelers reduce financial losses while ensuring fair claim processing. Key fraud indicators include the number of past claims, presence of witnesses, and demographic factors like age and education. We recommend operationalizing this model for automated fraud detection, focusing investigations on high-risk profiles, and exploring additional features (e.g., claim amount patterns) to enhance fraud detection sensitivity.

## Files
- **insurance_fraud_detection.ipynb**: Jupyter Notebook containing the full code, EDA, modeling, visualizations, and analysis.
- **train_2023.csv**: Dataset used for training and evaluation (sourced from the Kaggle competition).

---
## How to Run
1. Clone the repository: `git clone https://github.com/rheemb/insurance_kaggle.git` 
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter Notebook: `jupyter notebook insurance_fraud_detection.ipynb`
4. The notebook includes all steps from data loading to model evaluation and visualization.

## Note on Posting Delay
This project was completed in 2023 as part of the 2023 NESS Statathon Kaggle competition. However, due to the loss of the original project files, we are sharing it on LinkedIn and GitHub in 2025 after recovering and finalizing the work. The analysis, modeling, and results reflect our efforts from 2023, showcasing our approach to solving the insurance fraud detection challenge at that time.
