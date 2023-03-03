# Lists
# ------------
keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Churn_Gender', 'Churn_Education_Level', 'Churn_Marital_Status', 
        'Churn_Income_Category', 'Churn_Card_Category']

eda_columns = ["Churn", "Customer_Age", "Marital_Status", "Total_Trans_Ct"]

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]

# Paths
# ------------

# Images
dataset_path = "data/bank_data.csv"
images_results_path = "images/results/"
images_eda_path = "images/eda/"
feature_importance_path = images_results_path + "Feature_Importance.png"
correlation_heatmap_path = images_eda_path + "Correlation_Heatmap.png"
classification_report_rf_path = images_results_path + "Classification_Report_RF.png"
classification_report_lr_path = images_results_path + "Classification_Report_LR.png"
roc_curve_path = images_results_path + "ROC_Curve.png"
roc_curve_both_models_path = images_results_path + "ROC_Curve_Both_Models.png"
tree_explainer_path = images_results_path + "Tree_Explainer.png"
# Models
rfc_model_path = 'models/rfc_model.pkl'
lrc_model_path = 'models/logistic_model.pkl'

