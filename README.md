# Predict Customer Churn

This is the first project "**Predict Customer Churn**" of the ML DevOps Engineer Nanodegree by Udacity.

## Project Description
The project includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested).
Also, It introduces a problem data scientists across companies face all the time. How do we identify (and later intervene with) customers who are likely to churn?

The following diagram _(Source: Udacity)_ illustrates the sequence diagram of the project:

![image](./images/sequencediagram.jpeg)

## Files and data description
```
.
├── Guide.ipynb     # Getting started and troubleshooting tips
├── README.md       # Project READDME
├── churn_library.py        # The project's refactored library
├── churn_notebook.ipynb    # Contains the code to be refactored
├── constants.py    # All the constants used in the library
├── data
│   └── bank_data.csv   # The dataset of the project
├── images
│   ├── eda     # Statistics related to the data
│   │   ├── Churn.png
│   │   ├── Correlation_Heatmap.png
│   │   ├── Customer_Age.png
│   │   ├── Marital_Status.png
│   │   └── Total_Trans_Ct.png
│   ├── results     # The models' results
│   │   ├── Classification_Report_LR.png
│   │   ├── Classification_Report_RF.png
│   │   ├── Feature_Importance.png
│   │   ├── ROC_Curve.png
│   │   ├── ROC_Curve_Both_Models.png
│   │   └── Tree_Explainer.png
│   └── sequencediagram.jpeg    # The sequence diagram used in the README
├── logs
│   └── churn_library.log       # The logs file
├── models      # The trained models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── requirements_py3.6.txt  
├── requirements_py3.8.txt
├── test_churn_script_logging_and_tests.py      # Testing script (also logging)
```

## Running Files
1. Create a new environment `python3 -m venv mlops-p1`
2. Access the environment `source mlops-p1/bin/activate`
3. Install the requirements `pip install -r requirements_py3.XXXX.txt` _(choose the right requirements file depending on your Python version: 3.8 or 3.6)_
4. Run the library `python churn_library.py`
5. Run the testing script `python -m pytest`
_(Expected output: `test_churn_script_logging_and_tests.py .F.F.F.. `)_



