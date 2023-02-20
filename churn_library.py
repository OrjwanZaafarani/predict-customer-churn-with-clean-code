# library doc string
'''This library predicts whether a customer will churn in clean code'''

# import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

from parameters import *


def import_data(pth):
        '''
        returns dataframe for the csv found at pth

        input:
                pth: a path to the csv
        output:
                df: pandas dataframe
        '''
        df = pd.read_csv(pth)
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        return df


def perform_eda(df):
        '''
        perform eda on df and save figures to images folder
        input:
                df: pandas dataframe

        output:
                None
        '''

        plt.figure(figsize=(20,10)) 
        for column_name in ["Churn", "Customer_Age", "Martial_Status", "Total_Trans_Ct"]:
                if (column_name == "Churn") or (column_name == "Customer_Age"):
                        df[column_name].hist()
                elif column_name == "Marital_Status":
                        df.column_name.value_counts('normalize').plot(kind='bar')
                elif column_name == "Total_Trans_Ct":
                        # distplot is deprecated. Use histplot instead
                        # sns.distplot(df['Total_Trans_Ct']);
                        # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained using a kernel density estimate
                        sns.histplot(df[column_name], stat='density', kde=True)  
                plt.savefig("images/" + column_name + ".png")
                plt.cla()
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        plt.savefig("images/Correlation_Heatmap.png")


def encoder_helper(df, category_lst):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
 
        output:
                df: pandas dataframe with new columns for
        '''
        for category in category_lst:
                df.merge(df.groupby(category).mean()["Churn"], how='left', on=category, \
                        suffixes = (None, "_" + category))
        return df


def perform_feature_engineering(df):
        '''
        input:
                df: pandas dataframe
  
        output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        '''
        X = pd.DataFrame()
        X[keep_cols] = df[keep_cols]
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
        return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest

        output:
                None
        '''

        classification_reports = []
        classification_reports.append(classification_report(y_test, y_test_preds_rf))
        classification_reports.append(classification_report(y_train, y_train_preds_rf))
        classification_reports.append(classification_report(y_test, y_test_preds_lr))
        classification_reports.append(classification_report(y_train, y_train_preds_lr))
    
        model_names_list = ["Random Forest Test", "Random Forest Train", "Logistic Regression Test", "Logistic Regression Train"]
        
        
        plt.rc('figure', figsize=(5, 5))
        x_coordinate = 0.01
        y1_coordinate = 0.6
        y2_coordinate = 0.05
        for report, model_name in zip(classification_reports, model_names_list):
                plt.text(x_coordinate, y1_coordinate, str(model_name), {'fontsize': 10}, fontproperties = 'monospace')
                plt.text(x_coordinate, y2_coordinate, str(report), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
                y1_coordinate += 0.65
                y2_coordinate += 0.65
        plt.axis('off')
        plt.savefig("images/Classification_Reports.png")
        


def feature_importance_plot(model, X_data, output_pth):
        '''
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                None
        '''
        # Calculate feature importances
        importances = model.feature_importances_

        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20,5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(X_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)

        plt.savefig(output_pth)
        

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass