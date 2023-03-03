'''
This module tests all the functions in churn_library.py

Author: Orjuwan Zaafarani
Date: 3/3/2023
'''

import os
import logging
import pytest
import pandas as pd
from churn_library import import_data, perform_eda, encoder_helper, \
    perform_feature_engineering, train_models
from constants import keep_cols, eda_columns, cat_columns, dataset_path

##################### Configs #####################
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


##################### Fixtures #####################
@pytest.fixture(scope="module")
def load_df():
    '''Fixture for loading the dataframe and adding the Churn column'''
    data_frame = pd.read_csv(dataset_path)
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data_frame


@pytest.fixture(scope="module")
def encoder(load_df):
    '''Fixture for encoding the categorical columns'''
    return encoder_helper(load_df, cat_columns)


@pytest.fixture(scope="module")
def X_train(encoder):
    '''Fixture for retrieving X_train'''
    Xtrain, _, _, _ = perform_feature_engineering(encoder)
    return Xtrain


@pytest.fixture(scope="module")
def X_test(encoder):
    '''Fixture for retrieving X_test'''
    _, Xtest, _, _ = perform_feature_engineering(encoder)
    return Xtest


@pytest.fixture(scope="module")
def y_train(encoder):
    '''Fixture for retrieving y_train'''
    _, _, ytrain, _ = perform_feature_engineering(encoder)
    return ytrain


@pytest.fixture(scope="module")
def y_test(encoder):
    '''Fixture for retrieving y_test'''
    _, _, _, ytest = perform_feature_engineering(encoder)
    return ytest

##################### Unit tests #####################


@pytest.mark.parametrize("filename",
                         [dataset_path,
                          "data/no_file.csv"])
def test_import(filename):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''

    try:
        df = import_data(filename)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: ERROR - The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: ERROR - The file doesn't appear to have rows and columns")
        raise err


@pytest.mark.parametrize("cols_list",
                         [eda_columns,
                          ["Churnnnnnn",
                           "Customer_Agee",
                           "Marital_Status",
                           "Total_Trans_Ct"]])
def test_eda(cols_list, load_df):
    '''
    test perform eda function
    '''
    try:
        assert all([item in load_df.columns for item in cols_list])
        perform_eda(load_df)
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: ERROR - not all eda columns were in the dataframe's columns")
        raise err


@pytest.mark.parametrize("categorical_columns",
                         [cat_columns,
                          ['Genderrrr',
                           'Education_Level',
                           'Marital_Status',
                           'Income_Category',
                           'Card_Category']])
def test_encoder_helper(categorical_columns, load_df):
    '''
    test encoder helper
    '''
    try:
        assert all([item in load_df.columns for item in categorical_columns])
        _ = encoder_helper(load_df, categorical_columns)
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: ERROR - not all categorical columns were \
                in the dataframe's columns")
        raise err


def test_perform_feature_engineering(encoder):
    '''
    test perform_feature_engineering
    '''
    try:
        assert all([item in encoder.columns for item in keep_cols])
        _, _, _, _ = perform_feature_engineering(encoder)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: ERROR - not all chosen columns \
                were in the dataframe's columns")
        raise err


def test_train_models(X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
    except AssertionError as assertion_error:
        logging.error(
            "Testing train_models: ERROR - the training or testing datasets'\
                 shapes are equal to zero")
        raise assertion_error

    try:
        _, _ = train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except Exception as error_message:
        logging.error(
            "Testing train_models: ERROR - The exception was " +
            str(error_message))
        raise error_message
