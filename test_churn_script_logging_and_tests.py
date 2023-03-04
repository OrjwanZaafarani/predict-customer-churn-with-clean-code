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
@pytest.fixture(scope="module", name="load_df")
def fixture_load_df():
    '''Fixture for loading the dataframe and adding the Churn column'''
    data_frame = pd.read_csv(dataset_path)
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data_frame


@pytest.fixture(scope="module", name="encoder")
def fixture_encoder(load_df):
    '''Fixture for encoding the categorical columns'''
    return encoder_helper(load_df, cat_columns)


@pytest.fixture(scope="module", name="x_features_train")
def fixture_x_features_train(encoder):
    '''Fixture for retrieving x_train'''
    x_train, _, _, _ = perform_feature_engineering(encoder)
    return x_train


@pytest.fixture(scope="module", name="x_features_test")
def fixture_x_features_test(encoder):
    '''Fixture for retrieving x_test'''
    _, x_test, _, _ = perform_feature_engineering(encoder)
    return x_test


@pytest.fixture(scope="module", name="y_vector_train")
def fixture_y_vector_train(encoder):
    '''Fixture for retrieving y_train'''
    _, _, y_train, _ = perform_feature_engineering(encoder)
    return y_train


@pytest.fixture(scope="module", name="y_vector_test")
def fixture_y_vector_test(encoder):
    '''Fixture for retrieving y_test'''
    _, _, _, y_test = perform_feature_engineering(encoder)
    return y_test

##################### Unit tests #####################


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''

    try:
        data_frame = import_data(dataset_path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: ERROR - The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: ERROR - The file doesn't appear to have rows and columns")
        raise err


def test_eda(load_df):
    '''
    test perform eda function
    '''
    try:
        assert all(item in load_df.columns for item in eda_columns)
        perform_eda(load_df)
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: ERROR - not all eda columns were in the dataframe's columns")
        raise err


def test_encoder_helper(load_df):
    '''
    test encoder helper
    '''
    try:
        assert all(item in load_df.columns for item in cat_columns)
        _ = encoder_helper(load_df, cat_columns)
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
        assert all(item in encoder.columns for item in keep_cols)
        _, _, _, _ = perform_feature_engineering(encoder)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: ERROR - not all chosen columns \
                were in the dataframe's columns")
        raise err


def test_train_models(x_features_train, x_features_test, y_vector_train, y_vector_test):
    '''
    test train_models
    '''
    try:
        assert x_features_train.shape[0] > 0
        assert x_features_test.shape[0] > 0
        assert y_vector_train.shape[0] > 0
        assert y_vector_test.shape[0] > 0
        _, _ = train_models(x_features_train, x_features_test, y_vector_train, y_vector_test)
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as assertion_error:
        logging.error(
            "Testing train_models: ERROR - the training or testing datasets'\
                 shapes are equal to zero")
        raise assertion_error
        