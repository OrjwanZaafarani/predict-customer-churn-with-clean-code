'''This module tests all the functions in churn_library.py'''

import os
import pytest
import logging
from churn_library import *
from constants import *

####### Configs #######
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

for handler in logging.root.handlers[:]:
	logging.root.removeHandler(handler)

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


####### Fixtures #######
@pytest.fixture(scope="module")
def df():
	df = pd.read_csv(dataset_path)
	df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
	return df


####### Unit tests #######
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
		logging.error("Testing import_data: ERROR - The file doesn't appear to have rows and columns")
		raise err


@pytest.mark.parametrize("cols_list",
                         [eda_columns,
						 ["Churnnnnnn", "Customer_Agee", "Marital_Status", "Total_Trans_Ct"]])
def test_eda(cols_list, df):
	'''
	test perform eda function
	'''
	try:
		assert all([item in df.columns for item in cols_list])
		try:
			perform_eda(df)
			logging.info("Testing perform_eda: SUCCESS")
		except Exception as ex:
			logging.error("Testing perform_eda: ERROR - the exception was " + str(ex))
	except AssertionError as err:
		logging.error("Testing perform_eda: ERROR - not all eda columns were in the dataframe's columns")
		raise err


@pytest.mark.parametrize("categorical_columns",
                         [cat_columns,
						 ['Genderrrr', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']])
def test_encoder_helper(categorical_columns, df):
	'''
	test encoder helper
	'''
	try:
		assert all([item in df.columns for item in categorical_columns])
		try:
			encoder_helper(df, categorical_columns)
			logging.info("Testing encoder_helper: SUCCESS")
		except Exception as ex:
			logging.error("Testing encoder_helper: ERROR - the exception was " + str(ex))
	except AssertionError as err:
		logging.error("Testing encoder_helper: ERROR - not all categorical columns were in the dataframe's columns")
		raise err


# def test_perform_feature_engineering(perform_feature_engineering):
# 	'''
# 	test perform_feature_engineering
# 	'''


# def test_train_models(train_models):
# 	'''
# 	test train_models
# 	'''
# 	pass


if __name__ == "__main__":
	pass
	# test_import(import_data)
	# test_eda(perform_eda)








