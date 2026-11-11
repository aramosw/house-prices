import pandas as pd

TRAIN_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"

# The files use "NA" for a missing value, but "None" is a real category (MasVnrType),
# which pandas would otherwise read as NaN.
NA_VALUES = ["NA"]


def load_train(data_path: str = TRAIN_DATA_PATH) -> pd.DataFrame:
    """
    Load the training data from a CSV file.

    Parameters:
    data_path (str): The path to the CSV file containing the training data.

    Returns:
    pd.DataFrame: A DataFrame containing the training data.
    """
    return pd.read_csv(data_path, index_col="Id", keep_default_na=False, na_values=NA_VALUES)


def load_test(data_path: str = TEST_DATA_PATH) -> pd.DataFrame:
    """
    Load the test data from a CSV file.

    Parameters:
    data_path (str): The path to the CSV file containing the test data.

    Returns:
    pd.DataFrame: A DataFrame containing the test data.
    """
    return pd.read_csv(data_path, index_col="Id", keep_default_na=False, na_values=NA_VALUES)
