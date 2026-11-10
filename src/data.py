import pandas as pd

TRAIN_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"


def load_train(data_path: str = TRAIN_DATA_PATH) -> pd.DataFrame:
    """
    Load the training data from a CSV file.

    Parameters:
    data_path (str): The path to the CSV file containing the training data.

    Returns:
    pd.DataFrame: A DataFrame containing the training data.
    """
    return pd.read_csv(data_path)


def load_test(data_path: str = TEST_DATA_PATH) -> pd.DataFrame:
    """
    Load the test data from a CSV file.

    Parameters:
    data_path (str): The path to the CSV file containing the test data.

    Returns:
    pd.DataFrame: A DataFrame containing the test data.
    """
    return pd.read_csv(data_path)
