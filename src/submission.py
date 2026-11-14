import pandas as pd


SAMPLE_PATH = "../data/sample_submission.csv"
SUBMISSIONS_FOLDER = "../submissions"


def save_submission(ids: pd.Index, prices, model: str, version: int) -> str:
    """
    Check the predictions against the sample file, and save them as {model}_v{version}.csv, for example ols_v01.csv.

    Parameters:
    ids (pd.Index): The Id of each test house, in the order of the sample file.
    prices (array-like): The predicted SalePrice of each house, as a price and not a log price.
    model (str): The name of the model, for example "ols".
    version (int): The version of the model, which is increased with each change.

    Returns:
    str: The path of the saved file.
    """
    submission = pd.DataFrame({"Id": ids, "SalePrice": prices})

    sample = pd.read_csv(SAMPLE_PATH)
    if not submission["Id"].equals(sample["Id"]):
        raise ValueError("The Ids are not the same as in the sample file, or not in the same order")
    if submission["SalePrice"].isnull().any() or (submission["SalePrice"] <= 0).any():
        raise ValueError("Every price must be present and positive")

    path = f"{SUBMISSIONS_FOLDER}/{model}_v{version:02d}.csv"
    submission.to_csv(path, index=False)
    return path
