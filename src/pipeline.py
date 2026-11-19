import pandas as pd

from src.engineering import add_squares, engineer, log_transform
from src.features import clean


SQUARED_COLUMNS = ["OverallQual"]


def make_features(
    fit_raw: pd.DataFrame, df_raw: pd.DataFrame, skewed: list[str], drop_outliers: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the features of the linear regression notebook, without dropping the collinear columns.

    Everything that learns from the data (the fills, the levels, the means for the squares) is fitted on fit_raw only,
    so the same call can be used on each fold of a cross-validation, and on all of train and test at the end.

    Parameters:
    fit_raw (pd.DataFrame): The raw train houses to fit the cleaning and the encoding on.
    df_raw (pd.DataFrame): The raw houses to build features for, either the same train houses, a validation fold or test.
    skewed (list[str]): The columns to take log(1 + x) of.
    drop_outliers (bool): Whether to drop the outlier houses from df_raw. The fit houses never include them.

    Returns:
    tuple: The cleaned fit houses, the cleaned df houses, the features of the fit houses and the features of the df houses.
    """
    fit_clean, df_clean = clean(fit_raw, fit_raw), clean(df_raw, fit_raw, drop_outliers=drop_outliers)
    fit_features, df_features = engineer(fit_clean, fit_clean), engineer(df_clean, fit_clean)
    fit_features, df_features = log_transform(fit_features, skewed), log_transform(df_features, skewed)
    return (
        fit_clean,
        df_clean,
        add_squares(fit_features, fit_features, SQUARED_COLUMNS),
        add_squares(df_features, fit_features, SQUARED_COLUMNS),
    )
