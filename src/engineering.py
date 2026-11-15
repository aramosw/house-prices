import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

from src.features import EXTERIOR_SPELLING


OTHER = "Other"
MIN_LEVEL_COUNT = 10

BATHROOM_COLUMNS = ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
YEAR_COLUMNS = ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]
SUM_PARTS = ["1stFlrSF", "BsmtUnfSF"]
REPEATED_COLUMNS = ["HasGarage", "Exterior2nd"]
REPLACED_COLUMNS = BATHROOM_COLUMNS + YEAR_COLUMNS + SUM_PARTS + REPEATED_COLUMNS

DUPLICATE_COLUMNS = ["BldgType_Duplex"]

NEAR_DUPLICATE_COLUMNS = ["BldgType_2fmCon", "SaleCondition_Partial", "GarageCond", "PoolQC"]
OVERLAPPING_PREFIXES = ["MSSubClass_"]

CONTINUOUS_COLUMNS = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "TotalBsmtSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",
    "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]
SKEW_THRESHOLD = 1
MAX_ZERO_SHARE = 0.5

QUALITY_COLUMN = "OverallQual"
SIZE_COLUMN = "LogGrLivArea"
LOW_QUALITY_MAX = 3

MIN_DISTINCT_VALUES = 6

NEIGHBOURHOOD_PREFIX = "Neighborhood_"
SMOOTHING = 10
ENCODING_FOLDS = 5
ENCODING_SEED = 0

CODE_CATEGORIES = ["MSSubClass", "MoSold"]

QUALITY = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUALITY_COLUMNS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]
ORDERED_LEVELS = {column: QUALITY for column in QUALITY_COLUMNS} | {
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "LandSlope": {"Gtl": 0, "Mod": 1, "Sev": 2},
    "CentralAir": {"N": 0, "Y": 1},
    "Street": {"Grvl": 0, "Pave": 1},
}


def combine_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace columns that repeat or sum other columns with a combination, so the columns stay independent.

    Parameters:
    df (pd.DataFrame): The cleaned data, either train or test.

    Returns:
    pd.DataFrame: A copy of df with TotalBathrooms, HouseAge, RemodelAge, GarageAge and HasSecondExterior added,
    and the columns they replace dropped.
    """
    df = df.copy()
    df["TotalBathrooms"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodelAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    df["HasSecondExterior"] = (df["Exterior1st"].replace(EXTERIOR_SPELLING) != df["Exterior2nd"]).astype(int)
    return df.drop(columns=REPLACED_COLUMNS)


def encode_ordered(df: pd.DataFrame) -> pd.DataFrame:
    """Code the columns with a natural order, or two levels, as integers. A level that is not listed raises an error."""
    df = df.copy()
    for column, levels in ORDERED_LEVELS.items():
        df[column] = df[column].map(levels)
        if df[column].isnull().any():
            raise ValueError(f"{column} has a level that is not in its scale")
    return df


def one_hot(df: pd.DataFrame, train: pd.DataFrame, columns: list[str], min_count: int = MIN_LEVEL_COUNT) -> pd.DataFrame:
    """
    One-hot encode columns, with the levels taken from train only.

    The most common train level is the reference and gets no column, so the dummies cannot add up to the intercept.
    Levels with fewer than min_count train houses, and levels not in train, are grouped as Other.

    Parameters:
    df (pd.DataFrame): The data to encode, either train or test.
    train (pd.DataFrame): The training data, used to find the levels.
    columns (list[str]): The columns to encode.
    min_count (int): The fewest train houses a level needs to keep its own column.

    Returns:
    pd.DataFrame: A copy of df with each column replaced by its dummy columns.
    """
    encoded = {}
    for column in columns:
        counts = train[column].astype(str).value_counts()
        common = counts[counts >= min_count].index
        reference = common[0]
        grouped = df[column].astype(str).where(df[column].astype(str).isin(common), OTHER)

        levels = [level for level in common if level != reference]
        if len(common) < len(counts):
            levels.append(OTHER)
        for level in levels:
            encoded[f"{column}_{level}"] = (grouped == level).astype(int)
    return pd.concat([df.drop(columns=columns), pd.DataFrame(encoded, index=df.index)], axis=1)


def engineer(
    df: pd.DataFrame, train: pd.DataFrame, min_count: int = MIN_LEVEL_COUNT, drop_duplicates: bool = True
) -> pd.DataFrame:
    """
    Turn the cleaned data into numeric columns that are linearly independent.

    Combines columns, codes the ordered columns as integers, and one-hot encodes the rest. The levels come from
    train only, so the same call encodes train and test without using any test information. The target is not included.

    Parameters:
    df (pd.DataFrame): The cleaned data, either train or test.
    train (pd.DataFrame): The cleaned training data, used to find the levels.
    min_count (int): The fewest train houses a level needs to keep its own column.
    drop_duplicates (bool): Whether to drop the dummy columns in DUPLICATE_COLUMNS.

    Returns:
    pd.DataFrame: The numeric feature columns of df, without SalePrice.
    """
    df = encode_ordered(combine_columns(df.drop(columns="SalePrice", errors="ignore")))
    train = encode_ordered(combine_columns(train.drop(columns="SalePrice", errors="ignore")))

    for data in (df, train):
        data[CODE_CATEGORIES] = data[CODE_CATEGORIES].astype(str)
    nominal = [column for column in train.columns if not pd.api.types.is_numeric_dtype(train[column])]
    features = one_hot(df, train, nominal, min_count)
    return features.drop(columns=DUPLICATE_COLUMNS) if drop_duplicates else features


def drop_collinear(features: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the columns that are nearly copies of another column, and the MSSubClass dummies, which BldgType,
    HouseStyle and the age of the house largely determine.

    Parameters:
    features (pd.DataFrame): The output of engineer, either train or test.

    Returns:
    pd.DataFrame: A copy of features without those columns.
    """
    overlapping = [column for column in features.columns if column.startswith(tuple(OVERLAPPING_PREFIXES))]
    return features.drop(columns=NEAR_DUPLICATE_COLUMNS + overlapping)


def find_skewed(train_features: pd.DataFrame) -> list[str]:
    """
    Find the continuous columns worth a log transform: a skew above SKEW_THRESHOLD, and at most MAX_ZERO_SHARE zeros.

    A column that is mostly zeros has a spike at 0 that a log does not remove. Found on train only, then applied to both.

    Parameters:
    train_features (pd.DataFrame): The train features.

    Returns:
    list[str]: The names of the columns to transform.
    """
    columns = train_features[CONTINUOUS_COLUMNS]
    return [column for column in CONTINUOUS_COLUMNS if stats.skew(columns[column]) > SKEW_THRESHOLD and (columns[column] == 0).mean() <= MAX_ZERO_SHARE]


def log_transform(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Replace each column with log(1 + x), renamed with a Log prefix.

    Parameters:
    features (pd.DataFrame): The features, either train or test.
    columns (list[str]): The columns to transform.

    Returns:
    pd.DataFrame: A copy of features with the columns transformed.
    """
    features = features.copy()
    features[columns] = np.log1p(features[columns])
    return features.rename(columns={column: f"Log{column}" for column in columns})


def add_quality_features(features: pd.DataFrame, train_features: pd.DataFrame) -> pd.DataFrame:
    """
    Add the interaction of quality with size, and a flag for the lowest qualities.

    Both columns are centred on their train means before they are multiplied, so the product is not strongly
    correlated with its two parts.

    Parameters:
    features (pd.DataFrame): The features, either train or test, with LogGrLivArea.
    train_features (pd.DataFrame): The train features, used for the means.

    Returns:
    pd.DataFrame: A copy of features with QualityTimesArea and LowQuality added.
    """
    features = features.copy()
    quality_mean, size_mean = train_features[QUALITY_COLUMN].mean(), train_features[SIZE_COLUMN].mean()
    features["QualityTimesArea"] = (features[QUALITY_COLUMN] - quality_mean) * (features[SIZE_COLUMN] - size_mean)
    features["LowQuality"] = (features[QUALITY_COLUMN] <= LOW_QUALITY_MAX).astype(int)
    return features


def _smoothed_neighbourhood_means(neighbourhoods: pd.Series, log_price: pd.Series) -> tuple[pd.Series, float]:
    """The mean log price of each neighbourhood, pulled towards the overall mean as if it had SMOOTHING extra houses."""
    overall = log_price.mean()
    groups = log_price.groupby(neighbourhoods).agg(["sum", "count"])
    return (groups["sum"] + SMOOTHING * overall) / (groups["count"] + SMOOTHING), overall


def neighbourhood_price(train: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    """
    The smoothed mean log price of each house's neighbourhood, as one numeric column.

    A house's own price must not be part of its own neighbourhood value, or the column leaks the target. So when df is
    train itself, each house gets the value from the other folds of train. Any other df gets the value from all of train.

    Parameters:
    train (pd.DataFrame): The cleaned train data, with Neighborhood and SalePrice.
    df (pd.DataFrame): The cleaned data to encode, either train or test.

    Returns:
    pd.Series: The neighbourhood price of each house in df.
    """
    log_price = np.log(train["SalePrice"])
    if not df.index.equals(train.index):
        means, overall = _smoothed_neighbourhood_means(train["Neighborhood"], log_price)
        return df["Neighborhood"].map(means).fillna(overall).rename("NeighbourhoodPrice")

    encoded = pd.Series(np.nan, index=train.index, name="NeighbourhoodPrice")
    for fit_index, other_index in KFold(ENCODING_FOLDS, shuffle=True, random_state=ENCODING_SEED).split(train):
        fit, other = train.iloc[fit_index], train.iloc[other_index]
        means, overall = _smoothed_neighbourhood_means(fit["Neighborhood"], log_price.iloc[fit_index])
        encoded.iloc[other_index] = other["Neighborhood"].map(means).fillna(overall).to_numpy()
    return encoded


def replace_neighbourhood(features: pd.DataFrame, train: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the Neighborhood dummies with the neighbourhood price.

    Parameters:
    features (pd.DataFrame): The features of df.
    train (pd.DataFrame): The cleaned train data.
    df (pd.DataFrame): The cleaned data that features came from, either train or test.

    Returns:
    pd.DataFrame: A copy of features without the Neighborhood dummies, and with NeighbourhoodPrice.
    """
    dummies = [column for column in features.columns if column.startswith(NEIGHBOURHOOD_PREFIX)]
    features = features.drop(columns=dummies).copy()
    features["NeighbourhoodPrice"] = neighbourhood_price(train, df)
    return features


def square_candidates(features: pd.DataFrame) -> list[str]:
    """
    The columns worth testing for curvature: more than MIN_DISTINCT_VALUES distinct values, and at most MAX_ZERO_SHARE zeros.

    The square of a column that is mostly zeros is decided by a handful of houses.

    Parameters:
    features (pd.DataFrame): The train features.

    Returns:
    list[str]: The names of the candidate columns.
    """
    return [column for column in features.columns if features[column].nunique() > MIN_DISTINCT_VALUES and (features[column] == 0).mean() <= MAX_ZERO_SHARE]


def add_squares(features: pd.DataFrame, train_features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Add the square of each column, centred on its train mean, so that the square is not strongly correlated with the column.

    Parameters:
    features (pd.DataFrame): The features, either train or test.
    train_features (pd.DataFrame): The train features, used for the means.
    columns (list[str]): The columns to square.

    Returns:
    pd.DataFrame: A copy of features with a {column}Squared column added for each column.
    """
    squares = {f"{column}Squared": (features[column] - train_features[column].mean()) ** 2 for column in columns}
    return pd.concat([features, pd.DataFrame(squares, index=features.index)], axis=1)
