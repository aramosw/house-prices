import pandas as pd

from src.features import EXTERIOR_SPELLING


OTHER = "Other"
MIN_LEVEL_COUNT = 10

BATHROOM_COLUMNS = ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
YEAR_COLUMNS = ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]
SUM_PARTS = ["1stFlrSF", "BsmtUnfSF"]
REPEATED_COLUMNS = ["HasGarage", "Exterior2nd"]
REPLACED_COLUMNS = BATHROOM_COLUMNS + YEAR_COLUMNS + SUM_PARTS + REPEATED_COLUMNS

DUPLICATE_COLUMNS = ["BldgType_Duplex"]

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
