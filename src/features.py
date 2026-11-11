import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


NONE = "None"
TYPICAL_FUNCTIONAL = "Typ"
UNFINISHED = "Unf"
YEARS_PER_DECADE = 10

ABSENT_COLUMNS = ["Alley", "FireplaceQu", "Fence"]

OUTLIER_IDS = [1299, 524]

LOT_FRONTAGE_FIT_EXCLUDE = [935, 314, 1338]
LOT_CONFIGS = ["Corner", "CulDSac", "FR2", "FR3"]
TOWNHOUSES = ["Twnhs", "TwnhsE"]

EXTERIOR_SPELLING = {"CemntBd": "CmentBd", "WdShing": "Wd Shng", "BrkComm": "Brk Cmn"}

KITCHEN_QUAL_KEYS = ["OverallQual", "ExterQual"]

BASEMENT_CATEGORIES = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
BASEMENT_NUMBERS = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
BASEMENT_MOST_COMMON = ["BsmtCond", "BsmtExposure", "BsmtFinType1"]

GARAGE_CATEGORIES = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
GARAGE_SIZES = ["GarageArea", "GarageCars"]
GARAGE_DETAILS = ["GarageFinish", "GarageQual", "GarageCond"]


def _decade(years: pd.Series) -> pd.Series:
    """The decade of each year, for example 1957 becomes 1950."""
    return (years // YEARS_PER_DECADE) * YEARS_PER_DECADE


def _most_common(values: pd.Series):
    """The most common value (the first, if there is a tie)."""
    return values.mode()[0]


def _fill_with_group_mode(df: pd.DataFrame, train: pd.DataFrame, column: str, group: str) -> pd.DataFrame:
    """Fill missing values of column, in place, with the most common train value for the same group."""
    mode_by_group = train.groupby(group)[column].agg(_most_common)
    overall_mode = _most_common(train[column])  # fallback for a group not in train

    missing = df[column].isnull()
    df.loc[missing, column] = df.loc[missing, group].map(mode_by_group).fillna(overall_mode)
    return df


def _fill_with_decade_mode(df: pd.DataFrame, train: pd.DataFrame, column: str) -> pd.DataFrame:
    """Fill missing values of column, in place, with the most common train value for the same decade built."""
    mode_by_decade = train.groupby(_decade(train["YearBuilt"]))[column].agg(_most_common)
    overall_mode = _most_common(train[column])  # fallback for a decade not in train

    missing = df[column].isnull()
    decade = _decade(df.loc[missing, "YearBuilt"])
    df.loc[missing, column] = decade.map(mode_by_decade).fillna(overall_mode)
    return df


def _lot_frontage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Predictors for LotFrontage: log lot area, lot configuration and a townhouse flag."""
    features = pd.DataFrame(index=df.index)
    features["log_lot_area"] = np.log(df["LotArea"])
    for config in LOT_CONFIGS:
        features[config] = (df["LotConfig"] == config).astype(int)
    features["Townhouse"] = df["BldgType"].isin(TOWNHOUSES).astype(int)
    return features


def fit_lot_frontage(train: pd.DataFrame) -> tuple[LinearRegression, tuple[float, float]]:
    """
    Fit a linear regression of log(LotFrontage) on the train houses with an observed value.

    Parameters:
    train (pd.DataFrame): The training data, indexed by Id.

    Returns:
    tuple: The fitted model, and the (min, max) LotFrontage of the fit rows, used to clip predictions.
    """
    observed = train[train["LotFrontage"].notnull()]
    fit_rows = observed.drop(index=LOT_FRONTAGE_FIT_EXCLUDE, errors="ignore")

    model = LinearRegression()
    model.fit(_lot_frontage_features(fit_rows), np.log(fit_rows["LotFrontage"]))

    bounds = (fit_rows["LotFrontage"].min(), fit_rows["LotFrontage"].max())
    return model, bounds


def fill_lot_frontage(
    df: pd.DataFrame, model: LinearRegression, bounds: tuple[float, float]
) -> pd.DataFrame:
    """
    Fill missing LotFrontage values using a model fitted on the train data.

    Parameters:
    df (pd.DataFrame): The data to fill, either train or test.
    model (LinearRegression): The model returned by fit_lot_frontage.
    bounds (tuple): The (min, max) LotFrontage that predictions are clipped to.

    Returns:
    pd.DataFrame: A copy of df with LotFrontage filled and a LotFrontageMissing flag added.
    """
    df = df.copy()
    missing = df["LotFrontage"].isnull()
    df["LotFrontageMissing"] = missing.astype(int)

    if missing.any():
        predicted = np.exp(model.predict(_lot_frontage_features(df[missing])))
        df.loc[missing, "LotFrontage"] = predicted.clip(*bounds)

    return df


def fill_ms_zoning(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing MSZoning with the most common zone in the same Neighborhood in train.

    Parameters:
    df (pd.DataFrame): The data to fill, either train or test.
    train (pd.DataFrame): The training data, used to find the most common zones.

    Returns:
    pd.DataFrame: A copy of df with MSZoning filled.
    """
    return _fill_with_group_mode(df.copy(), train, "MSZoning", "Neighborhood")


def fill_sale_type(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing SaleType with the most common type in the same SaleCondition in train.

    Parameters:
    df (pd.DataFrame): The data to fill, either train or test.
    train (pd.DataFrame): The training data, used to find the most common types.

    Returns:
    pd.DataFrame: A copy of df with SaleType filled.
    """
    return _fill_with_group_mode(df.copy(), train, "SaleType", "SaleCondition")


def fill_exterior(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing Exterior1st and Exterior2nd.

    Exterior1st is filled with the most common material in train for houses built in the same decade.
    Exterior2nd is filled with the (filled) Exterior1st, using the Exterior2nd spelling.

    Parameters:
    df (pd.DataFrame): The data to fill, either train or test.
    train (pd.DataFrame): The training data, used to find the most common materials.

    Returns:
    pd.DataFrame: A copy of df with Exterior1st and Exterior2nd filled.
    """
    df = _fill_with_decade_mode(df.copy(), train, "Exterior1st")

    missing_second = df["Exterior2nd"].isnull()
    df.loc[missing_second, "Exterior2nd"] = df.loc[missing_second, "Exterior1st"].replace(EXTERIOR_SPELLING)

    return df


def fill_electrical(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing Electrical with the most common system in train for houses built in the same decade.

    Parameters:
    df (pd.DataFrame): The data to fill, either train or test.
    train (pd.DataFrame): The training data, used to find the most common systems.

    Returns:
    pd.DataFrame: A copy of df with Electrical filled.
    """
    return _fill_with_decade_mode(df.copy(), train, "Electrical")


def fill_kitchen_qual(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing KitchenQual with the most common value in train for houses with the same OverallQual and ExterQual.

    Falls back to the most common value for the same OverallQual, then to the most common value overall.

    Parameters:
    df (pd.DataFrame): The data to fill, either train or test.
    train (pd.DataFrame): The training data, used to find the most common values.

    Returns:
    pd.DataFrame: A copy of df with KitchenQual filled.
    """
    df = df.copy()
    missing = df["KitchenQual"].isnull()
    if not missing.any():
        return df

    by_both = train.groupby(KITCHEN_QUAL_KEYS)["KitchenQual"].agg(_most_common)
    by_overall = train.groupby("OverallQual")["KitchenQual"].agg(_most_common)

    rows = df.loc[missing, KITCHEN_QUAL_KEYS]
    filled = pd.Series(by_both.reindex(pd.MultiIndex.from_frame(rows)).to_numpy(), index=rows.index)
    filled = filled.fillna(rows["OverallQual"].map(by_overall)).fillna(_most_common(train["KitchenQual"]))

    df.loc[missing, "KitchenQual"] = filled
    return df


def fill_functional(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Functional with "Typ", as the data description says to assume typical functionality."""
    df = df.copy()
    df["Functional"] = df["Functional"].fillna(TYPICAL_FUNCTIONAL)
    return df


def drop_utilities(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Utilities, which is AllPub for all but one train house and so carries no information."""
    return df.drop(columns="Utilities", errors="ignore")


def fill_masonry(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing MasVnrType and MasVnrArea, and add a MasVnrMissing flag.

    MasVnrType is filled with the most common type in train for the same Neighborhood. If the area is known
    and above 0 there is a veneer, so it is filled with the most common veneer type instead.
    MasVnrArea is filled with 0 for the type "None", otherwise with the median area of that type in train.

    Parameters:
    df (pd.DataFrame): The data to fill, either train or test.
    train (pd.DataFrame): The training data, loaded with "None" as a real MasVnrType category.

    Returns:
    pd.DataFrame: A copy of df with MasVnrType and MasVnrArea filled and a MasVnrMissing flag added.
    """
    df = df.copy()
    df["MasVnrMissing"] = (df["MasVnrType"].isnull() | df["MasVnrArea"].isnull()).astype(int)

    veneers = train[(train["MasVnrType"].notnull()) & (train["MasVnrType"] != NONE) & (train["MasVnrArea"] > 0)]
    veneer_type = _most_common(veneers["MasVnrType"])
    area_by_type = veneers.groupby("MasVnrType")["MasVnrArea"].median()

    has_veneer = df["MasVnrType"].isnull() & (df["MasVnrArea"] > 0)
    df.loc[has_veneer, "MasVnrType"] = veneer_type
    df = _fill_with_group_mode(df, train, "MasVnrType", "Neighborhood")

    missing_area = df["MasVnrArea"].isnull()
    df.loc[missing_area, "MasVnrArea"] = df.loc[missing_area, "MasVnrType"].map(area_by_type).fillna(0)

    return df


def fill_absent(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill NaN with "None" in columns where a NaN means the feature is absent, such as no alley access."""
    df = df.copy()
    df[columns] = df[columns].fillna(NONE)
    return df


def fill_pool_qc(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing PoolQC with "None" if the house has no pool, otherwise with the most common rating among
    train pools for houses with the same ExterQual.

    Parameters:
    df (pd.DataFrame): The data to fill, either train or test.
    train (pd.DataFrame): The training data, used to find the most common rating.

    Returns:
    pd.DataFrame: A copy of df with PoolQC filled.
    """
    df = df.copy()
    no_pool = df["PoolQC"].isnull() & ~(df["PoolArea"] > 0)
    df.loc[no_pool, "PoolQC"] = NONE

    train_pools = train[train["PoolArea"] > 0]
    return _fill_with_group_mode(df, train_pools, "PoolQC", "ExterQual")


def fill_misc_feature(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing MiscFeature with "None" if MiscVal is 0, otherwise with the type whose median MiscVal in train is closest.

    Parameters:
    df (pd.DataFrame): The data to fill, either train or test.
    train (pd.DataFrame): The training data, used to find the median value of each type.

    Returns:
    pd.DataFrame: A copy of df with MiscFeature filled.
    """
    df = df.copy()
    missing = df["MiscFeature"].isnull()
    if not missing.any():
        return df

    median_value = train[train["MiscVal"] > 0].groupby("MiscFeature")["MiscVal"].median()
    values = df.loc[missing, "MiscVal"]
    closest = values.apply(lambda value: (median_value - value).abs().idxmin())
    df.loc[missing, "MiscFeature"] = closest.where(values > 0, NONE)
    return df


def _has_basement(df: pd.DataFrame) -> pd.Series:
    """A house has a basement if its area is above 0 or any of its ratings is present."""
    return (df["TotalBsmtSF"] > 0) | df[BASEMENT_CATEGORIES].notnull().any(axis=1)


def fill_basement(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """
    Fill the missing basement columns.

    Houses with no basement get "None" for the ratings and 0 for the sizes and bathroom counts.
    For houses with a basement, a missing rating is filled as follows:
    - BsmtFinType2 is "Unf" if BsmtFinSF2 is 0, otherwise the most common type among train houses with a second area.
    - BsmtQual is the most common height in train for houses built in the same decade.
    - BsmtCond, BsmtExposure and BsmtFinType1 are the most common value in train.

    Parameters:
    df (pd.DataFrame): The data to fill, either train or test.
    train (pd.DataFrame): The training data, used to find the most common values.

    Returns:
    pd.DataFrame: A copy of df with the basement columns filled.
    """
    df = df.copy()
    has_basement = _has_basement(df)
    train_basement = train[_has_basement(train)]

    df.loc[~has_basement, BASEMENT_CATEGORIES] = NONE
    df.loc[~has_basement, BASEMENT_NUMBERS] = df.loc[~has_basement, BASEMENT_NUMBERS].fillna(0)

    second_type = _most_common(train_basement.loc[train_basement["BsmtFinSF2"] > 0, "BsmtFinType2"])
    missing = df["BsmtFinType2"].isnull()
    df.loc[missing, "BsmtFinType2"] = np.where(df.loc[missing, "BsmtFinSF2"] > 0, second_type, UNFINISHED)

    df = _fill_with_decade_mode(df, train_basement, "BsmtQual")
    for column in BASEMENT_MOST_COMMON:
        df[column] = df[column].fillna(_most_common(train_basement[column]))

    return df


def _has_garage(df: pd.DataFrame) -> pd.Series:
    """A house has a garage if its area or car capacity is above 0 or any descriptive column is present."""
    present = df[GARAGE_CATEGORIES + ["GarageYrBlt"]].notnull().any(axis=1)
    return (df["GarageArea"] > 0) | (df["GarageCars"] > 0) | present


def fill_garage(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """
    Fill the missing garage columns, and add a HasGarage flag.

    Houses with no garage get "None" for the descriptive columns and 0 for the area and car capacity.
    For houses with a garage, the missing columns are filled from train as follows:
    - GarageFinish, GarageQual and GarageCond are the most common value for the same GarageType.
    - GarageCars is the most common value for the same GarageType and decade built.
    - GarageArea is the median for the same GarageType and GarageCars.
    GarageYrBlt is YearBuilt when it is missing, and also when it is after both the sale and the house.

    Parameters:
    df (pd.DataFrame): The data to fill, either train or test.
    train (pd.DataFrame): The training data, used to find the most common values.

    Returns:
    pd.DataFrame: A copy of df with the garage columns filled and a HasGarage flag added.
    """
    df = df.copy()
    has_garage = _has_garage(df)
    train_garage = train[_has_garage(train)]
    df["HasGarage"] = has_garage.astype(int)

    df.loc[~has_garage, GARAGE_CATEGORIES] = NONE
    df.loc[~has_garage, GARAGE_SIZES] = 0

    df["GarageType"] = df["GarageType"].fillna(_most_common(train_garage["GarageType"]))
    for column in GARAGE_DETAILS:
        df = _fill_with_group_mode(df, train_garage, column, "GarageType")

    cars_by_type_decade = train_garage.groupby(["GarageType", _decade(train_garage["YearBuilt"])])["GarageCars"].agg(_most_common)
    cars_by_type = train_garage.groupby("GarageType")["GarageCars"].agg(_most_common)
    missing = df["GarageCars"].isnull()
    if missing.any():
        rows = df.loc[missing]
        keys = pd.MultiIndex.from_arrays([rows["GarageType"], _decade(rows["YearBuilt"])])
        cars = pd.Series(cars_by_type_decade.reindex(keys).to_numpy(), index=rows.index)
        df.loc[missing, "GarageCars"] = cars.fillna(rows["GarageType"].map(cars_by_type))

    area_by_type_cars = train_garage.groupby(["GarageType", "GarageCars"])["GarageArea"].median()
    missing = df["GarageArea"].isnull()
    if missing.any():
        rows = df.loc[missing]
        keys = pd.MultiIndex.from_arrays([rows["GarageType"], rows["GarageCars"]])
        area = pd.Series(area_by_type_cars.reindex(keys).to_numpy(), index=rows.index)
        df.loc[missing, "GarageArea"] = area.fillna(train_garage["GarageArea"].median())

    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])
    impossible = df["GarageYrBlt"] > np.maximum(df["YrSold"], df["YearBuilt"])
    df.loc[impossible, "GarageYrBlt"] = df.loc[impossible, "YearBuilt"]

    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the training houses that the EDA found to be outliers (OUTLIER_IDS).

    Test houses are never removed, as their Ids are not in the list.

    Parameters:
    df (pd.DataFrame): The data, indexed by Id.

    Returns:
    pd.DataFrame: A copy of df without the outlier houses.
    """
    return df.drop(index=OUTLIER_IDS, errors="ignore")


def clean(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the outlier houses, fill every missing value in df, drop Utilities, and add the missing-value flags.

    All statistics come from train without the outlier houses, so the same call cleans train and test
    without using any test information. Pass train as it is loaded from the CSV file, not an already cleaned copy.

    Parameters:
    df (pd.DataFrame): The data to clean, either train or test.
    train (pd.DataFrame): The raw training data, used to fit the models and find the most common values.

    Returns:
    pd.DataFrame: A cleaned copy of df, without the outlier houses and with no missing values, and the flags
    LotFrontageMissing, MasVnrMissing and HasGarage added.
    """
    train = remove_outliers(train)
    df = remove_outliers(df)
    model, bounds = fit_lot_frontage(train)
    df = fill_lot_frontage(df, model, bounds)
    df = fill_ms_zoning(df, train)
    df = drop_utilities(df)
    df = fill_exterior(df, train)
    df = fill_kitchen_qual(df, train)
    df = fill_functional(df)
    df = fill_electrical(df, train)
    df = fill_sale_type(df, train)
    df = fill_masonry(df, train)
    df = fill_absent(df, ABSENT_COLUMNS)
    df = fill_pool_qc(df, train)
    df = fill_misc_feature(df, train)
    df = fill_basement(df, train)
    df = fill_garage(df, train)
    return df
