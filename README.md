Solution to [Kaggle Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/). Ranked 281/4936 as of 20/11/2026. Since this is an unverified competition, a large majority of the scores above us contain data leakage or train models using the entire dataset.

## Final Model
The final submission has a leaderboard score of 0.11692 (RMSE of the log price). It combines three models, in the log price, and is built in `notebooks/Stacking.ipynb`:
- **CatBoost, weight 0.68.** An average of three seeds with settings tuned in `Boosting.ipynb`, trained with the two outlier houses and the extra columns of `add_tree_features`.
- **Lasso, weight 0.19.** A linear model on the same columns without logs, trained without the two outliers.
- **Support vector regression, weight 0.13.** An RBF kernel on the lasso's standardised columns, also trained without the outliers.

The lasso and the support vector regression can be far wrong for houses larger than any they were fitted on, so CatBoost alone predicts those (only `Id` 2550 in the test set).

## Reasoning
- The three models make different errors (correlations of 0.885 to 0.915), and the combination beats CatBoost alone on both of two fold splits (out-of-fold RMSE 0.1124 and 0.1129, against 0.1145 and 0.1161), and in every fold.
- The weights learned by nested cross-validation give the lasso and the support vector regression more than half of the combination, but the data determines that share only roughly. So the weights are taken halfway towards CatBoost alone, which keeps about three quarters of the gain.

## Notebooks
- `EDA`, `LinearRegression`, `Regularisation`: Exploration and linear models.
- `TreeModels`, `RandomForest`: Trees and forests, first written by hand.
- `Boosting`: XGBoost, LightGBM and CatBoost, error analysis and the CatBoost settings.
- `Stacking`: The models above and their combination.
