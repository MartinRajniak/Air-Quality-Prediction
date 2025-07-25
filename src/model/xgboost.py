from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

def create_regressor() -> MultiOutputRegressor:
    base_regressor = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    multi_regressor = MultiOutputRegressor(base_regressor)
    return multi_regressor
