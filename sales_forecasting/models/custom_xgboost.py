import xgboost as xgb


class CustomXGBRegressor(xgb.XGBRegressor):
    def __init__(self, n_estimators, monthly_avg, weekly_avg):
        super().__init__(n_estimators=n_estimators)
        self.monthly_avg = monthly_avg
        self.weekly_avg = weekly_avg
