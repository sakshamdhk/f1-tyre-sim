from sklearn.ensemble import RandomForestRegressor
from evaluate import evaluate

FEATURES = ["tyre_age", "soft", "medium", "hard",
            "track_temp", "air_temp", "humidity", "stint_number"]

evaluate(lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                       n_jobs=-1, random_state=42), FEATURES, "rf")
