from sklearn.linear_model import LinearRegression
from evaluate import evaluate

FEATURES = ["tyre_age", "soft", "medium", "hard",
            "age_soft", "age_medium", "age_hard",
            "track_temp", "air_temp", "humidity", "stint_number"]

evaluate(LinearRegression, FEATURES, "lr")
