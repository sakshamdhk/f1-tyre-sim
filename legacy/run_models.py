from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from nn import MLP
from stint_eval import CompMean, evaluate_stints

evaluate_stints(CompMean, "mean", save=True)
evaluate_stints(LinearRegression, "lr", save=True)
evaluate_stints(lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=20,
                                              n_jobs=-1, random_state=42), "rf", save=True)
evaluate_stints(lambda: MLP(hidden=(16, 8), lr=0.003), "nn", save=True)
