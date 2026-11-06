K = 0.035          # fuel effect, s per lap (same value as in clean.py)
PIT_LOSS = 20.0    # seconds lost in a pit stop
MIN_STINT = 5      # shortest stint the optimiser will consider (laps)

BASE = ["tyre_age", "soft", "medium", "hard",
        "track_temp", "air_temp", "humidity", "stint_number"]
FEATURES = {
    "lr": BASE + ["age_soft", "age_medium", "age_hard"],
    "rf": BASE,
    "nn": BASE,
}
