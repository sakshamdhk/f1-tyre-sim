# Results

All numbers come from `python pipeline.py` and are saved in `results/`. Every prediction is
out-of-sample. A race is always predicted by a model that never saw its circuit (cross-validation)
or its season (temporal test).

## Data

| | |
|---|---|
| Seasons | 2023 and 2024, Grands Prix only (no sprints), from the OpenF1 API |
| Races | 46 downloaded, 40 dry (6 wet races auto-detected and excluded), Monaco excluded → **39 races, 23 circuits** |
| Clean laps | 34,692 (filters: lap 1, pit in/out, SC/VSC/red flag and restart laps, yellow flags, field-wide slow laps, > 3 s off the stint median) |
| Stints | 1,606 with at least 8 laps of tyre-age range |
| Target | `deg`: lap time corrected for fuel (k = 3.0 / race laps s/lap) and track evolution, minus the median of the stint's first 3 laps |

Two corrections in the target are worth highlighting:

* **Track evolution.** Street circuits get much faster as rubber goes down, which hides tyre wear.
  Per race, a fixed-effects regression `lap time = driver + compound + wear·age + evo·lap` separates
  the two. One fixed effect per *stint* cannot work: within a stint age and lap rise together, so
  wear and evolution are perfectly collinear. The pace jump at each pit stop is what identifies them.
* **Compound offsets.** A fixed-effects regression on stint pace gives soft −0.25 s and medium
  −0.07 s per lap relative to hard. Drivers who ran medium→hard and hard→medium let the model separate
  the compound effect from "later in the race is faster".

## 1. How well is tyre degradation predicted?

Lap-level MAE (s), with a 95% bootstrap interval that resamples whole races. *P(beats baseline)* is
the share of bootstrap resamples in which the model has a lower MAE than the baseline. *Curve* metrics
average out lap noise. They compare the mean observed and predicted degradation per race, compound
and tyre age.

**Unseen circuits (5-fold cross-validation grouped by circuit)**

| Model | Lap MAE | 95% CI | Lap R² | Curve MAE | Curve R² | P(beats baseline) |
|---|---|---|---|---|---|---|
| Baseline: one wear rate per compound | 0.590 | 0.543–0.645 | 0.15 | 0.386 | 0.30 | – |
| Linear wear-rate model (OLS) | 0.709 | 0.596–0.880 | −0.57 | 0.528 | −1.20 | 0.00 |
| Ridge (L2 on condition effects) | 0.667 | 0.572–0.813 | −0.33 | 0.464 | −0.68 | 0.03 |
| **Random forest** | **0.568** | 0.521–0.621 | **0.20** | **0.351** | **0.37** | **0.96** |
| Gradient boosting | 0.588 | 0.542–0.644 | 0.14 | 0.373 | 0.27 | 0.56 |
| Neural network (numpy MLP) | 0.609 | 0.565–0.657 | 0.13 | 0.398 | 0.24 | 0.10 |
| Physics-informed NN ensemble (numpy) | 0.605 | 0.558–0.662 | 0.12 | 0.401 | 0.22 | 0.04 |

**Next season (train 2023, test 2024)**

| Model | Lap MAE | 95% CI | Lap R² | Curve MAE | Curve R² | P(beats baseline) |
|---|---|---|---|---|---|---|
| Baseline | 0.592 | 0.536–0.650 | 0.14 | 0.393 | 0.28 | – |
| Linear (OLS) | 0.609 | 0.530–0.712 | 0.07 | 0.424 | 0.09 | 0.39 |
| Ridge | 0.591 | 0.524–0.659 | 0.15 | 0.385 | 0.26 | 0.58 |
| **Random forest** | **0.571** | 0.504–0.646 | **0.19** | **0.360** | **0.32** | **0.74** |
| Gradient boosting | 0.599 | 0.531–0.677 | 0.14 | 0.404 | 0.18 | 0.43 |
| Neural network (MLP) | 0.618 | 0.542–0.703 | 0.09 | 0.422 | 0.11 | 0.20 |
| Physics-informed NN ensemble | 0.594 | 0.539–0.648 | 0.18 | 0.390 | 0.28 | 0.44 |

**What the noise floor is.** Even a model that knew each race's true wear rate per compound
(fitted in-sample) would only reach a lap MAE of about 0.48 s. With each individual stint's wear
known, it would reach 0.28 s. Most lap-to-lap variation is driver-specific: tyre management, setup,
traffic. About 32% of the variation in stint wear rates is explained by race and compound together,
and the same circuit's wear rate correlates at only r = 0.46 between 2023 and 2024. That leaves
little room for any race-level model, and it is why the differences between models are a few
hundredths of a second.

## 2. Does track temperature drive degradation?

The project's starting hypothesis was that hotter tracks wear tyres faster (plot 3).

* Raw correlation between a stint's wear rate and track temperature: r = +0.01 (hard), +0.02
  (medium), −0.24 (soft).
* With other conditions held at their median, the physics-informed NN predicts that wear rises
  about 25% from 18 °C to 49 °C (soft 0.035 → 0.043 s/lap, hard 0.021 → 0.026 s/lap). This is a
  small, physically sensible effect.
* In the permutation importance (plot 7), track temperature adds almost nothing to either model.
  Tyre age dominates. Gradient boosting draws some signal from stint number, corner count, air
  temperature and traffic.

**Conclusion:** the effect is real but small in 2023–24 race data. A large part of the reason is
that Pirelli already chooses harder or softer compounds for each race to match its expected
conditions, which cancels most of the temperature effect before the race starts.

## 3. Does the optimiser pick the pit lap teams picked?

There are 13 green-flag one-stop races (≥ 5 one-stoppers, fewer than half of them stopping under a
safety car or VSC). In five more races, most stops came under a safety car, and these are reported
separately. Errors are in laps (+ = model later than the teams).

| Prediction | Mean abs error | Bias | Within 3 laps |
|---|---|---|---|
| Half race distance (rule of thumb) | 6.7 | +6.7 | 23% |
| **Oracle**: the race's own measured wear rates | 6.8 | +3.9 | 23% |
| Baseline, tyre-optimal | 10.0 | +9.9 | 8% |
| Random forest, tyre-optimal | 10.9 | +10.5 | 8% |
| Gradient boosting, tyre-optimal | 7.6 | +6.2 | 23% |
| MLP, tyre-optimal | 8.5 | +8.1 | 15% |
| Physics-informed NN, tyre-optimal | 7.9 | +7.7 | 15% |
| Baseline + strategy calibration | 6.5 | −0.5 | 23% |
| Gradient boosting + calibration | 5.7 | −0.8 | 31% |
| **MLP + calibration** | **5.2** | −1.5 | **38%** |
| **Physics-informed NN + calibration** | **5.4** | **−0.3** | 31% |

**The key finding:** even with the *true* measured wear rates of each race, the tyre-optimal pit lap
is about 7 laps away from what teams did, and still 4 laps late on average. Teams do not run the
race that is fastest for a car alone. They stop earlier, because pitting before a rival (the
*undercut*) gains track position. A pure tyre model cannot see rivals.

*Strategy calibration* learns that tendency as a shift (a fraction of race distance), using only the
training races of each fold. With it, the neural networks predict the real pit lap best, with a
mean error of about 5 laps. That beats both the rule of thumb and the oracle.

Under the temporal protocol (only 8 races), the baseline's tyre-optimal lap is closest (4.8 laps).
With so few races, this ranking is not reliable, and calibration is not possible because there are
only 5 green-flag 2023 races to learn it from.

## 4. Would more data help? (plot 5)

The learning curve below is the test MAE averaged over the 5 folds, training on 25–100% of the
training circuits (one network instead of the ensemble).

| Training races | ~8 | ~16 | ~24 | ~31 |
|---|---|---|---|---|
| Ridge | 0.636 | 0.643 | 0.676 | 0.649 |
| Gradient boosting | 0.669 | 0.635 | 0.646 | 0.596 |
| Physics-informed NN | 0.875 | 0.645 | 0.641 | 0.621 |

Ridge has flattened, but gradient boosting and the network are still improving at the full dataset.
More seasons (2022, or 2025 when it is complete) should help the flexible models most.

## 5. Neural network notes

* Both networks are written from scratch in numpy: forward pass, backpropagation, Adam, L2 weight
  decay, dropout, He initialisation and early stopping on held-out *circuits*. Gradient checks pass
  (relative error ~1e-8), and both networks can overfit 64 rows (`tests/`).
* **The first version overfit badly.** With about 19 training circuits, the network memorised each
  circuit from its descriptors: validation loss doubled while training loss fell (plot 6, MLP).
* **The fix was architectural.** The physics-informed network predicts the *parameters* of a wear
  curve, and a softplus keeps them non-negative so tyres can never get faster with age. It is also
  residual: each compound has a free base wear rate, and the network only learns corrections from
  conditions. Weight decay then shrinks it toward the baseline rather than toward a constant. Nested
  validation chose strong weight decay (0.1–0.3) and, in two folds of five, the smaller physical
  feature set. Its training and validation losses stay together (plot 6).
* The ensemble of five networks gives an uncertainty band (plot 3, the what-if page) and a spread of
  optimal pit laps.

## Limitations

* No safety cars, traffic or rivals in the optimiser; one race car on its own.
* Fuel effect is a constant assumption (0.03 s per kg, 100 kg); pit loss is measured per race.
* Two seasons; 2022 would need an F1TV subscription (the official timing feed has required one
  since 2026).
* Pirelli's compound allocation (C1–C5) per race is not in the data; it would likely be the single
  most useful extra feature.
* Wear is measured relative to each stint's first laps, so it captures everything that changes
  within a stint, including how hard the driver chooses to push.
