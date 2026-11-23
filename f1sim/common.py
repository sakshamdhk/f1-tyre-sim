"""Constants shared by the whole pipeline."""

SEED = 42
N_FOLDS = 5
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# Fuel: cars start with about 100 kg and burn it evenly over the race; each kg costs
# about 0.03 s per lap. So the per-lap gain depends on race length (k = 3.0 / laps).
FUEL_KG = 100.0
FUEL_S_PER_KG = 0.03


def fuel_k(total_laps):
    return FUEL_KG * FUEL_S_PER_KG / total_laps


PIT_LOSS_DEFAULT = 22.0   # s, used only when a race has no usable green-flag stop
REF_LAPS = 3              # first clean laps of a stint that define its reference pace
TRAFFIC_GAP = 1.0         # s to the car ahead below which a lap counts as in traffic
MIN_STINT = 8             # shortest stint the optimiser will consider (laps)

# Street and semi-street circuits (OpenF1 circuit short names): the track surface is dirty and
# "rubbers in" strongly during the race, which hides tyre wear. A circuit type, not a track name.
STREET = {"Monte Carlo", "Singapore", "Baku", "Jeddah", "Las Vegas", "Melbourne", "Miami", "Montreal"}
# Races removed from modelling, with the reason.
EXCLUDED = {"Monte Carlo": "pace set by traffic and track position, not tyres "
                           "(2024: lap-1 red flag, field lapped 2-3 s off the pace for 77 laps)"}
