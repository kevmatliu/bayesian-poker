ACTION_MAP = {
    0: "f",
    1: "cc",
    2: "cbr_small",
    3: "cbr_medium",
    4: "cbr_large",
}


SMALL_THRESH = 0.5
MEDIUM_THRESH = 1.0


def classify(action, pot_size, bet=None):
    if action == "f":
        return 0
    if action == "cc":
        return 1
    if action == "cbr":
        if bet is None:
            raise ValueError("Bet size must be provided for 'cbr' action.")
        if bet < SMALL_THRESH * pot_size:
            return 2
        if bet < MEDIUM_THRESH * pot_size:
            return 3
        return 4
    raise ValueError(f"Unknown action: {action}")
