"""Optional stdout notices for evaluation (``verbose=True``)."""


def eval_log(verbose: bool, msg: str, *, prefix: str = "[eval]") -> None:
    if verbose:
        print(f"{prefix} {msg}", flush=True)
