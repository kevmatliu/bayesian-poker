"""Optional stdout notices for evaluation (``verbose=True``)."""


def eval_log(verbose: bool, msg: str, *, prefix: str = "[eval]") -> None:
    if verbose:                               # caller asked for progress / diagnostic output
        print(f"{prefix} {msg}", flush=True)  # flush so logs appear immediately in pipelines
