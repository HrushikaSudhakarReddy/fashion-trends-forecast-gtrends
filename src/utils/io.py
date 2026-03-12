
from pathlib import Path
BASE = Path(__file__).resolve().parents[2]
def data_path(*parts):
    return BASE.joinpath("data", *parts)
def ensure_dir(p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    return p
