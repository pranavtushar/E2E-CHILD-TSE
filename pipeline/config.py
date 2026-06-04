"""TSA mixture dataset parameters (AA / CA / CC)."""
import os
from pathlib import Path

TSA_RUN = os.environ.get("TSA_RUN", "tsa_mix")
DEFAULT_SEED = 42
L_SEC = 5.0
SR = 16000
MIN_UTT_LENGTH_SEC = 5.0
SNR_DB = 0
MIX_PEAK = 0.9
OVERLAPS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_MIXTURES_PER_OVERLAP = 50
SUBSETS = ("AA", "CC", "CA")


def _data_subdir() -> str:
    return os.environ.get("DATA_SUBDIR", "data").strip().rstrip("/") or "data"


def get_base_dir():
    return os.environ.get("TSA_MIX_BASE_DIR", os.environ.get("BASE_DIR", ""))


def _bundled(base: Path, *parts: str) -> Path:
    return base / _data_subdir() / Path(*parts)


def get_myst_csv(base: Path) -> Path:
    bundled = _bundled(base, "myst", "myST_test_filtered", "filtered_list_has_trn.csv")
    if bundled.is_file():
        return bundled
    return base / "latest-child-speech-dataset" / "myST_test_filtered" / "filtered_list_has_trn.csv"


def get_myst_root(base: Path) -> Path:
    bundled = _bundled(base, "myst", "myST_test_filtered")
    if bundled.is_dir():
        return bundled
    return base / "latest-child-speech-dataset" / "myST_test_filtered"


def get_libri_root(base: Path) -> Path:
    bundled = _bundled(base, "libri", "test-clean-normalised")
    if bundled.is_dir():
        return bundled
    return base / "LibriSpeech" / "test-clean-normalised"


def get_run_root(base) -> Path:
    if base and str(base).strip():
        return Path(base) / "tse" / TSA_RUN
    return Path(__file__).resolve().parent


def get_output_root(base: Path) -> Path:
    return get_run_root(base) / "data" / "TSA_MIXED"
