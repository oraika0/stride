import os
import pwd
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _user_home():
    """Resolve the invoking user's home dir, even under sudo.
    Path.home() returns /root under sudo; we want /home/<user> so the
    spawned terminal finds the user's conda install."""
    sudo_uid = os.environ.get("SUDO_UID")
    if sudo_uid:
        try:
            return pwd.getpwuid(int(sudo_uid)).pw_dir
        except (ValueError, KeyError):
            pass
    return str(Path.home())


config = {
    "conda_sh": str(Path(_user_home()) / "miniconda3/etc/profile.d/conda.sh"),
    "conda_env": "stride",
    "controller_dir": str(_REPO_ROOT / "utils"),
    "controller_entry": "simple_monitor.py",
}