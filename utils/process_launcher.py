"""Launch the controller and the agent as separate processes.

A real run is three processes: the orchestrator (main.py or test_single_tm.py)
running as root to build the Mininet topology, the Ryu controller, and the DRL
agent. The last two need to be visible while the run proceeds, which is why they
get their own terminal rather than being piped into the orchestrator's output.

Backends
--------
`tmux`    a tmux window each, in a detached session you can attach to later.
          Needs no DISPLAY and no session bus, so it works over SSH, and the
          window survives the process so a crash stays readable.
`gnome`   a gnome-terminal window each, the historical behaviour.
`inline`  no terminal: the children inherit the orchestrator's stdout. Works
          anywhere; the two output streams interleave.
`auto`    tmux if installed, else gnome when a DISPLAY is present, else inline.

Why gnome-terminal is awkward here: it talks to the user's DBus session bus,
which root cannot reach. Under sudo the launcher therefore drops back to
SUDO_USER and forwards DISPLAY, XAUTHORITY, the bus address and XDG_RUNTIME_DIR
by hand. The bus address is guessed from /run/user/<uid>/bus when the variable
itself did not survive sudo.

Every backend returns an object with Popen's terminate() / wait() / poll(), so
callers do not branch on which one they got.
"""

import os
import pwd
import shlex
import shutil
import subprocess
import time

from utils.project_root import PROJECT_ROOT


TERMINAL_CHOICES = ("auto", "tmux", "gnome", "inline")

# The detached session the children are put in when the orchestrator is not
# already inside tmux. `tmux attach -t stride` to watch them.
TMUX_SESSION = "stride"

# Variables the child needs and `sudo -u` strips. ENV_NAME and ALG_NAME are set
# by the orchestrator so the controller and the agent resolve the same dataset
# and results directory; without them simple_monitor falls back to "32node" and
# raises a KeyError on geant dpids.
_FORWARD = ("DISPLAY", "XAUTHORITY", "ENV_NAME", "ALG_NAME", "NUM_LINK",
            "DRSIR_REWARD_CORRECTED", "STRIDE_VARIANT", "STRIDE_EVAL_SAMPLE",
            "DELAY_TRACE_LINKS", "DELAY_TRACE_FILE", "LLDP_RAW_TRACE_FILE")


class TmuxWindow:
    """Popen-compatible handle on a command running in its own tmux window.

    tmux keeps the window after the command exits (remain-on-exit), so a crash
    stays on screen. That also means the exit status is readable afterwards,
    which is how wait() and poll() report it.
    """

    def __init__(self, pane_id, name):
        self.pane_id = pane_id
        self.name = name
        self.returncode = None
        self._terminated = False

    def _query(self):
        """(alive?, status) for the pane, or (None, None) if it is gone."""
        r = subprocess.run(
            ["tmux", "display-message", "-p", "-t", self.pane_id,
             "#{pane_dead}:#{pane_dead_status}"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        if r.returncode != 0:
            return None, None
        dead, _, status = r.stdout.strip().partition(":")
        # tmux does not fail on a pane that no longer exists: it exits 0 and
        # expands the format to nothing, so the reply for a vanished pane is
        # ":" rather than an error. Reading that as dead != "1" reports the
        # pane as alive for ever, and since terminate() is `kill-window`, a
        # terminate() followed by wait() then never returns -- which is exactly
        # what hung test_single_tm after its first traffic matrix.
        if dead not in ("0", "1"):
            return None, None
        if dead != "1":
            return True, None
        return False, int(status) if status.strip().isdigit() else 0

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        alive, status = self._query()
        if alive:
            return None
        if alive is None:
            # The pane vanished -- someone killed the window, or the server
            # went away. If we did it, that is the expected path; otherwise the
            # status is genuinely unknown and must not be reported as success.
            self.returncode = 0 if self._terminated else -1
        else:
            self.returncode = status
        return self.returncode

    def wait(self, timeout=None):
        waited = 0.0
        while True:
            rc = self.poll()
            if rc is not None:
                return rc
            if timeout is not None and waited >= timeout:
                raise subprocess.TimeoutExpired(f"tmux window {self.name}", timeout)
            time.sleep(0.5)
            waited += 0.5

    def terminate(self):
        self._terminated = True
        subprocess.run(["tmux", "kill-window", "-t", self.pane_id],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    kill = terminate


def _give_to_sudo_user(path, mode=None):
    """Hand `path` to SUDO_USER when we are running as root under sudo.

    Anything root creates here is later touched by a child running as the
    invoking user -- the tmux server writing a pane log, run_drl reading a
    config -- and root ownership turns that into a silent failure rather than
    an error anyone sees.
    """
    sudo_uid = os.environ.get("SUDO_UID")
    if not (sudo_uid and os.geteuid() == 0):
        return
    try:
        uid = int(sudo_uid)
        os.chown(path, uid, pwd.getpwuid(uid).pw_gid)
        if mode is not None:
            os.chmod(path, mode)
    except (ValueError, KeyError, OSError):
        pass


def _tmux_target():
    """Where to put the child's window, creating a detached session if needed.

    Being inside tmux is not required: when the orchestrator is not, the window
    goes into a detached session you can `tmux attach -t stride` to. That also
    sidesteps sudo dropping TMUX from the environment.
    """
    if os.environ.get("TMUX"):
        return None            # current session
    exists = subprocess.run(["tmux", "has-session", "-t", TMUX_SESSION],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            check=False).returncode == 0
    if not exists:
        subprocess.run(["tmux", "new-session", "-d", "-s", TMUX_SESSION],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return TMUX_SESSION


def resolve_terminal(requested="auto"):
    """Turn 'auto' into the backend that is actually available."""
    if requested != "auto":
        return requested
    if shutil.which("tmux"):
        return "tmux"
    term = os.environ.get("TERMINAL", "gnome-terminal")
    if os.environ.get("DISPLAY") and shutil.which(term):
        return "gnome"
    return "inline"


def _env_passthrough():
    """`VAR=value` strings for the run-defining variables, if they are set."""
    return [f"{k}={os.environ[k]}" for k in _FORWARD if os.environ.get(k)]


def _sudo_env_passthrough():
    """As above, plus what a GUI child needs and `sudo -u` would drop."""
    out = _env_passthrough()
    sudo_uid = os.environ.get("SUDO_UID")
    dbus = os.environ.get("DBUS_SESSION_BUS_ADDRESS")
    if not dbus and sudo_uid:
        bus_path = f"/run/user/{sudo_uid}/bus"
        if os.path.exists(bus_path):
            dbus = f"unix:path={bus_path}"
    if dbus:
        out.append(f"DBUS_SESSION_BUS_ADDRESS={dbus}")
    if sudo_uid:
        out.append(f"XDG_RUNTIME_DIR=/run/user/{sudo_uid}")
    return out


def launch(cmd, name, terminal="auto"):
    """Run `cmd` under bash and return a Popen-compatible handle.

    `name` only labels the log line, so a reader can tell the two children apart.
    """
    backend = resolve_terminal(terminal)

    if backend == "inline":
        # No `exec bash` tail here: nothing keeps the shell alive afterwards, so
        # terminate() reaches the process itself rather than a wrapper.
        print(f"[launch] {name}: inline (output interleaves with this terminal)")
        return subprocess.Popen(["bash", "-c", cmd])

    if backend == "tmux":
        if not shutil.which("tmux"):
            raise RuntimeError("terminal backend 'tmux' needs tmux on PATH.")
        session = _tmux_target()
        target = [] if session is None else ["-t", session]
        # tmux new-window does not inherit this process's environment -- the
        # child gets the tmux *server's*, which was set up long before this run.
        # Everything in _FORWARD would silently vanish: ALG_NAME missing sends
        # the controller to results/ls2ic/drl_paths.json, ENV_NAME missing makes
        # simple_monitor fall back to 32node, STRIDE_VARIANT missing trains the
        # default architecture. Export them inside the command instead of using
        # `env VAR=v cmd`, because cmd is a && chain and env would only reach its
        # first link.
        exports = " ".join(shlex.quote(p) for p in _env_passthrough())
        wrapped = f"export {exports}; {cmd}" if exports else cmd
        r = subprocess.run(
            ["tmux", "new-window", "-d", "-P", "-F", "#{pane_id}", "-n", name]
            + target + ["bash", "-lc", wrapped],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        pane_id = r.stdout.strip()
        # Per-window, not -g: a global setting would change every window the
        # user already has open.
        subprocess.run(["tmux", "set-option", "-t", pane_id, "remain-on-exit", "on"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        # Keep the pane's output on disk too. A pane holds two thousand lines and
        # the controller prints a link table every cycle, so by the time anyone
        # looks, the traceback that explains the failure has scrolled off -- the
        # controller death that cost a 32-node run was nearly undiagnosable for
        # exactly this reason. pipe-pane costs a `cat`, and the log makes the
        # difference between reading the exception and re-running to guess at it.
        log_dir = os.path.join(PROJECT_ROOT, "results", "_terminal_logs")
        log = os.path.join(log_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        try:
            os.makedirs(log_dir, exist_ok=True)
            # main.py runs under sudo, so this directory is root-owned, while
            # pipe-pane's `cat` runs as the tmux server's user. Without this the
            # redirect fails and the log is silently never written.
            _give_to_sudo_user(log_dir, 0o755)
            subprocess.run(["tmux", "pipe-pane", "-t", pane_id, "-o",
                            f"cat >> {shlex.quote(log)}"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False)
        except OSError as exc:
            print(f"[launch] {name}: no pane log ({exc})")
            log = None
        where = "current session" if session is None else f"session '{session}'"
        print(f"[launch] {name}: tmux window {pane_id} in {where}"
              + (f", logging to {log}" if log else "")
              + ("" if session is None else f" -- attach with: tmux attach -t {session}"))
        return TmuxWindow(pane_id, name)

    if backend != "gnome":
        raise ValueError(f"unknown terminal backend {backend!r}; "
                         f"expected one of {TERMINAL_CHOICES}")

    term = os.environ.get("TERMINAL", "gnome-terminal")
    if not shutil.which(term):
        raise RuntimeError(
            f"terminal backend 'gnome' needs {term} on PATH. Use --terminal inline.")

    # `exec bash` keeps the window open after the command exits so the last
    # error stays readable. The cost is that the shell, not the command, owns
    # the process slot -- terminate() reaches the shell, and the command inside
    # has to be cleaned up separately (scripts/clean.sh does this).
    cmd_keep_open = f"{cmd}; exec bash"

    if term in ("xterm", "uxterm"):
        print(f"[launch] {name}: {term}")
        return subprocess.Popen([term, "-e", "bash", "-c", cmd_keep_open])

    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user and os.geteuid() == 0:
        print(f"[launch] {name}: {term} as {sudo_user}")
        return subprocess.Popen(
            ["sudo", "-u", sudo_user, "env"] + _sudo_env_passthrough() +
            [term, "--", "bash", "-c", cmd_keep_open])

    print(f"[launch] {name}: {term}")
    return subprocess.Popen([term, "--", "bash", "-c", cmd_keep_open])


def write_child_config(merged_cfg):
    """Dump the merged config to a temp file the child can read.

    Under sudo the file would be root-owned 0600 while the child runs as
    SUDO_USER, so it is chowned and made readable. Returns the path.
    """
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as fp:
        json.dump(merged_cfg, fp)
        cfg_path = fp.name

    _give_to_sudo_user(cfg_path, 0o644)
    return cfg_path
