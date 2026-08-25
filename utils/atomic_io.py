"""Whole-file replacement for the files the controller and the agent share.

The Ryu controller and the DRL agent are separate processes that talk to each
other through four files under ``results/<alg>/``:

    drl_paths.json          agent -> controller   (~88 KB, rewritten per step)
    paths_metrics.json      controller -> agent   (~1.3 MB, rewritten per cycle)
    net_info.csv            controller -> agent
    net_info_directed.csv   controller -> agent

``open(path, "w")`` truncates the file at once and then flushes in 8 KB blocks,
so for the whole duration of a write the file on disk is a prefix of the new
content -- and the reader on the other side has no way to tell that from a
complete file. It reads whatever is there and fails to parse it.

That is not theoretical. A 32-node run died this way: the controller read
``drl_paths.json`` mid-write, ``json.load`` raised ``Expecting value: line 815
column 7 (char 8192)`` -- char 8192 being exactly two buffer flushes in -- the
exception escaped the monitor greenthread, and ryu.lib.hub printed a traceback
and let the greenthread end. Metrics froze at cycle 85 while every other part
of the controller stayed healthy, and the agent trained on for as long as it
was left running against a file nobody was updating.

``os.replace()`` is atomic within one filesystem, so writing to a temporary file
beside the target and renaming it over the top gives readers either the whole
previous version or the whole new one, never a prefix of either.
"""

import contextlib
import json
import os
import tempfile
import time


@contextlib.contextmanager
def replacing(path, newline=""):
    """Yield a writable handle whose content replaces `path` atomically.

    The temporary file is created in the target's own directory, because
    os.replace is only atomic within a filesystem. If the body raises, the
    temporary file is removed and `path` keeps its previous content.
    """
    path = os.path.abspath(path)
    directory = os.path.dirname(path) or "."
    handle_fd, temporary = tempfile.mkstemp(
        dir=directory, prefix=".%s." % os.path.basename(path), suffix=".tmp")
    try:
        with os.fdopen(handle_fd, "w", newline=newline) as handle:
            yield handle
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def dump_json(path, obj, **kwargs):
    """json.dump to `path`, atomically. Defaults to indent=2 like the callers."""
    kwargs.setdefault("indent", 2)
    with replacing(path) as handle:
        json.dump(obj, handle, **kwargs)


def load_json(path, attempts=5, delay=0.2):
    """Read JSON that another process rewrites, retrying a torn or missing read.

    Writers using this module make a torn read impossible, so in a correctly
    updated tree this never retries. It stays here because the readers are not
    all in this repository's control at once -- an older writer, a hand-edited
    file, a partially copied results directory -- and because the caller that
    matters most is the controller's monitor greenthread, where an exception is
    not an error but the end of measurement for the rest of the run.
    """
    failure = None
    for attempt in range(attempts):
        try:
            with open(path) as handle:
                return json.load(handle)
        except (ValueError, OSError) as exc:
            failure = exc
            if attempt + 1 < attempts:
                time.sleep(delay)
    raise failure
