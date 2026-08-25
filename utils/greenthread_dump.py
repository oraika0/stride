"""On-demand dump of Ryu's event queues and greenthread stacks.

Ryu runs each application on a single greenthread fed by a bounded queue
(``hub.Queue(128)``, ryu/base/app_manager.py). When one application stops
draining its queue, the producers block -- and the producers are the
per-datapath receive loops. The controller then stops reading every switch
socket while remaining alive: the process sits in ``epoll_wait``, CPU is low,
nothing crashes, and the training loop happily keeps reading the last metrics
file that was written. See docs/ryu_controller_deadlock.md.

py-spy cannot diagnose this. Greenthreads are not OS threads, so it only ever
shows whichever greenthread happens to be running -- the hub, in epoll_wait --
which is the one thing already visible from /proc/<pid>/wchan.

So the controller installs a SIGUSR1 handler instead:

    kill -USR1 $(pgrep -f 'ryu-manager --observe-link')

Same user, no root, no external tools. It prints every application's queue
depth (the one sitting at 128 is the one that stalled) and the stack of every
greenthread (which says what it stalled on). Output goes to stderr and to
artifacts/greenthread_dump_<pid>_<timestamp>.txt, because the terminal
scrollback is usually far too short to still hold the interesting cycle.
"""

import datetime
import gc
import os
import signal
import sys
import traceback

from project_root import PROJECT_ROOT


def _queue_report(out):
    try:
        from ryu.base.app_manager import SERVICE_BRICKS
    except Exception as exc:
        out.append("  <app_manager unavailable: %r>" % (exc,))
        return

    out.append("event queues")
    for name, app in sorted(SERVICE_BRICKS.items()):
        queue = getattr(app, "events", None)
        if queue is None:
            continue
        try:
            size = queue.qsize()
            cap = getattr(queue, "maxsize", 0) or 0
        except Exception as exc:
            out.append("  %-26s <unreadable: %r>" % (name, exc))
            continue
        note = ""
        if cap and size >= cap:
            note = "   <== FULL, this app stalled and is blocking its producers"
        elif cap and size * 2 >= cap:
            note = "   <== filling"
        out.append("  %-26s %4d / %-5d%s" % (name, size, cap, note))


def _greenthread_report(out):
    try:
        import greenlet
    except ImportError as exc:
        out.append("  <greenlet unavailable: %r>" % (exc,))
        return

    current = greenlet.getcurrent()
    threads = [obj for obj in gc.get_objects()
               if isinstance(obj, greenlet.greenlet)]
    out.append("greenthreads: %d" % len(threads))

    for index, thread in enumerate(threads):
        if thread is current:
            state, frame = "running", sys._getframe()
        elif getattr(thread, "dead", False):
            state, frame = "dead", None
        else:
            state, frame = "suspended", thread.gr_frame
        out.append("")
        out.append("  [%d] %r  %s" % (index, thread, state))
        if frame is None:
            out.append("      <no frame>")
            continue
        for chunk in traceback.format_stack(frame):
            for line in chunk.rstrip("\n").split("\n"):
                out.append("      " + line)


def _dump(signum, frame):
    stamp = datetime.datetime.now()
    header = ["=" * 78,
              "greenthread dump   pid %d   %s" % (os.getpid(),
                                                  stamp.strftime("%Y-%m-%d %H:%M:%S")),
              "=" * 78]
    queues, stacks = [], []
    try:
        _queue_report(queues)
        _greenthread_report(stacks)
    except Exception:
        stacks.append(traceback.format_exc())

    path = os.path.join(PROJECT_ROOT, "artifacts",
                        "greenthread_dump_%d_%s.txt"
                        % (os.getpid(), stamp.strftime("%Y%m%d_%H%M%S")))
    written = None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write("\n".join(header + queues + [""] + stacks) + "\n")
        written = path
    except OSError as exc:
        stacks.insert(0, "[dump] could not write the file: %s" % exc)

    # Only the queue table goes to the terminal. The stacks are hundreds of
    # lines and the controller's scrollback holds two thousand, so printing
    # them here would push out whatever traceback led you to ask for the dump
    # -- which is exactly what happened the first time this ran.
    summary = header + queues
    if stacks:
        summary.append(stacks[0])   # the "greenthreads: N" line
    summary.append("[dump] full stacks in %s" % (written or "<not written>"))
    sys.stderr.write("\n".join(summary) + "\n")
    sys.stderr.flush()


def install(sig=signal.SIGUSR1):
    """Arm the dump. Returns whether the handler was accepted."""
    try:
        signal.signal(sig, _dump)
    except (ValueError, OSError) as exc:
        # ValueError: not the main thread. Not fatal, just no diagnostic.
        sys.stderr.write("[dump] SIGUSR1 handler not installed: %s\n" % exc)
        return False
    sys.stderr.write("[dump] kill -USR1 %d dumps queue depths and greenthread "
                     "stacks\n" % os.getpid())
    sys.stderr.flush()
    return True
