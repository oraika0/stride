#!/usr/bin/env python
"""Apply the LLDP delay patch to the installed Ryu, or report that it is applied.

    python scripts/patch_ryu.py            # apply, then verify
    python scripts/patch_ryu.py --check    # verify only, exit 1 if unpatched
    python scripts/patch_ryu.py --revert   # put the original file back

WHY THIS EXISTS

The controller measures per-link latency from LLDP round-trip timing, which
upstream Ryu records nowhere: PortData timestamps when a frame was *sent*, and
lldp_packet_in_handler never notes when the reply came back. Two small edits fix
that, and both have to be made inside the installed package.

Doing it by hand is how it went wrong before. A missed edit raises nothing --
every link delay simply reads 0, training optimises against a constant, and the
run looks healthy the whole way. So this applies the edits, verifies them by
importing the patched module, and refuses to guess when the file does not look
the way it expects.

WHAT IT CHANGES

1. PortData.__init__ gains `self.delay = 0`, the field the second edit fills.
2. lldp_packet_in_handler takes a receive timestamp on entry, and just before it
   resolves the source port, subtracts the send timestamp to get the one-way
   delay for that port.

Idempotent: running it twice is a no-op. The original is kept next to the file as
`switches.py.orig` the first time only, so --revert always restores the untouched
version rather than a previously patched one.
"""

import argparse
import importlib
import inspect
import os
import shutil
import sys

# (anchor, insertion, name) -- the anchor is matched once, and the insertion is
# placed after it at the same indentation.
EDITS = [
    (
        # `self.sent = 0` alone appears twice -- also in lldp_received -- so the
        # anchor carries the two lines above it to pin PortData.__init__.
        "        self.timestamp = None\n"
        "        self.sent = 0\n",
        "        self.delay = 0\n",
        "PortData.delay field",
    ),
    (
        "    def lldp_packet_in_handler(self, ev):\n",
        "        recv_timestamp = time.time()\n",
        "receive timestamp",
    ),
    (
        "        src = self._get_port(src_dpid, src_port_no)\n",
        "        for port in self.ports.keys():\n"
        "            if src_dpid == port.dpid and src_port_no == port.port_no:\n"
        "                send_timestamp = self.ports[port].timestamp\n"
        "                if send_timestamp:\n"
        "                    self.ports[port].delay = recv_timestamp - send_timestamp\n",
        "delay computation",
    ),
]


def switches_path():
    try:
        import ryu.topology.switches as s
    except ImportError as exc:
        # Distinguish "no ryu here" from "ryu is here but broken". The second is
        # the common one: the last PyPI release predates eventlet 0.30.3 and dies
        # on ALREADY_HANDLED, and reporting that as "not installed" sends people
        # off to install it a second time.
        try:
            import ryu                                    # noqa: F401
        except ImportError:
            sys.exit("ryu is not installed in this interpreter. "
                     "Activate the env first, or use its python by full path.")
        hint = ("\nThat is the PyPI release, which predates eventlet 0.30.3. "
                "Install ryu from git instead -- see README section 2.3."
                if "ALREADY_HANDLED" in str(exc) else "")
        sys.exit(f"ryu is installed but fails to import: {exc}{hint}")
    return s.__file__


def is_patched():
    """True when the running interpreter imports a patched Ryu."""
    for mod in [m for m in sys.modules if m.startswith("ryu")]:
        del sys.modules[mod]
    s = importlib.import_module("ryu.topology.switches")
    src_init = inspect.getsource(s.PortData.__init__)
    src_handler = inspect.getsource(s.Switches.lldp_packet_in_handler)
    return ("self.delay" in src_init
            and "recv_timestamp" in src_handler
            and "delay = recv_timestamp" in src_handler)


def apply_patch(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()

    applied = []
    for anchor, insertion, name in EDITS:
        if insertion.lstrip() in text.replace(" " * 4, " " * 4):
            marker = insertion.strip().splitlines()[0]
            if marker in text:
                print(f"  already present: {name}")
                continue
        n = text.count(anchor)
        if n != 1:
            sys.exit(f"cannot patch: the anchor for {name!r} appears {n} times, "
                     f"expected once. This Ryu is not the 4.34 the patch was "
                     f"written against; apply the edits by hand (README 2.3).")
        # Edit 3 goes BEFORE its anchor, the other two after.
        if name == "delay computation":
            text = text.replace(anchor, insertion + anchor, 1)
        else:
            text = text.replace(anchor, anchor + insertion, 1)
        applied.append(name)

    if not applied:
        return []

    backup = path + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"  original saved to {backup}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return applied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only")
    ap.add_argument("--revert", action="store_true", help="restore the original")
    args = ap.parse_args()

    path = switches_path()
    print(f"ryu/topology/switches.py: {path}")

    if args.revert:
        backup = path + ".orig"
        if not os.path.exists(backup):
            sys.exit("no switches.py.orig next to it -- nothing to revert to.")
        shutil.copy2(backup, path)
        print("  reverted")
        print("  verified:", "NOT PATCHED" if not is_patched() else "STILL PATCHED (?)")
        return

    if args.check:
        ok = is_patched()
        print("  " + ("PATCHED" if ok else "NOT PATCHED"))
        sys.exit(0 if ok else 1)

    if is_patched():
        print("  already patched, nothing to do")
        return

    if not os.access(path, os.W_OK):
        sys.exit(f"no write permission on {path}. Ryu should live in your own "
                 f"conda env, not a system directory -- see README 2.2.")

    applied = apply_patch(path)
    for name in applied:
        print(f"  applied: {name}")

    if is_patched():
        print("  verified: PATCHED")
    else:
        sys.exit("edits were written but the import still looks unpatched. "
                 "Inspect the file by hand.")


if __name__ == "__main__":
    main()
