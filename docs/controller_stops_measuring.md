# When the controller stops measuring and nothing notices

A 32-node real-Mininet run reached step 591 with a moving reward curve and a
saved model while the newest link measurement it had seen was 85 minutes old.
The training loop never noticed, because nothing in it looks.

This note covers two separate failures with that same symptom, found on the same
machine within a few hours. One is understood, reproduced, and fixed. The other
is measured and still unexplained, and is written down here so the next person
recognises it rather than rediscovering it.

## The symptom, and why the obvious signals miss it

A wedged run looks healthy from every angle anyone normally checks. The step
count advances, checkpoints are written, and the reward curve keeps moving —
reward follows the action, so it goes on responding to the policy long after the
link measurements behind it stopped changing. One frozen run produced 40 distinct
reward values over 40 frozen steps.

MLU pinned at 1.0 is a hint, not proof: a policy that never learns to relieve
saturation produces the same flat line.

What is decisive is whether measurements were still being written:

```bash
date; stat -c %y results/<alg>/net_info_directed.csv
```

Completed runs also archive `measurement.txt`, giving the cycle count and how
long before the end the last measurement landed. Anything past a monitoring
period means the agent was reading a stale file.

## Failure A: a torn read kills the monitor greenthread

**Proven, reproduced, fixed.**

The controller and the DRL agent are separate processes that talk through four
files in `results/<alg>/`:

| File | Direction | Size |
| --- | --- | --- |
| `drl_paths.json` | agent → controller | ~88 KB, rewritten per step |
| `paths_metrics.json` | controller → agent | ~1.3 MB, rewritten per cycle |
| `net_info.csv` | controller → agent | ~2 KB |
| `net_info_directed.csv` | controller → agent | ~6 KB |

Every one of them was written with `open(path, "w")`, which truncates the file
immediately and then flushes in 8 KB blocks. For the whole duration of a write
the file on disk is a prefix of the new content, and a reader cannot tell that
from a complete file.

The controller died on exactly that:

```
File "utils/simple_monitor.py", line 394, in get_dRL_paths
    paths_dict = json.load(json_file)
json.decoder.JSONDecodeError: Expecting value: line 815 column 7 (char 8192)

During handling of the above exception, another exception occurred:
  File "utils/simple_monitor.py", line 109, in monitor
  File "utils/simple_monitor.py", line 403, in get_dRL_paths
json.decoder.JSONDecodeError: Expecting ',' delimiter: line 4861 column 6 (char 49129)
```

`char 8192` is two buffer flushes in — the reader caught the file between blocks.

`get_dRL_paths` already retried once after `time.sleep(0.35)`, and the retry was
not itself guarded, so when the second read also landed inside a write the
exception escaped into the monitor greenthread. `ryu.lib.hub` prints the
traceback and lets the greenthread end. Nothing else in the controller is
affected: the event loops, the receive loops, the delay detector and the
topology worker all stayed healthy, the process kept using CPU, and the
monitoring cycle simply never ran again. Metrics froze at cycle 85 and the agent
trained on for as long as it was left running.

The retry, and the commented-out `except ValueError as e: #error exception when
trying to read the json and is still been updated` beside it, say plainly that
this race was known and worked around rather than removed.

How wide the window is, one process rewriting the file while another reads it
for six seconds each way:

```
plain open('w')   152,782 reads   152,755 torn   (100.0%)
atomic replace     27,391 reads         0 torn   (  0.0%)
```

### The fix

`utils/atomic_io.py` writes to a temporary file in the target's own directory
and `os.replace()`s it over the top. That rename is atomic within a filesystem,
so a reader sees either the whole previous version or the whole new one. All
nine write sites across `utils/manager.py` and `loader/train_loader.py` now go
through it.

Two follow-on changes, because the cause being fixed is not the same as the
failure being survivable:

- `get_dRL_paths` no longer raises. If a read fails anyway it logs and reuses
  the previous cycle's routing. Repeating one cycle's paths is a far smaller
  error than never measuring again.
- The monitor loop body moved into `_monitor_cycle`, and `monitor` catches
  exceptions around it, logs the traceback with a consecutive-failure count, and
  continues. A greenthread that raises is gone for good, and this is the third
  recorded instance of that class of failure — the comment above
  `MONITOR_START_DELAY` records an earlier one, a `KeyError` on a partial
  topology.

Atomic replacement also removes the `os.remove()`-then-recreate dance that two
call sites used to work around a root-owned `drl_paths.json` left by a prior
sudo run: renaming a fresh file over the target needs write permission on the
directory, not on the file.

## Failure B: the receive loops stop, and we do not know why

**Measured, not explained.** Same machine, a few hours earlier, and *not* the
same as Failure A — here the monitor greenthread was alive and printing cycles
601, 602, 603 while the port-stats warning fired every time:

```
[monitor] cycle 601: 0/32 switches returned port stats after 3.0s
```

Zero, not a partial return, and zero on every cycle. What the process looked
like:

| Measurement | Value | How |
| --- | --- | --- |
| Unread data on the OpenFlow sockets | 26.5 MB, ~830 KB each | `ss -tn state established '( sport = :6653 )'`, Recv-Q |
| Drain rate | 0 B/s over 18 s | two Recv-Q samples |
| Sockets in the hub's epoll set | 2 of 34 — both listeners | `/proc/<pid>/fdinfo/<epollfd>`, `tfd:` lines |
| Process state | `S (sleeping)`, `wchan = ep_poll` | `/proc/<pid>/wchan` |
| Controller CPU | 27 % of one core | `top -bn2 -p <pid>` |
| Machine load | 1.97 on 32 cores | `/proc/loadavg` |

Eventlet watches a descriptor only while some greenthread is waiting to read it.
The 32 switch connections being absent from the epoll set means no greenthread
was reading any of them — the hub was not overloaded, it had nothing registered
to wake up for. Meanwhile the controller still *wrote* ~40 KB/s of flow
modifications into those same sockets, and `[Flow Installation Ok]` still
printed, being an unconditional `print` rather than a confirmation.

The natural explanation is Ryu's backpressure. Each app has one greenthread
behind a bounded queue, and delivery blocks the producer once it is full:

```python
# ryu/base/app_manager.py:160
self.events = hub.Queue(128)
self._events_sem = hub.BoundedSemaphore(self.events.maxsize)

# ryu/base/app_manager.py:301
def _send_event(self, ev, state):
    self._events_sem.acquire()
    self.events.put((ev, state))
```

The producers are the per-datapath receive loops, so one app that stops draining
would stop every socket, permanently, because whatever it is waiting for would
have to arrive over the sockets nobody is reading.

**That explanation is unverified.** The dump tool did not exist yet during this
incident, and when the next failure was captured with it, every queue read
`0 / 128` — that turned out to be Failure A, a different thing. So the queue
theory has never actually been observed, and Failure B may be something else
entirely. If it recurs, `kill -USR1` settles it in one shot.

### The handler rule, and a change made on its own merits

While chasing Failure B, `get_topology` in `utils/simple_awareness.py` turned
out to be a handler that blocks. It called `get_switch()` and `get_link()`,
which are not local lookups but synchronous requests into the `Switches` app:

```python
# ryu/topology/api.py:20
def get_switch(app, dpid=None):
    rep = app.send_request(event.EventSwitchRequest(dpid))
    return rep.switches

# ryu/base/app_manager.py:265
def send_request(self, req):
    req.sync = True
    req.reply_q = hub.Queue()
    self.send_event(req.dst, req)
    return req.reply_q.get()          # blocks
```

Ryu's own documentation forbids this, in `doc/source/ryu_app_api.rst` under
*Threads, events, and event queues*:

> Because the event handler is called in the context of the event processing
> thread, it should be careful when blocking. While an event handler is
> blocked, no further events for the Ryu application will be processed.

Being precise about what upstream does *not* say: Ryu never names "synchronous
call from inside a handler" as a deadlock. The word `deadlock` appears exactly
once in the tree, a few lines further down the same file, and it is about the
opposite — why replies get their own queue:

> While such requests uses the same machinary as ordinary events, their replies
> are put on a queue dedicated to the transaction to avoid deadlock.

That dedicated reply queue stops the reply getting stuck behind the caller's own
backlog. It does nothing about the callee being the app that is behind.
Combining the documented handler rule with the bounded-queue backpressure is a
conclusion drawn here, not a warning quoted from upstream.

The handler is now the whole of it:

```python
@set_ev_cls(events)
def get_topology(self, ev):
    self._topo_dirty = True
```

and `_topology_worker` does the rebuild on a greenthread of its own, every
`setting.TOPO_REBUILD_PERIOD` (0.5 s). The previous handler body is unchanged as
`_rebuild_topology`. An AST pass over every `@set_ev_cls` handler in `utils/`
reports all ten clean; `get_topology` was the only offender.

**This did not fix the failure that was actually observed next.** It is a real
rule violation and worth removing, and the coalescing is a genuine reduction in
cross-app request traffic — discovery's burst of `LinkAdd`/`PortAdd` events used
to trigger one full rebuild each, every rebuild issuing up to a hundred requests
— but the run that followed it died of Failure A, and Failure B has not recurred
to say whether this addressed it.

## Why the paper machine never hit either

Every result in the paper comes from PC0 (i7-13700K): 26 archived real-Mininet
runs, 24 of them 32-node, about 78,000 steps, `sim_training = False` throughout.
The archived outputs were checked for constant-MLU stretches and for measurement
files older than the run; they are clean.

PC0 was not immune to Failure A — it ran the same racy writes. It almost
certainly hit the torn read repeatedly and recovered, because the single retry
was enough: the second read lands 0.35 s later, by which time an 88 KB write on
that machine is long finished. The run that died needed *both* reads to land
inside a write, which takes a writer slow enough, or busy enough, to still be
writing a third of a second later.

For Failure B the difference is less certain, since the mechanism is unknown.
What can be said is what was measured: the wedged machine was a shared Xeon
Silver 4110 (2.1 GHz base, 3.0 turbo, ten login sessions, other users' jobs)
against PC0's i7-13700K boosting to ~5.4 GHz. Ryu is single-threaded, so
anything queue-depth-related tracks single-core speed, roughly a 2.5–3×
difference on exactly the thing that drains.

Note what is *not* on that list: total CPU capacity. The wedged machine had 32
cores at load 2. Both failures happened with every resource available, which is
why "run it somewhere bigger" is not a fix for either.

The 32-node topology matters more than the machine. It produces far more
LLDP packet-in traffic and far larger shared files than Geant's 23 nodes. The
two Geant runs in the archive were never close.

## Diagnosing the next one

**`kill -USR1`.** The controller arms `utils/greenthread_dump.py` at startup:

```
kill -USR1 $(pgrep -f 'ryu-manager --observe-link')
```

It prints every app's queue depth — the one at its maximum is the one that
stalled — and writes the stack of every greenthread to
`artifacts/greenthread_dump_<pid>_<timestamp>.txt`. A greenthread that is
*missing* from the dump is one that died; that is how Failure A was identified,
by there being no `simple_monitor.monitor` frame anywhere among the 114.

Only the queue table goes to the terminal. The first time this ran it printed
1,600 lines of stacks and pushed the traceback that explained the failure out of
the 2,000-line scrollback, which is also why pane output is now written to
`results/_terminal_logs/<window>_<timestamp>.log` by the tmux launcher.

**py-spy cannot diagnose any of this**, and it is worth knowing before reaching
for it mid-outage. Greenthreads are not OS threads — the process has exactly one
— so py-spy reports whichever greenthread is running, the hub in `epoll_wait`,
which `/proc/<pid>/wchan` gives for free. It has no greenlet support, and
attaching needs root wherever `kernel.yama.ptrace_scope` is 1.
