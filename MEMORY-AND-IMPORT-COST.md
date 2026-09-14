# Import memory cost: findings and remaining work

Written 2026-09-11, from a memory audit of the FlexMeasures docker compose stack, where
`timely_beliefs.beliefs.classes` was consistently the single largest entry in the import ledger
(~297 MB per process boot). All numbers below are measured, with the method given at the bottom so
they can be re-checked rather than trusted.

**Measured against branch `remove-scipy-properscoring-dependencies` (745b5b5), not 4.2.0.** That
branch has already done most of the work. Read the state section before planning anything.

## State: what this branch already fixed

Marginal cost of `import timely_beliefs` over an interpreter that already has pandas loaded:

| version | maxrss | anon | file-backed | pulls in |
|---|---|---|---|---|
| 4.2.0 (released) | 319.5 MB | ~124 MB | ~174 MB | openturns, properscoring, numba, llvmlite, scipy |
| `remove-scipy-properscoring-dependencies` | **143.9 MB** | **66.2 MB** | 78.0 MB | openturns only |

Confirmed absent from `sys.modules` after import on this branch: `properscoring`, `numba`,
`llvmlite`, `scipy`. Vendoring CRPS into `beliefs/crps.py` and replacing scipy's `erfinv` with
stdlib `NormalDist` removed ~175 MB. That is the bulk of the problem, already solved.

The anon/file split matters and is easy to miss: file-backed `.so` text is shared between processes
mapping the same library and is reclaimable under memory pressure, while anonymous pages are neither.
For a service running several processes, anon is the number that hurts. So the honest remaining
figure for openturns is **66 MB anon**, not 144 MB.

## What is left

`import openturns as ot` at `timely_beliefs/beliefs/probabilistic_utils.py:9`, still module-level, and
`openturns>=1.23` still a hard dependency at `pyproject.toml:41`.

`probabilistic_utils` is imported eagerly by `beliefs/classes.py` <- `timely_beliefs/__init__.py`, so
every importer pays it. Only four functions use `ot`:

- `interpret_complete_cdf` (62 lines)
- `probabilistic_nan_mean` (45 lines)
- `multivariate_marginal_to_univariate_joint_cdf` (185 lines)
- `get_mean_belief` (31 lines)

## Task 1: defer the openturns import

Move `import openturns as ot` from module scope into those four functions. I tested exactly this
shape for both openturns and properscoring against 4.2.0: the suite gave 224 passed, 1 failed, and
that failure (`test_viz__plotting.py::test_chart_creation`, an altair deprecation) reproduces
identically unpatched. So it is pre-existing and unrelated.

Effort: small, an hour including the changelog. Risk: very low, no behaviour change, same code paths.

Payoff, and be careful reading this number. In a single process, deferring the import removes 66 MB
anon and 78 MB file-backed. But in a real container the figure is smaller, for two reasons: cgroup
`anon` is not process RSS, and under gunicorn `--preload` the import happens once in the master and
the workers share those pages copy-on-write.

Measured end to end: deferring **both** openturns and properscoring in a FlexMeasures container moved
idle cgroup `anon` from 321 to 262 MiB on the server and 288 to 228 MiB on the worker, so about
-59 MiB, n=5, ranges non-overlapping. Since this branch has already removed properscoring, the
openturns-only share of that is roughly a third, call it 20 MiB of container anon, plus the
file-backed pages and ~62 MiB off the boot peak.

So: worth doing, cheap, but do not quote 66 MB as the container win. FlexMeasures never calls these
four functions deliberately (see Task 2 for the exception that currently makes it call them anyway).

Cost to be honest about: the first call to any of the four now pays the openturns import (~0.15 s)
instead of paying it at process start. For a long-running service that is a good trade.

## Task 2: `resample_events` reaches openturns for deterministic data

This is a latency bug, independent of the memory work, and it is a **prerequisite** for Task 3.

Downsampling a BeliefsDataFrame with more than one source is ~50x slower than the same data with one
source, even when every belief is a single deterministic point, because openturns builds a joint
distribution over a product space in order to average point values.

Measured on 4.2.0, deterministic beliefs, one belief time, 15-minute events resampled to hourly:

| data | `resample_events(1h)` | with `keep_only_most_recent_belief=True` |
|---|---|---|
| 96 events, 1 source | 7.2 ms | 5.4 ms |
| 96 events, 2 sources | 354.8 ms | 325.8 ms |
| 192 events, 1 source | 5.6 ms | 5.4 ms |
| 192 events, 2 sources | 771.3 ms | 655.0 ms |

cProfile on the 96-event two-source case: 0.536 s total, 0.495 s inside `probabilistic_utils`, and
**19,200 calls** to `DistributionImplementation_computeQuantile`. For 96 deterministic values.

Cause: the fast track at `classes.py:1832-1841` requires all three of

1. `number_of_beliefs == number_of_events`
2. `keep_only_most_recent_belief` or upsampling or `number_of_belief_times == 1`
3. `number_of_sources == 1`

Condition 3 is the one that bites. Two sources drops you to the slow track regardless of anything
else, into `belief_utils.resample_event_start` -> `join_beliefs` -> `probabilistic_nan_mean`, which
unconditionally builds `ot.UserDefined` marginals and an `ot.IndependentCopula`/`ot.JointDistribution`
without ever checking whether the beliefs are actually probabilistic. For single-row deterministic
groups the whole construction collapses to `np.nanmean`.

Worth stating explicitly because it is the obvious thing to reach for: `keep_only_most_recent_belief=True`
does **not** rescue this. It only satisfies condition 2.

Repro:

```python
import pandas as pd
from datetime import timedelta
import timely_beliefs as tb

s = tb.Sensor("t", event_resolution=timedelta(minutes=15))
srcs = [tb.BeliefSource(f"s{i}") for i in range(2)]
start = pd.Timestamp("2026-01-01T00:00:00+00:00")
b = [tb.TimedBelief(sensor=s, source=src, event_start=start + i*timedelta(minutes=15),
                    belief_time=start, event_value=float(i))
     for i in range(96) for src in srcs]
bdf = tb.BeliefsDataFrame(b, sensor=s)
bdf.resample_events(timedelta(hours=1))   # ~355 ms, 19200 openturns quantile calls
```

Suggested fix: short-circuit `probabilistic_nan_mean` to `np.nanmean` when every group it receives is
a single deterministic value, leaving the slow path intact for genuinely probabilistic beliefs. I have
not written it, so I do not know whether the cleaner place is inside `probabilistic_nan_mean` or one
level up in `join_beliefs` where the groups are still visible.

An alternative is relaxing condition 3 so multi-source deterministic data takes the fast track. I like
that less: the fast track's cross-source aggregation semantics would need thinking about, whereas
short-circuiting a mean is locally obvious and cannot change results.

Whichever way: the test should assert the openturns path is **not taken**, not merely that the numbers
match, or it will silently regress.

Effort: fix is small, the test is the real work. Half a day. Risk: medium, it touches the path genuine
probabilistic users depend on, mitigated by the existing 254-line `test_probabilistic_downsampling.py`.

Aside: `ot.UserDefined` emits `class UserDefined is deprecated in favor of FiniteDiscreteDistribution`
on every call, so that call site wants updating regardless.

## Task 3: make openturns an optional extra

Only after Task 2. The repo already has `viz` and `forecast` extras, so a `probabilistic` extra fits
the existing convention rather than inventing one.

Do not do this before Task 2. Today deterministic data reaches openturns through `resample_events`, so
an uninstalled extra would surface as an `ImportError` at call time in production, which is a worse
failure mode than the import-time cost it removes. Task 1 alone captures the memory win with none of
this risk, so Task 3 is about disk and image size (openturns is ~238 MB installed) rather than RSS.

This one is breaking: anyone calling the probabilistic functions without declaring the extra breaks.
It needs a major version bump, a loud changelog entry, and ideally a friendly `ImportError` in those
four functions telling people which extra to install.

## What not to do: replacing openturns

I looked at this and recommend against it. Recording why, so it is not re-litigated:

- **Selective submodule import does not help.** Measured: `import openturns`, `openturns.dist`,
  `openturns.model_copula`, and both together all cost ~115 MB. SWIG backs every submodule with one
  shared library.
- **Arbitrary copulas are public API, not an implementation detail.** `copula` is a parameter of
  `multivariate_marginal_to_univariate_joint_cdf` (`probabilistic_utils.py:136`), and the test suite
  exercises it with `ot.NormalCopula(R)` at `test_probabilistic_downsampling.py:99,113,177,190`. Any
  replacement either drops that support (API break) or reimplements Gaussian copulas too.
- **The alternatives lose.** `statsmodels.distributions.copula` means swapping one heavy dependency
  (~97 MB) for another. `copulas` (SDV) is small and pure-numpy but has no exact joint-CDF-over-a-
  product-space or `computeQuantile` equivalent, so the 185-line function is still a rewrite. scipy
  has the univariate pieces and no copula abstraction at all.
- **The scope is ~323 lines** of copula, joint-CDF and quantile-inversion logic, with an exact branch
  for dim<=3 and a Monte-Carlo branch above it. Tails and edge cases are exactly where a
  reimplementation goes subtly wrong, and wrong statistics is the worst failure class here.

Task 1 removes the cost for everyone who does not call these functions, which is the common case. That
is a much better trade than rewriting a working C++ uncertainty-quantification library.

## Method, so these numbers can be checked

Import cost: fresh subprocess per sample, n=5, median reported, measuring `ru_maxrss` delta and
`/proc/self/smaps_rollup` `Pss_Anon`/`Pss_File` deltas around the import, with pandas already imported
as the baseline (that is the realistic baseline for a consumer). Attribution to a specific dependency
was done by inserting a dummy module into `sys.modules` before the import and re-measuring, so the
numbers are causal rather than order-dependent.

Resampling timings: `time.perf_counter` around `resample_events` on synthetic deterministic beliefs,
plus `cProfile` sorted by cumulative time for the call breakdown.

For context on why this mattered: in the FlexMeasures container, `import
flexmeasures.data.models.time_series` dropped from 439 MB to 170 MB peak RSS with openturns and
properscoring both deferred, with all four of openturns, properscoring, numba and llvmlite absent from
`sys.modules`.

## Suggested order

1. Task 1 (defer openturns). Small, low risk, captures the remaining anon cost.
2. Task 2 (resample slow path). Independent value, and unblocks Task 3.
3. Task 3 (optional extra). Only after 2, and accept it as a breaking change.

Tasks 1 and 2 are independent of each other and can go in either order or in parallel.
