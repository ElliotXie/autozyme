# autozyme

Drop-in accelerators for scientific Python packages. Each accelerator is a validated, monkey-patched fast replacement for a hot function in an upstream library (scanpy, scvelo, etc.). Install once; existing scripts run faster automatically.

## Install

```bash
pip install autozyme
```

`autozyme` itself has no hard dependencies on upstream packages. A patch activates only when its upstream is importable in your environment.

## Use

```python
import autozyme
# autozyme 0.3.0: 16 patches available, 4 subsets — autozyme.activate(<name>) to enable

autozyme.activate("scvelo")
import scvelo as scv
scv.tl.recover_dynamics(adata)   # uses fast version transparently
```

### Supported parameter scope

Each accelerator is validated for a specific parameter envelope. Supported
parameters (`n_comps`, resolution, the data itself, …) can be set freely; a
handful of parameters per patch are not on the fast path and **fall back to the
upstream implementation automatically** (correct result, just not accelerated).
A few documented approximations are validated only at their stated
configuration. Each patch ships its own `SCOPE.md` in its package directory
(`src/autozyme/<patch>/SCOPE.md`) documenting its supported scope and
out-of-scope behavior.

### Opt-out levels

```python
# per-call (only namespace-fn patches; class-method patches don't expose this)
fn(..., zyme=False)

# per-block (works for ALL patches incl. class-method — hard kill switch)
with autozyme.disabled():
    fn(...)        # uses captured original
    other_fn(...)  # also original

# per-package
autozyme.restore("scvelo")
autozyme.activate("scvelo")

# session-wide
autozyme.restore_all()

# never activate (set before import; `activate()` returns False without binding)
import os; os.environ["AUTOZYME_DISABLED"] = "1"

# suppress per-activation stderr marker (keeps logs clean; banner unaffected)
import os; os.environ["AUTOZYME_QUIET"] = "1"
```

### Multiprocessing workers

`activate()` binds patches into the upstream namespace of the **current process**. Fork-mode workers (default on Linux for `multiprocessing`) inherit the bindings via copy-on-write memory. Spawn-mode workers (default on macOS / Windows, and what `loky`, `joblib`, `concurrent.futures.ProcessPoolExecutor(mp_context="spawn")` use) start fresh — they import autozyme but no patch is active.

For spawn-mode pools, activate inside each worker via the pool's `initializer`:

```python
from multiprocessing import Pool

def _init():
    import autozyme
    autozyme.activate("cell2location")

with Pool(4, initializer=_init) as pool:
    pool.map(work_fn, items)
```

`joblib.Parallel(backend="loky")` uses the same pattern via the `initializer=` arg on its underlying executor.

### Threading

```python
autozyme.set_threads(8)
```

## Status and introspection

```bash
python -m autozyme              # one-screen dashboard of patches + upstream availability
```

```python
autozyme.list_patches()                   # ['cell2location', 'obspy', 'prody', ...]
autozyme.list_patches(installed=True)     # only patches whose upstream is importable
autozyme.status()                         # {'cell2location': 'inactive', 'prody': 'active', ...}
autozyme.inspect("prody")                 # detailed binding info for one patch
autozyme.env_snapshot()                   # structured dict for provenance / Methods capture
```

## Citation

`autozyme` releases on GitHub get a Zenodo DOI. Cite the version + the patch name(s) you activated. The activation marker logged to stderr is the canonical provenance record, e.g.

```
[autozyme] activated cell2location -> 2 target(s) in cell2location 0.1.5
```

In Methods: "We used autozyme v0.3.0 with the `cell2location` patch (tested against cell2location 0.1.5)."

## License

MIT
