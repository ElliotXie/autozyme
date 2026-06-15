# autozyme `obspy` (Python) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `obspy.core.stream.Stream.filter`

- **In-scope output equivalence:** tolerance
- **Validated at:** `Stream.filter("bandpass", freqmin=1.0, freqmax=10.0, corners=4, zerophase=True) applied to <data> (synthetic float32 white-noise Streams; all tiers identical filter_kwargs); the timed body repeats the same call n_repeats times (3 for dev tiers, 15 for ood_xlarge) in place on the same Stream`
- **Supported scope:** Three coordinated patches. (1) Stream.filter (fast_stream_filter) is filter-type-agnostic: it simply dispatches tr.filter(type, *args, **options) per Trace across a ThreadPoolExecutor (workers = min(len(self), auto_threads(cap=24))), falling to a serial loop when workers<=1 or len(self)<=1. Any filter type, args, and options are forwarded unchanged, so bandstop/lowpass/highpass/lowpass_cheby_2/etc. all still work correctly (just routed to upstream per-trace functions, with only the Stream-level parallelism as the speedup). (2) obspy.signal.filter.bandpass (fast_bandpass) is a line-for-line reimplementation of upstream bandpass — same Nyquist edge logic, same highpass fallback when freqmax>=Nyquist (warns), same ValueError when low corner>Nyquist, same sosfilt / zerophase np.flip path along arbitrary axis — with the ONLY change being that the iirfilter SOS-coefficient design is memoized via lru_cache keyed on (corners, (freqmin,freqmax), df, rp, rs, 'band', ftype). This is bit-exact to upstream for any combination of corners/freqmin/freqmax/df/rp/rs/ftype/zerophase/axis (benchmark reports max_abs_diff=0.0, pearson=1.0). (3) obspy.core.trace._get_function_from_entry_point is wrapped in lru_cache to amortize per-call entry-point resolution. Correctly handles: bandpass with any well-formed numeric params; all other filter types via passthrough; single- and multi-trace Streams.
- **Out-of-scope behavior:** Handles the **full parameter signature**.

