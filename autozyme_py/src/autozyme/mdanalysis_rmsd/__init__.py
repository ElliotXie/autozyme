"""Patch for MDAnalysis RMSD.run (trajectory RMSD).

Lifted from autozyme task `test_mdanalysis`. Four coordinated targets on
the QCP-based RMSD pipeline:

  - ``MDAnalysis.analysis.rms.RMSD._single_frame`` -> ``fast_single_frame``:
        per-frame hot path. Caches the mobile selection's atom indices,
        reuses an f32 scratch buffer for ``np.take(out=...)`` (avoids the
        per-frame allocator churn that ``AtomGroup.positions[ix]`` does),
        skips the secondary-RMSD branch when ``_groupselections_atoms`` is
        empty (the backbone-only case here).
  - ``MDAnalysis.analysis.rms.RMSD._compute`` -> ``fast_compute``: replaces
        ``AnalysisBase._compute`` with a tight loop that hoists all per-frame
        state to locals (drops ~15 ``self.*`` lookups + the ProgressBar
        wrap). Adds a bulk-read fast path for ChainReader-of-DCDReader:
        ``DCDFile.readframes(indices=...)`` collapses the trajectory
        advance + atom-selection stack into bulk reads. It works for both
        ChainReader-of-DCDReader and a single long DCDReader, and processes
        long files in bounded 8192-frame chunks. The earlier repeated-file
        shortcuts that tiled one segment across duplicated DCD inputs were
        removed because they only accelerated the benchmark construction.
        Time-column population uses Kahan-compensated accumulation when
        needed (n_seg=45000 drifts ~2e-6 with naive cumsum, above the 1e-6
        threshold).
  - ``MDAnalysis.coordinates.DCD.DCDReader._read_next_timestep`` ->
        ``fast_read_next_timestep``: skip the deprecated ``self.ts.copy()``
        (~6.6s cumulative on tiny, ~10%). The ``AnalysisBase`` consumer
        only retains the current ts, so reusing ``self.ts`` in-place is
        safe for sequential analysis.
Universe construction must NOT run with the DCD patches active --
``ChainReader``'s first-pass trajectory inspection at ``mda.Universe(...)``
init time depends on upstream DCD semantics. The smoke recipe wraps Universe
construction in ``autozyme.disabled()`` so the dispatcher short-circuits the
DCD patch to upstream originals during init, then re-enables it for
``R.run()``.
"""
from __future__ import annotations

import io
import os
import warnings

import numpy as np

import MDAnalysis
from MDAnalysis.analysis import rms
from MDAnalysis.coordinates.DCD import DCDReader
from MDAnalysis.lib import qcprot as _qcp

import autozyme
from autozyme._core import register_patch
from autozyme._utils import resolve_dataset_path


# ============================================================
# Capture upstream originals BEFORE register_patch rebinds.
# ============================================================
_orig_single_frame = rms.RMSD._single_frame
_orig_compute = rms.RMSD._compute


# ============================================================
# Fast methods (class-method patches; signatures match upstream).
# ============================================================
def fast_single_frame(self):
    ix = getattr(self, "_zyme_ix", None)
    if ix is None:
        ix = self.mobile_atoms.ix
        self._zyme_ix = ix
        self._zyme_pos_f32 = np.empty((ix.shape[0], 3), dtype=np.float32)
    np.take(self._ts.positions, ix, axis=0, out=self._zyme_pos_f32)
    buf = self._mobile_coordinates64
    buf[:] = self._zyme_pos_f32
    w = self.weights_select
    if w is None:
        mobile_com = buf.mean(axis=0)
    else:
        w_sum = getattr(self, "_zyme_w_sum", None)
        if w_sum is None:
            w_sum = float(w.sum())
            self._zyme_w_sum = w_sum
        mobile_com = np.dot(w, buf) / w_sum
    buf -= mobile_com

    self.results.rmsd[self._frame_index, :2] = (
        self._ts.frame,
        self._trajectory.time,
    )

    if self._groupselections_atoms:
        return _orig_single_frame(self)

    self.results.rmsd[self._frame_index, 2] = _qcp.CalcRMSDRotationalMatrix(
        self._ref_coordinates64,
        buf,
        self._n_atoms,
        None,
        w,
    )


def fast_compute(self, indexed_frames, verbose=None, *, progressbar_kwargs=None):
    frames = indexed_frames[:, 1]
    n = len(frames)
    if n == 0:
        return self
    # Scope guard (additive): the bulk DCD path reads frames sequentially from
    # the start and labels them 0..n-1, which is correct only for a full,
    # in-order run. Any start/stop/step/frames slicing -> defer to upstream
    # _compute (which still dispatches the fast per-frame kernel), so the
    # requested frames and the frame/time columns are read correctly. A full
    # run (frames == 0..n-1) passes through unchanged.
    if not np.array_equal(np.asarray(frames), np.arange(n)):
        return _orig_compute(self, indexed_frames, verbose=verbose,
                             progressbar_kwargs=progressbar_kwargs)
    self._prepare_sliced_trajectory(slicer=frames)
    self._prepare()

    sliced = self._sliced_trajectory
    rmsd = self.results.rmsd
    frames_arr = self.frames
    times_arr = self.times
    has_groups = bool(self._groupselections_atoms)

    if has_groups:
        for idx, ts in enumerate(sliced):
            self._frame_index = idx
            self._ts = ts
            frames_arr[idx] = ts.frame
            times_arr[idx] = ts.time
            self._single_frame()
        self._frame_index = n - 1
        return self

    ix = self.mobile_atoms.ix
    n_atoms = ix.shape[0]
    buf = self._mobile_coordinates64
    w = self.weights_select
    if w is None:
        n_atoms_inv = 1.0 / n_atoms
    else:
        w_inv_sum = 1.0 / float(w.sum())
    ref_coords = self._ref_coordinates64
    nat = self._n_atoms
    qcp_call = _qcp.CalcRMSDRotationalMatrix
    np_take = np.take
    traj = self._trajectory

    # Per-segment frame counts cached once for ChainReader.
    seg_lengths_cache = None
    if hasattr(traj, "readers"):
        if hasattr(traj, "_start_frames"):
            seg_lengths_cache = np.diff(
                np.asarray(traj._start_frames, dtype=np.int64)
            )
        else:
            seg_lengths_cache = np.array(
                [r.n_frames for r in traj.readers], dtype=np.int64
            )
    elif hasattr(traj, "n_frames"):
        seg_lengths_cache = np.array([traj.n_frames], dtype=np.int64)

    # Precompute global cumulative time per frame. Kahan-compensated when
    # segment dts / total_times aren't uniform — naive cumsum drifts ~2e-6
    # at n_seg=45000, above the 1e-6 task threshold.
    times_global = None
    if (
        seg_lengths_cache is not None
        and hasattr(traj, "dts")
        and hasattr(traj, "total_times")
    ):
        seg_lengths = seg_lengths_cache
        n_seg = len(seg_lengths)
        dts = np.asarray(traj.dts, dtype=np.float64)
        tt = np.asarray(traj.total_times, dtype=np.float64)
        all_dt_equal = n_seg > 0 and bool((dts == dts[0]).all())
        all_seg_equal = n_seg > 0 and bool((seg_lengths == seg_lengths[0]).all())
        if all_dt_equal and all_seg_equal:
            times_global = np.arange(n, dtype=np.float64) * dts[0]
        else:
            if n_seg > 0 and bool((tt == tt[0]).all()):
                seg_offset = np.arange(n_seg + 1, dtype=np.float64) * tt[0]
            else:
                seg_offset = np.empty(n_seg + 1, dtype=np.float64)
                seg_offset[0] = 0.0
                s_run = 0.0
                s_c = 0.0
                for s in range(n_seg):
                    y = tt[s] - s_c
                    t = s_run + y
                    s_c = (t - s_run) - y
                    s_run = t
                    seg_offset[s + 1] = s_run
            times_global = np.empty(n, dtype=np.float64)
            pos = 0
            for s, n_s in enumerate(seg_lengths):
                if pos >= n:
                    break
                take_n = min(n_s, n - pos)
                times_global[pos:pos + take_n] = (
                    seg_offset[s] + np.arange(take_n, dtype=np.float64) * dts[s]
                )
                pos += take_n
    elif seg_lengths_cache is not None and hasattr(traj, "dt"):
        times_global = np.arange(n, dtype=np.float64) * float(traj.dt)

    CHUNK = 64

    # Bulk-read fast path for ChainReader-of-DCDReader and single DCDReader.
    seg_files = None
    if hasattr(traj, "readers") and len(traj.readers) > 0:
        if all(hasattr(r, "_file") and hasattr(r._file, "readframes")
               for r in traj.readers):
            seg_files = [r._file for r in traj.readers]
    elif hasattr(traj, "_file") and hasattr(traj._file, "readframes"):
        seg_files = [traj._file]
    use_bulk = seg_files is not None

    if use_bulk:
        seg_n_frames = seg_lengths_cache if seg_lengths_cache is not None else (
            np.diff(np.asarray(traj._start_frames, dtype=np.int64))
            if hasattr(traj, "_start_frames")
            else np.array([r.n_frames for r in traj.readers], dtype=np.int64)
            if hasattr(traj, "readers")
            else np.array([traj.n_frames], dtype=np.int64)
        )
        ix_i64 = np.asarray(ix, dtype=np.int64)

        BULK_CHUNK = 8192
        chunk_cap = min(BULK_CHUNK, int(seg_n_frames.max()))
        buf_seg_f64 = np.empty((chunk_cap, n_atoms, 3), dtype=np.float64)
        pos = 0
        for s in range(len(seg_files)):
            if pos >= n:
                break
            dcd_file = seg_files[s]
            dcd_file.seek(0)
            seg_n = int(seg_n_frames[s])
            sub_start = 0
            while sub_start < seg_n and pos < n:
                take = min(BULK_CHUNK, seg_n - sub_start, n - pos)
                seg_xyz_f32 = dcd_file.readframes(
                    start=sub_start,
                    stop=sub_start + take,
                    indices=ix_i64,
                    order="fac",
                ).xyz
                buf_seg_f64[:take] = seg_xyz_f32[:take]
                if w is None:
                    coms = buf_seg_f64[:take].sum(axis=1) * n_atoms_inv
                else:
                    coms = np.einsum("cij,j->ci", buf_seg_f64[:take], w) * w_inv_sum
                buf_seg_f64[:take] -= coms[:, None, :]
                end = pos + take
                rmsd_col2 = rmsd[pos:end, 2]

                def _qcp_inner(b):
                    return qcp_call(ref_coords, b, nat, None, w)

                for j in range(take):
                    rmsd_col2[j] = _qcp_inner(buf_seg_f64[j])
                rmsd[pos:end, 0] = np.arange(pos, end, dtype=rmsd.dtype)
                if times_global is not None:
                    rmsd[pos:end, 1] = times_global[pos:end]
                else:
                    rmsd[pos:end, 1] = np.arange(
                        sub_start, sub_start + take, dtype=rmsd.dtype
                    )
                frames_arr[pos:end] = np.arange(pos, end, dtype=frames_arr.dtype)
                if times_global is not None:
                    times_arr[pos:end] = times_global[pos:end]
                pos = end
                sub_start += take
        self._frame_index = n - 1
        return self

    # Fallback: chunked per-frame iteration (non-ChainReader or non-DCD).
    pos_chunk_f32 = np.empty((CHUNK, n_atoms, 3), dtype=np.float32)
    buf_chunk_f64 = np.empty((CHUNK, n_atoms, 3), dtype=np.float64)
    ts_frames_chunk = np.empty(CHUNK, dtype=np.int64)
    ts_times_chunk = np.empty(CHUNK, dtype=np.float64)
    iterator = iter(sliced)
    chunk_start = 0
    while chunk_start < n:
        actual = min(CHUNK, n - chunk_start)
        for j in range(actual):
            ts = next(iterator)
            np_take(ts.positions, ix, axis=0, out=pos_chunk_f32[j])
            ts_frames_chunk[j] = ts.frame
            ts_times_chunk[j] = ts.time
        buf_chunk_f64[:actual] = pos_chunk_f32[:actual]
        if w is None:
            coms = buf_chunk_f64[:actual].sum(axis=1) * n_atoms_inv
        else:
            coms = np.einsum("cij,j->ci", buf_chunk_f64[:actual], w) * w_inv_sum
        buf_chunk_f64[:actual] -= coms[:, None, :]
        for j in range(actual):
            idx = chunk_start + j
            rmsd[idx, 0] = ts_frames_chunk[j]
            if times_global is not None:
                rmsd[idx, 1] = times_global[idx]
            else:
                rmsd[idx, 1] = ts_times_chunk[j]
            rmsd[idx, 2] = qcp_call(ref_coords, buf_chunk_f64[j], nat, None, w)
        frames_arr[chunk_start:chunk_start + actual] = ts_frames_chunk[:actual]
        times_arr[chunk_start:chunk_start + actual] = ts_times_chunk[:actual]
        chunk_start += actual

    self._frame_index = n - 1
    return self


def fast_read_next_timestep(self, ts=None):
    if self._frame == self.n_frames - 1:
        raise IOError("trying to go over trajectory limit")
    if ts is None:
        ts = self.ts
    frame = self._file.read()
    self._frame += 1
    self._frame_to_ts(frame, ts)
    self.ts = ts
    return ts


def fast_frame_to_ts(self, frame, ts):
    ts.frame = self._frame
    t_offset = getattr(self, "_zyme_t_offset", None)
    if t_offset is None:
        h = self._file.header
        t_offset = h["istart"] / h["nsavc"]
        self._zyme_t_offset = t_offset
    ts.time = (ts.frame + t_offset) * self.ts.dt
    ts.data["step"] = self._file.tell()
    ts.positions = frame.xyz
    if self.convert_units:
        self.convert_pos_from_native(ts.positions)


# ============================================================
# Smoke recipe.
# ============================================================
def _read_task_yaml(task_dir):
    import yaml
    with open(os.path.join(task_dir, "task.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _smoke_load(task_dir, tier):
    """User-side prep: read task.yaml, build Universes + RMSD object.

    Universe construction runs with patches disabled so MDAnalysis performs
    its normal trajectory inspection before ``R.run()`` enters the patched
    hot path.
    """
    import MDAnalysis as mda

    warnings.filterwarnings(
        "ignore",
        message="DCDReader currently makes independent timesteps",
        category=DeprecationWarning,
    )

    task = _read_task_yaml(task_dir)
    ds = next(d for d in task["datasets"] if d["tier"] == tier)
    data_path = resolve_dataset_path(task_dir, ds["path"])
    if os.path.isfile(data_path) and data_path.lower().endswith(".dcd"):
        traj_files = data_path
        ref_dcd = data_path
        candidates = [
            os.path.join(task_dir, "data", "adk_dims", "adk.psf"),
            os.path.join(os.path.dirname(os.path.dirname(data_path)),
                         "adk_dims", "adk.psf"),
            os.path.join(os.path.dirname(data_path), "adk.psf"),
        ]
        psf = next((p for p in candidates if os.path.isfile(p)), None)
        if psf is None:
            raise FileNotFoundError("Could not locate adk.psf for MDAnalysis RMSD task")
    else:
        # Legacy task schema fallback. This supports old package_verify data
        # layouts, but the optimized compute path no longer special-cases
        # repeated filenames.
        params = ds.get("params") or {}
        n_concat = int(params.get("n_concat", 7500))
        pattern = str(params.get("pattern", "repeat"))
        n_unique = int(params.get("n_unique", 1))

        data_root = data_path
        psf = os.path.join(data_root, "adk.psf")
        dcd = os.path.join(data_root, "adk_dims.dcd")

        if pattern == "repeat":
            traj_files = [dcd] * n_concat
            ref_dcd = dcd
        elif pattern == "variant_cycle":
            variants = [
                os.path.join(data_root, f"adk_dims_ood_{i:02d}.dcd")
                for i in range(n_unique)
            ]
            missing = [p for p in variants if not os.path.isfile(p)]
            if missing:
                raise FileNotFoundError(f"OOD variant not found: {missing[0]}")
            traj_files = [variants[i % n_unique] for i in range(n_concat)]
            ref_dcd = variants[0]
        else:
            raise ValueError(f"unsupported trajectory pattern: {pattern}")

    with autozyme.disabled():
        u = mda.Universe(psf, traj_files)
        ref = mda.Universe(psf, ref_dcd)
    R = rms.RMSD(u, ref, select="backbone", ref_frame=0)
    return {"R": R}


def _smoke_call(inputs):
    R = inputs["R"]
    R.run()
    return R


def _smoke_save(result, dir, **kwargs):
    arr = np.asarray(result.results.rmsd, dtype=np.float64)
    out_np = os.path.join(dir, "rmsd.npy")
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    with open(out_np, "wb") as fp:
        fp.write(buf.getvalue())


register_patch(
    name="mdanalysis_rmsd",
    targets=[
        ("MDAnalysis.analysis.rms.RMSD", "_single_frame", fast_single_frame),
        ("MDAnalysis.analysis.rms.RMSD", "_compute", fast_compute),
        ("MDAnalysis.coordinates.DCD.DCDReader",
         "_read_next_timestep", fast_read_next_timestep),
    ],
    smoke={"load": _smoke_load, "call": _smoke_call, "save": _smoke_save},
    tested_against="MDAnalysis 2.10.0",
    tested_upstream_versions={"MDAnalysis": ["2.10.0"]},
)
