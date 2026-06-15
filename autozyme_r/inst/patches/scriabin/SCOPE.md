# autozyme `scriabin` (R) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `scriabin::GenerateCCIM`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `GenerateCCIM(object=<data>, assay="RNA", slot="data", species="human", database="OmniPath", ligands=NULL, recepts=NULL, senders=NULL, receivers=NULL, weighted=FALSE, nichenet_results=NULL, pearson.cutoff=0.075, scale.factors=c(1.5, 3), weight.method="sum")`
- **Supported scope:** Fast path is taken ONLY for the default unweighted CCIM call: zyme=TRUE (default), weighted=FALSE, ligands=NULL, recepts=NULL, senders=NULL, receivers=NULL, and database != "custom". In that regime it computes sqrt of the per-LR-pair rank-1 outer products (sender index fastest, receiver slowest) over ALL cells (senders=receivers=colnames(object)) using the OmniPath/CellChatDB/etc. built-in LR resource filtered at lit_support=7, exactly mirroring upstream's pbsapply(tcrossprod) + sqrt + Seurat CCIM constructor + MapMetaData(all columns). Any assay name and slot/layer name are honored (passed through to GetAssayData; a deprecated-slot= shim is installed). species other than human (mouse/rat, or any value) and any non-custom database name are passed through to the cached real LoadLR, which validates them and errors on unsupported values exactly as upstream. Verified bit-exact (ccim_pearson=1, ccim_max_abs_diff=0, ccim_nnz_jaccard=1) across small/medium/large/ood_large/ood_xlarge tiers.
- **Out-of-scope behavior:** Out-of-scope parameters **fall back to the upstream implementation** (correct result, no speedup).

