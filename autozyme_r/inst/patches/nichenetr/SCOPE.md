# autozyme `nichenetr` (R) — supported parameter scope

This patch accelerates the function(s) below but is **validated only for a
specific parameter envelope**. Within that envelope the fast path reproduces the
upstream result (to the stated output-equivalence class); outside it, behavior
is one of: falls back to upstream, raises, or — for a few documented
approximations — silently approximates.

Supported parameters (e.g. `n_comps` / `npcs`, resolution, the data itself) can
be set freely. Only the parameters listed under each entry change behavior.

_Auto-generated from `scripts/patch_scope.tsv`. Do not edit by hand — run
`scripts/gen_scope_docs.py`._


## `nichenetr::predict_ligand_activities`

- **In-scope output equivalence:** bit_exact
- **Validated at:** `nichenetr::predict_ligand_activities(geneset = <data>, background_expressed_genes = <data>, ligand_target_matrix = <data>, potential_ligands = <data>)  # single omitted -> upstream default single=TRUE; dense base matrix; benchmarked tiers: small=1287 ligands x 22521 genes (lcmv_mouse_full), medium=2452 (ifnb_human_x2), large=4904 (ifnb_human_x4), ood_large=ltm x6, ood_xlarge=ltm x10`
- **Supported scope:** Fast path is taken only when zyme=TRUE (default) AND single=TRUE (default), i.e. the per-ligand scoring mode that is nichenetr's common/default use case. It correctly computes the four output metrics (auroc, aupr, aupr_corrected, pearson) for any potential_ligands all present in colnames(ligand_target_matrix), any geneset/background with at least one gene matching rownames(ligand_target_matrix), for a DENSE base R numeric matrix ligand_target_matrix. Dispatch by ligand count: n_ligands >= 64 -> parallel C++ kernel score_ligands_cpp (threads = min(getOption('autozyme.threads',14), n_ligands, physical cores)); n_ligands < 64 -> vectorized R fallback using .nichenetr_fast_score_selected_metrics (bit-exact to upstream caTools::trapz formulation). Within the C++ kernel, n_pos<=2048 uses the binary-search LigandScoreBinaryWorker, n_pos>2048 uses the heap-vector LigandScoreWorker; both produce the same metrics. The benchmark exercises this exact path: all dev/OOD tiers call with the four data args only (single defaults to upstream TRUE), so the benchmarked_call is pure upstream defaults except for data, and bit-exact agreement is reported (max_abs_diff_aupr_corrected=0, all correlations=1).
- **Out-of-scope behavior:** Out-of-scope parameters **raise an error** (no silent wrong result).

