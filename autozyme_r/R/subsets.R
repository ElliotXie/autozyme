# Curated patch bundles and known cross-patch conflicts. See
# ../../autozyme_py/src/autozyme/_subsets.py for the matching Python registry —
# keep both languages coordinated when adding patches that exist in both.

# Bundle naming reflects realistic co-activation -- the cross-domain CI
# matrix would burn time (and produce false-positive bugs) trying to
# co-activate maftools (cancer mutations) with vegan (ecology), so bundles
# are scoped per-domain and CI exercises each bundle as one unit.
#
#   - "scrna_signaling": seurat + cellchat + nichenetr + scriabin --
#     three cell-cell communication tools that all run downstream of a
#     Seurat object; canonical post-clustering workflow.
#   - "scrna_trajectory": seurat + mast + slingshot + tradeseq --
#     trajectory inference + lineage-DE pipeline.
#   - "scrna_spatial": seurat + bayesspace + infercnv + rctd -- the
#     spatial-transcriptomics-with-CNV-and-deconvolution combo, common
#     in tumor / TME analyses.
.zyme_subsets <- list(
  scrna_signaling  = c("seurat", "cellchat", "nichenetr", "scriabin"),
  scrna_trajectory = c("seurat", "mast", "slingshot", "tradeseq"),
  scrna_spatial    = c("seurat", "bayesspace", "infercnv", "rctd")
)

# Pairs of patches known to interact badly when activated in the same process.
# `activate()` consults this and warns (does not stop) when both sides are
# being lit up. Each entry: list(pair = c("a", "b"), reason = "...").
# Empty for now — no R-side conflicts have been observed in 5 lifted patches.
.zyme_conflicts <- list()

# Declarative manifest: patch name -> upstream package(s) it depends on.
# `list_patches(installed = TRUE)` / dashboard / env_snapshot consult this via
# `requireNamespace(..., quietly = TRUE)` to answer "is upstream available?".
# This avoids sourcing the patch file (which would probe every helper in
# inst/patches/<name>.R via getFromNamespace, slowing dashboard significantly).
#
# Keep in sync with each patch's register_patch(upstream = ...). New patches
# packaged via 4_package.md must add their entry here.
# AUTOZYME-GENERATED-UPSTREAMS-BEGIN
.zyme_upstreams <- list(
  bayesspace      = "BayesSpace",
  cellchat        = "CellChat",
  clusterprofiler = "clusterProfiler",
  decontx         = "celda",
  fgsea           = "fgsea",
  infercnv        = "infercnv",
  maftools        = "maftools",
  mast            = "MAST",
  nichenetr       = "nichenetr",
  rctd            = "spacexr",
  scriabin        = "scriabin",
  seurat          = "Seurat",
  slingshot       = "slingshot",
  tradeseq        = "tradeSeq",
  vegan           = "vegan",
  wgcna           = "WGCNA"
)
# AUTOZYME-GENERATED-UPSTREAMS-END
