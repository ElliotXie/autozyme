# Patch for clusterProfiler::compareCluster (GO ORA path).
#
# Lifted from autozyme task `test_clusterprofiler`. Targets the
# `compareCluster(fun = "enrichGO", ont = "ALL", ...)` workflow that drives
# per-cluster enrichGO calls. Three coordinated overrides:
#
#   1. clusterProfiler::get_GO_data
#        -> fast_get_GO_data: payload-cache the (org, ont, keytype) PATHID2EXTID
#           / EXTID2PATHID / PATHID2NAME / GO2ONT lists in a patch-scope env and
#           write them into the shared .Anno_clusterProfiler_Env on each call.
#           Builds the four ont entries (ALL/BP/CC/MF) from a single goAnno
#           via vector split + character-paste dedupe (vs orig's data.frame
#           unique + redundant rebuild per ont).
#
#   2. DOSE::enricher_internal      + clusterProfiler::enricher_internal
#        -> fast_enricher_internal: vectorize phyper (drop the row-wise apply),
#           drop the no-op universe-intersect when universe is NULL, use cached
#           per-ont ALLEXTID / TERM_LENGTHS to skip recomputation. Both bindings
#           must be patched: clusterProfiler::enrichGO's bare call to
#           enricher_internal resolves through clusterProfiler's binding, not
#           DOSE's. Patching only DOSE silently no-ops.
#
#   3. clusterProfiler::compareCluster
#        -> fast_compareCluster: fan out the per-cluster enrichGO calls across
#           cores via .zyme_mclapply (each cluster's enrichGO is independent).
#           Falls back to lapply on Windows or when N_PROC <= 1. Skips
#           extract_params slow .call parsing on the final compareClusterResult
#           constructor.
#
if (requireNamespace("clusterProfiler", quietly = TRUE) &&
    requireNamespace("DOSE",            quietly = TRUE) &&
    requireNamespace("GOSemSim",        quietly = TRUE) &&
    requireNamespace("AnnotationDbi",   quietly = TRUE) &&
    requireNamespace("GO.db",           quietly = TRUE)) {

  # Originals + internal helpers captured via getFromNamespace. The fast
  # functions reference them via lexical closure — no environment(fast_) <-
  # asNamespace() trickery (see CAVEATS.md #3 / convention #3).
  .orig_get_GO_data        <- utils::getFromNamespace("get_GO_data",        "clusterProfiler")
  .orig_enricher_internal  <- utils::getFromNamespace("enricher_internal",  "DOSE")
  .orig_compareCluster     <- utils::getFromNamespace("compareCluster",     "clusterProfiler")

  .get_organism_fn      <- utils::getFromNamespace("get_organism",      "clusterProfiler")
  .get_GO_Env_fn        <- utils::getFromNamespace("get_GO_Env",        "clusterProfiler")
  .get_GO2TERM_table_fn <- utils::getFromNamespace("get_GO2TERM_table", "clusterProfiler")
  .load_OrgDb_fn        <- utils::getFromNamespace("load_OrgDb",        "GOSemSim")

  .EXTID2TERMID_fn <- utils::getFromNamespace("EXTID2TERMID", "DOSE")
  .ALLEXTID_fn     <- utils::getFromNamespace("ALLEXTID",     "DOSE")
  .TERMID2EXTID_fn <- utils::getFromNamespace("TERMID2EXTID", "DOSE")
  .TERM2NAME_fn    <- utils::getFromNamespace("TERM2NAME",    "DOSE")

  # ============================================================
  # get_GO_data: payload-cache PATHID2EXTID / EXTID2PATHID / PATHID2NAME / GO2ONT
  # ============================================================
  # DOSE::build_Anno returns a SHARED GLOBAL env (.Anno_clusterProfiler_Env)
  # and overwrites its slots each call — caching the env reference is
  # meaningless. Instead, cache the DATA payload per (org, ont, keytype) and
  # on every call WRITE the cached lists into the shared env.
  .zyme_anno_cache <- new.env(parent = emptyenv())

  .zyme_anno_env <- function() {
    if (!exists(".Anno_clusterProfiler_Env", envir = .GlobalEnv)) {
      assign(".Anno_clusterProfiler_Env", new.env(), envir = .GlobalEnv)
    }
    get(".Anno_clusterProfiler_Env", envir = .GlobalEnv)
  }

  .zyme_apply_cache <- function(cached, anno_env) {
    assign("PATHID2EXTID", cached$PATHID2EXTID, envir = anno_env)
    assign("EXTID2PATHID", cached$EXTID2PATHID, envir = anno_env)
    assign("PATHID2NAME",  cached$PATHID2NAME,  envir = anno_env)
    if (!is.null(cached$GO2ONT)) {
      assign("GO2ONT", cached$GO2ONT, envir = anno_env)
    }
    if (!is.null(cached$ALLEXTID)) {
      assign("ZYME_ALLEXTID", cached$ALLEXTID, envir = anno_env)
    }
    if (!is.null(cached$TERM_LENGTHS)) {
      assign("ZYME_TERM_LENGTHS", cached$TERM_LENGTHS, envir = anno_env)
    }
    invisible(anno_env)
  }

  # Build goAnno via mapIds + character-vector dedupe (vs orig's
  # `unique(goAnno[!is.na(goAnno[, 1]), ])` data.frame unique, ~5× faster).
  # End state matches orig: goAnno data.frame populated in .GO_clusterProfiler_Env.
  .zyme_fast_build_goAnno <- function(OrgDb, keytype) {
    GO_Env <- .get_GO_Env_fn()
    OrgDb <- .load_OrgDb_fn(OrgDb)
    goterms <- AnnotationDbi::Ontology(GO.db::GOTERM)
    go2gene <- suppressMessages(AnnotationDbi::mapIds(
      OrgDb, keys = names(goterms), column = keytype,
      keytype = "GOALL", multiVals = "list"
    ))
    ng <- lengths(go2gene)
    gene_vec  <- unlist(go2gene, use.names = FALSE)
    goall_vec <- rep(names(go2gene), ng)
    not_na <- !is.na(gene_vec)
    gene_vec  <- gene_vec[not_na]
    goall_vec <- goall_vec[not_na]
    dup <- duplicated(paste(goall_vec, gene_vec, sep = "\x01"))
    if (any(dup)) {
      gene_vec  <- gene_vec[!dup]
      goall_vec <- goall_vec[!dup]
    }
    ontology_vec <- as.character(goterms[goall_vec])
    goAnno <- data.frame(V1 = gene_vec, GOALL = goall_vec,
                         ONTOLOGYALL = ontology_vec, stringsAsFactors = FALSE)
    colnames(goAnno)[1] <- keytype
    assign("goAnno", goAnno, envir = GO_Env)
    assign("keytype", keytype, envir = GO_Env)
    assign("ont", "ALL", envir = GO_Env)
    assign("organism", .get_organism_fn(OrgDb), envir = GO_Env)
    invisible(goAnno)
  }

  # First cache miss: ensure goAnno is loaded with ont="ALL" via the fast path,
  # then derive all 4 ont cache entries (ALL/BP/CC/MF) from goAnno via split()
  # — bypassing upstream's redundant `unique(goAnno[, c(2,1)])` + `build_Anno`
  # work, since goAnno is already unique and NA-filtered. Collapses 4
  # expensive misses into 1 mapIds + 4 cheap splits.
  .zyme_populate_caches <- function(OrgDb, keytype, org) {
    GO_Env <- .get_GO_Env_fn()
    need_orig <- TRUE
    if (exists("goAnno",   envir = GO_Env, inherits = FALSE) &&
        exists("ont",      envir = GO_Env, inherits = FALSE) &&
        exists("keytype",  envir = GO_Env, inherits = FALSE) &&
        exists("organism", envir = GO_Env, inherits = FALSE)) {
      if (get("ont", envir = GO_Env) == "ALL" &&
          get("keytype", envir = GO_Env) == keytype &&
          get("organism", envir = GO_Env) == org) {
        need_orig <- FALSE
      }
    }
    if (need_orig) {
      .zyme_fast_build_goAnno(OrgDb, keytype)
    }
    goAnno <- get("goAnno", envir = GO_Env)

    path2name <- .get_GO2TERM_table_fn()
    pn_df <- as.data.frame(path2name)
    pn_keep <- !is.na(pn_df[[1]]) & !is.na(pn_df[[2]])
    PATHID2NAME <- as.character(pn_df[pn_keep, 2])
    names(PATHID2NAME) <- as.character(pn_df[pn_keep, 1])

    goAnno_genes <- as.character(goAnno[[1]])
    goAnno_terms <- as.character(goAnno$GOALL)
    goAnno_onts  <- as.character(goAnno$ONTOLOGYALL)
    unique_terms <- unique(goAnno_terms)
    GO2ONT <- setNames(goAnno_onts[match(unique_terms, goAnno_terms)], unique_terms)

    for (target_ont in c("ALL", "BP", "CC", "MF")) {
      if (target_ont == "ALL") {
        sub_genes <- goAnno_genes
        sub_terms <- goAnno_terms
      } else {
        mask <- goAnno_onts == target_ont
        sub_genes <- goAnno_genes[mask]
        sub_terms <- goAnno_terms[mask]
      }
      PATHID2EXTID <- split(sub_genes, sub_terms)
      EXTID2PATHID <- split(sub_terms, sub_genes)
      ALLEXTID_cached <- unique(sub_genes)
      TERM_LENGTHS_cached <- lengths(PATHID2EXTID)
      sub_key <- paste(org, target_ont, keytype, sep = "::")
      assign(sub_key, list(
        PATHID2EXTID = PATHID2EXTID,
        EXTID2PATHID = EXTID2PATHID,
        PATHID2NAME  = PATHID2NAME,
        GO2ONT       = if (target_ont == "ALL") GO2ONT else NULL,
        ALLEXTID     = ALLEXTID_cached,
        TERM_LENGTHS = TERM_LENGTHS_cached
      ), envir = .zyme_anno_cache)
    }
    invisible(NULL)
  }

  fast_get_GO_data <- function(OrgDb, ont, keytype, zyme = TRUE) {
    if (!isTRUE(zyme)) {
      return(.orig_get_GO_data(OrgDb, ont, keytype))
    }
    org <- tryCatch(.get_organism_fn(OrgDb), error = function(e) {
      if (is.character(OrgDb)) OrgDb else "unknown"
    })
    key <- paste(org, ont, keytype, sep = "::")
    cached <- get0(key, envir = .zyme_anno_cache, inherits = FALSE)
    anno_env <- .zyme_anno_env()
    if (is.null(cached)) {
      .zyme_populate_caches(OrgDb, keytype, org)
      cached <- get0(key, envir = .zyme_anno_cache, inherits = FALSE)
    }
    .zyme_apply_cache(cached, anno_env)
    anno_env
  }

  # ============================================================
  # enricher_internal: vectorize phyper + skip redundant intersect when
  # universe is NULL (the compareCluster default)
  # ============================================================
  fast_enricher_internal <- function(gene, pvalueCutoff, pAdjustMethod = "BH",
                                     universe = NULL, minGSSize = 10,
                                     maxGSSize = 500, qvalueCutoff = 0.2,
                                     USER_DATA, zyme = TRUE) {
    if (!isTRUE(zyme)) {
      return(.orig_enricher_internal(
        gene = gene, pvalueCutoff = pvalueCutoff,
        pAdjustMethod = pAdjustMethod, universe = universe,
        minGSSize = minGSSize, maxGSSize = maxGSSize,
        qvalueCutoff = qvalueCutoff, USER_DATA = USER_DATA
      ))
    }
    gene <- as.character(unique(gene))
    qExtID2TermID <- .EXTID2TERMID_fn(gene, USER_DATA)
    qTermID <- unlist(qExtID2TermID)
    if (is.null(qTermID)) {
      message("--> No gene can be mapped....")
      if (inherits(USER_DATA, "environment")) {
        p2e <- get("PATHID2EXTID", envir = USER_DATA)
        sg <- unique(unlist(p2e[1:10]))
      } else {
        sg <- unique(USER_DATA@gsid2gene$gene[1:100])
      }
      sg <- sample(sg, min(length(sg), 6))
      message("--> Expected input gene ID: ", paste0(sg, collapse = ","))
      message("--> return NULL...")
      return(NULL)
    }
    # Skip data.frame construction; dedupe (extID, termID) pairs via
    # character-paste + duplicated() (C-level dedupe) and split directly.
    ext_vec  <- rep(names(qExtID2TermID), times = lengths(qExtID2TermID))
    term_vec <- as.character(qTermID)
    dup <- duplicated(paste(ext_vec, term_vec, sep = "\x01"))
    if (any(dup)) {
      ext_vec  <- ext_vec[!dup]
      term_vec <- term_vec[!dup]
    }
    qTermID2ExtID <- split(ext_vec, term_vec)
    extID <- get0("ZYME_ALLEXTID", envir = USER_DATA, inherits = FALSE)
    if (is.null(extID)) extID <- .ALLEXTID_fn(USER_DATA)
    if (missing(universe)) universe <- NULL
    use_universe <- FALSE
    if (!is.null(universe)) {
      if (is.character(universe)) {
        force_universe <- getOption("enrichment_force_universe", FALSE)
        if (force_universe) {
          extID <- universe
        } else {
          extID <- intersect(extID, universe)
        }
        use_universe <- TRUE
      } else {
        message("`universe` is not in character and will be ignored...")
      }
    }
    if (use_universe) {
      qTermID2ExtID <- lapply(qTermID2ExtID, intersect, extID)
    }
    qTermID <- names(qTermID2ExtID)
    termID2ExtID <- .TERMID2EXTID_fn(qTermID, USER_DATA)
    if (use_universe) {
      termID2ExtID <- lapply(termID2ExtID, intersect, extID)
    }
    geneSets <- termID2ExtID
    cached_lens <- if (!use_universe)
      get0("ZYME_TERM_LENGTHS", envir = USER_DATA, inherits = FALSE) else NULL
    geneSet_size <- if (!is.null(cached_lens))
      cached_lens[qTermID] else lengths(termID2ExtID)
    idx <- !is.na(geneSet_size) & geneSet_size >= minGSSize &
           geneSet_size <= maxGSSize
    if (sum(idx) == 0) {
      msg <- paste("No gene sets have size between", minGSSize,
                   "and", maxGSSize, "...")
      message(msg)
      message("--> return NULL...")
      return(NULL)
    }
    termID2ExtID  <- termID2ExtID[idx]
    qTermID2ExtID <- qTermID2ExtID[idx]
    qTermID <- names(qTermID2ExtID)
    k <- lengths(qTermID2ExtID)[qTermID]
    M <- if (!is.null(cached_lens)) cached_lens[qTermID]
         else lengths(termID2ExtID)[qTermID]
    N <- rep(length(extID), length(M))
    n <- rep(length(qExtID2TermID), length(M))
    args.df <- data.frame(numWdrawn = k - 1, numW = M, numB = N - M,
                          numDrawn = n)
    pvalues <- phyper(args.df$numWdrawn, args.df$numW, args.df$numB,
                      args.df$numDrawn, lower.tail = FALSE)
    GeneRatio      <- sprintf("%s/%s", k, n)
    BgRatio        <- sprintf("%s/%s", M, N)
    RichFactor     <- k / M
    FoldEnrichment <- RichFactor * N / n
    mu    <- M * n / N
    sigma <- mu * (N - n) * (N - M) / N / (N - 1)
    zScore <- (k - mu) / sqrt(sigma)
    Over <- data.frame(ID = as.character(qTermID), GeneRatio = GeneRatio,
                       BgRatio = BgRatio, RichFactor = RichFactor,
                       FoldEnrichment = FoldEnrichment, zScore = zScore,
                       pvalue = pvalues, stringsAsFactors = FALSE)
    p.adj <- p.adjust(Over$pvalue, method = pAdjustMethod)
    qobj <- tryCatch(qvalue::qvalue(p = Over$pvalue, lambda = 0.05,
                                    pi0.method = "bootstrap"),
                     error = function(e) NULL)
    qvalues <- if (inherits(qobj, "qvalue")) qobj$qvalues else NA
    geneID <- vapply(qTermID2ExtID, paste, FUN.VALUE = character(1),
                     collapse = "/")[qTermID]
    Over <- data.frame(Over, p.adjust = p.adj, qvalue = qvalues,
                       geneID = geneID, Count = k, stringsAsFactors = FALSE)
    Description <- .TERM2NAME_fn(qTermID, USER_DATA)
    if (length(qTermID) != length(Description)) {
      idx <- qTermID %in% names(Description)
      Over <- Over[idx, ]
    }
    Over$Description <- Description
    nc <- ncol(Over)
    Over <- Over[, c(1, nc, 2:(nc - 1))]
    Over <- Over[order(pvalues), ]
    Over$ID <- as.character(Over$ID)
    Over$Description <- as.character(Over$Description)
    row.names(Over) <- as.character(Over$ID)
    x <- methods::new("enrichResult", result = Over,
                      pvalueCutoff = pvalueCutoff,
                      pAdjustMethod = pAdjustMethod,
                      qvalueCutoff = qvalueCutoff,
                      gene = as.character(gene), universe = extID,
                      geneSets = geneSets, organism = "UNKNOWN",
                      keytype = "UNKNOWN", ontology = "UNKNOWN",
                      readable = FALSE)
    if (inherits(USER_DATA, "GSON")) {
      if (!is.null(USER_DATA@keytype))  x@keytype  <- USER_DATA@keytype
      if (!is.null(USER_DATA@species))  x@organism <- USER_DATA@species
      if (!is.null(USER_DATA@gsname))   x@ontology <- gsub(".*;", "", USER_DATA@gsname)
    }
    return(x)
  }

  # ============================================================
  # compareCluster: parallelize per-cluster enrichGO via mclapply
  # ============================================================
  fast_compareCluster <- function(geneClusters, fun = "enrichGO",
                                  data = "", source_from = NULL, ...,
                                  zyme = TRUE) {
    if (!isTRUE(zyme)) {
      return(.orig_compareCluster(
        geneClusters = geneClusters, fun = fun,
        data = data, source_from = source_from, ...
      ))
    }
    # Scope guard (additive): the fast path iterates geneClusters as a plain
    # named list and resolves a character `fun` from the clusterProfiler
    # namespace only. The formula interface (geneClusters as a formula + data),
    # a custom source_from, or a `fun` name not in clusterProfiler are not
    # handled here -> defer to upstream. The benchmarked list + namespaced-fun
    # call (e.g. fun="enrichGO") passes through unchanged.
    if (inherits(geneClusters, "formula") || !is.null(source_from) ||
        (is.character(fun) &&
         !exists(fun, envir = asNamespace("clusterProfiler"), inherits = FALSE))) {
      return(.orig_compareCluster(
        geneClusters = geneClusters, fun = fun,
        data = data, source_from = source_from, ...
      ))
    }
    fun_name <- if (is.character(fun)) fun else deparse(substitute(fun))
    if (is.character(fun)) {
      fun <- utils::getFromNamespace(fun, "clusterProfiler")
    }
    args_passed <- list(...)
    do_one <- function(i) {
      x <- do.call(fun, c(list(i), args_passed))
      if (!inherits(x, c("enrichResult", "groupGOResult", "gseaResult"))) {
        return(NULL)
      }
      df <- x@result
      pcut <- x@pvalueCutoff
      if (length(pcut) > 0) {
        df <- df[df$pvalue <= pcut & df$p.adjust <= pcut, , drop = FALSE]
      }
      qcut <- x@qvalueCutoff
      if (length(qcut) > 0 && !any(is.na(df$qvalue))) {
        df <- df[df$qvalue <= qcut, , drop = FALSE]
      }
      df
    }

    # Shared resolver: AUTOZYME_THREADS env > getOption("autozyme.threads")
    # (set by set_threads()) > physical cores - 1, capped at 8. Routing through
    # auto_threads() is what lets the documented global thread knobs reach this
    # patch; under attest it resolves to the same value the harness pins.
    n_threads <- max(1L, as.integer(autozyme::auto_threads(cap = 8L)))
    use_parallel <- (.Platform$OS.type != "windows") &&
                    length(geneClusters) > 1 && n_threads > 1L

    if (use_parallel) {
      clProf <- .zyme_mclapply(
        geneClusters, do_one,
        mc.cores       = n_threads,
        mc.preschedule = TRUE,
        mc.set.seed    = TRUE
      )
    } else {
      clProf <- lapply(geneClusters, do_one)
    }
    names(clProf) <- names(geneClusters)
    ok <- !vapply(clProf, is.null, logical(1))
    clProf <- clProf[ok]
    if (length(clProf) == 0) {
      warning("No enrichment found in any of gene cluster, please check your input...")
      return(NULL)
    }
    cluster_names <- rep(names(clProf), times = vapply(clProf, nrow, integer(1)))
    clProf.df <- do.call(rbind, c(clProf, list(make.row.names = FALSE)))
    clProf.df <- cbind(Cluster = cluster_names, clProf.df, stringsAsFactors = FALSE)
    clProf.df$Cluster <- factor(clProf.df$Cluster, levels = names(geneClusters))
    keytype  <- args_passed[["keyType"]] %||% "UNKNOWN"
    readable <- args_passed[["readable"]] %||% FALSE
    res <- methods::new(
      "compareClusterResult",
      compareClusterResult = clProf.df,
      geneClusters = geneClusters,
      keytype  = keytype,
      readable = as.logical(readable),
      fun      = fun_name
    )
    res
  }

  register_patch(
    name = "clusterprofiler",
    upstream = "clusterProfiler",
    targets = list(
      get_GO_data       = fast_get_GO_data,
      enricher_internal = fast_enricher_internal,
      compareCluster    = fast_compareCluster
    ),
    smoke = list(
      load = function(task_dir, tier) {
        task <- yaml::read_yaml(file.path(task_dir, "task.yaml"))
        ds <- Filter(function(d) d$tier == tier, task$datasets)
        if (length(ds) == 0L) stop("no dataset for tier '", tier, "' in task.yaml")
        ds <- ds[[1]]
        params <- if (is.null(ds$params)) list() else ds$params
        cache_key <- params$cache_key %||% ds$name %||%
                     basename(dirname(ds$path))
        org_db    <- params$org_db   %||% "org.Hs.eg.db"
        key_type  <- params$key_type %||% "ENTREZID"

        # Pre-built DEG input is upstream-of-target prep (the markers run +
        # SYMBOL->ENTREZID conversion) — same work the user would do before
        # calling compareCluster. Untimed in pipeline/run.R; untimed here.
        deg_input_path <- file.path(task_dir, "data", "deg_input",
                                    paste0(cache_key, ".rds"))
        if (!file.exists(deg_input_path)) {
          stop(sprintf(
            "DEG input cache not found at %s — run reference.R for tier '%s' first",
            deg_input_path, tier))
        }
        gene_clusters <- readRDS(deg_input_path)
        if (!requireNamespace(org_db, quietly = TRUE)) {
          stop(sprintf("Required OrgDb package not installed: %s", org_db))
        }

        list(
          gene_clusters = gene_clusters,
          org_db        = org_db,
          key_type      = key_type
        )
      },
      call = function(inputs) {
        # Only compareCluster is timed. Dispatch goes through
        # clusterProfiler::compareCluster, which picks up our wrapper when the
        # patch is active and the upstream original otherwise.
        clusterProfiler::compareCluster(
          geneClusters  = inputs$gene_clusters,
          fun           = "enrichGO",
          OrgDb         = inputs$org_db,
          keyType       = inputs$key_type,
          ont           = "ALL",
          pvalueCutoff  = 0.05,
          qvalueCutoff  = 0.2,
          pAdjustMethod = "BH",
          minGSSize     = 10,
          maxGSSize     = 500
        )
      },
      save = function(result, dir, ...) {
        # evaluate.R reads pipeline/result.rds as a data.frame; mirror exactly.
        result_df <- as.data.frame(result)
        saveRDS(result_df, file.path(dir, "result.rds"))
      }
    ),
    tested_against = "clusterProfiler 4.16.0",
    tested_upstream_versions = list(clusterProfiler = "4.16.0")
  )
}
