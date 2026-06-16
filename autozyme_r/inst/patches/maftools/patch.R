# Patch for maftools::read.maf (+ its two heaviest internal helpers,
# maftools:::validateMaf and maftools:::summarizeMaf).
#
# Lifted from autozyme task `test_maftools`. Three coordinated namespace-fn
# overrides, all on maftools' namespace. Together they accelerate the full
# read.maf pipeline on a MAF.gz path:
#
#   1. maftools:::summarizeMaf -> fast_summarizeMaf
#        Replaces per-gene `length(unique(Tumor_Sample_Barcode))` calls with
#        data.table::uniqueN (single C-level pass), short-circuits the
#        MutatedSamples == AlteredSamples branch when no CNV variants are
#        present, swaps data.table::dcast(drop=FALSE) for an Rcpp-fill on
#        the variant_classification + variant_type per-sample summaries.
#
#   2. maftools:::validateMaf -> fast_validateMaf
#        `duplicated(maf, by = c(<5 keys>))` instead of paste(.) + duplicated;
#        forderv/chmatch in-place factor build for Tumor_Sample_Barcode
#        (skips structure() + column-replacement copies); skips redundant
#        as.character round-trips on Chromosome / Start / End_Position;
#        skips the unique(VC) / unique(VT) report scans when chatty=FALSE.
#
#   3. maftools::read.maf -> fast_read.maf
#        Calls our patched validateMaf + summarizeMaf via the namespace
#        chain; reads .gz inputs via data.table::fread(file=) — the native
#        handler is ~7% slower than the old `gunzip -c` pipe but portable
#        (no shell-out, works on Windows where `gunzip` isn't in PATH);
#        factor builds for Variant_Classification + Variant_Type use the
#        same forderv/chmatch shortcut. Defers rare branches (gistic,
#        cnTable, isTCGA, useAll=FALSE) to upstream.
#
# Patch kind: namespace function (three targets, same upstream). One small
# C++ kernel (src/maftools.cpp -> zyme_fill_dcast) drives the dcast-fill.
# No threading: data.table is the dominant parallelism path and is set at
# the user level (data.table::setDTthreads); the patch does not override.

if (requireNamespace("maftools",   quietly = TRUE) &&
    requireNamespace("data.table", quietly = TRUE)) {

  # data.table's `[.data.table` and `:=` gate dispatch through `cedta()`,
  # which checks `topenv(parent.frame())` against data.table's whitelist.
  # autozyme declares `data.table` in DESCRIPTION's Imports field for
  # exactly this — `cedta` accepts any namespace whose Imports table
  # mentions data.table. (We don't `importFrom(data.table, ...)` in
  # NAMESPACE because none of the data.table symbols are looked up by
  # bare name in patches — every call site uses `data.table::`.)
  # See `?data.table::cedta`.

  # Originals + internal helpers captured at file scope via getFromNamespace.
  # Fast fns reference these via lexical closure — see CAVEATS.md / convention #3.
  .orig_read_maf       <- utils::getFromNamespace("read.maf",     "maftools")
  .orig_validateMaf    <- utils::getFromNamespace("validateMaf",  "maftools")
  .orig_summarizeMaf   <- utils::getFromNamespace("summarizeMaf", "maftools")
  .orig_flags          <- utils::getFromNamespace("flags",        "maftools")
  .orig_MAF            <- utils::getFromNamespace("MAF",          "maftools")

  # data.table internal — both pipeline/run.R and our patch use the
  # un-exported forderv for stable, fast level ordering.
  .dt_forderv <- utils::getFromNamespace("forderv", "data.table")

  # ============================================================
  # Shared helpers
  # ============================================================
  # In-place factor build for a data.table column — skips $<- copy-on-set.
  .fast_factor_dt_col <- function(dt, col) {
    v <- dt[[col]]
    if (is.factor(v)) return(invisible())
    if (!is.character(v)) v <- as.character(v)
    u <- unique(v); levs <- u[.dt_forderv(u)]
    data.table::set(dt, j = col, value = data.table::chmatch(v, levs))
    data.table::setattr(dt[[col]], "levels", levs)
    data.table::setattr(dt[[col]], "class", "factor")
    invisible()
  }

  # dcast(drop=FALSE) replacement: Rcpp fill over (row, col, val) triplets.
  # zyme_fill_dcast is the compiled kernel in src/maftools.cpp — exposed in
  # autozyme's namespace via compileAttributes/useDynLib.
  zyme_dcast_drop_false <- function(dt, row_col, col_col, val_col,
                                    row_levs, col_levs) {
    M <- zyme_fill_dcast(as.integer(dt[[row_col]]),
                         as.integer(dt[[col_col]]),
                         as.integer(dt[[val_col]]),
                         length(row_levs), length(col_levs))
    out <- data.table::as.data.table(M)
    data.table::setnames(out, col_levs)
    data.table::set(out, j = row_col,
                    value = structure(seq_along(row_levs),
                                      levels = row_levs, class = "factor"))
    data.table::setcolorder(out, c(row_col, col_levs))
    out
  }

  # ============================================================
  # fast_summarizeMaf
  # ============================================================
  fast_summarizeMaf <- function(maf, anno = NULL, chatty = TRUE, zyme = TRUE) {
    if (!isTRUE(zyme)) return(.orig_summarizeMaf(maf = maf, anno = anno, chatty = chatty))
    if (!is.data.frame(maf)) maf <- data.table::as.data.table(maf)

    cnv_mask <- maf$Variant_Type != 'CNV'
    if ('NCBI_Build' %in% colnames(maf)) {
      NCBI_Build <- unique(maf$NCBI_Build[cnv_mask])
      NCBI_Build <- NCBI_Build[!is.na(NCBI_Build)]
      if (length(NCBI_Build) == 0) NCBI_Build <- NA
      if (chatty && length(NCBI_Build) > 1) {
        cat('--Mutiple reference builds found\n')
        NCBI_Build <- do.call(paste, c(as.list(NCBI_Build), sep = ";"))
        cat(NCBI_Build)
      }
      if (length(NCBI_Build) == 0) NCBI_Build <- NA
    } else NCBI_Build <- NA
    if ('Center' %in% colnames(maf)) {
      Center <- unique(maf$Center[cnv_mask])
      if (length(Center) > 1) {
        Center <- do.call(paste, c(as.list(Center), sep = ";"))
        if (chatty) { cat('--Mutiple centers found\n'); cat(Center) }
      }
      if (length(Center) == 0) Center <- NA
    } else Center <- NA

    nGenes   <- length(unique(maf[, Hugo_Symbol]))
    maf.tsbs <- levels(maf[, Tumor_Sample_Barcode])
    nSamples <- length(maf.tsbs)

    flags <- .orig_flags(top = 20)

    # variants.per.sample: tabulate over TSB factor codes — skips dt grouping.
    .tsb_codes <- as.integer(maf$Tumor_Sample_Barcode)
    .tsb_cnt   <- tabulate(.tsb_codes, nbins = nSamples)
    .nz        <- which(.tsb_cnt > 0L)
    tsb <- data.table::data.table(
      Tumor_Sample_Barcode = structure(.nz, levels = maf.tsbs, class = "factor"),
      Variants = .tsb_cnt[.nz]
    )
    data.table::setorder(tsb, -Variants)

    vc <- maf[, .N, .(Tumor_Sample_Barcode, Variant_Classification)]
    vc.cast <- zyme_dcast_drop_false(vc, "Tumor_Sample_Barcode", "Variant_Classification", "N",
                                     maf.tsbs, levels(maf$Variant_Classification))

    if (any(colnames(vc.cast) %in% c('Amp', 'Del'))) {
      vc.cast.cnv <- vc.cast[, c('Tumor_Sample_Barcode', colnames(vc.cast)[colnames(vc.cast) %in% c('Amp', 'Del')]), with = FALSE]
      vc.cast.cnv$CNV_total <- rowSums(vc.cast.cnv[, 2:ncol(vc.cast.cnv)], na.rm = TRUE)
      vc.cast <- vc.cast[, !colnames(vc.cast)[colnames(vc.cast) %in% c('Amp', 'Del')], with = FALSE]
      if (ncol(vc.cast) > 1) vc.cast[, total := rowSums(vc.cast[, 2:ncol(vc.cast), with = FALSE])] else vc.cast[, total := 0]
      vc.cast <- merge(vc.cast, vc.cast.cnv, by = 'Tumor_Sample_Barcode', all = TRUE)[order(total, CNV_total, decreasing = TRUE)]
      vc.mean   <- as.numeric(as.character(c(NA, NA, NA, NA, apply(vc.cast[, 2:ncol(vc.cast), with = FALSE], 2, mean))))
      vc.median <- as.numeric(as.character(c(NA, NA, NA, NA, apply(vc.cast[, 2:ncol(vc.cast), with = FALSE], 2, median))))
    } else {
      vc.cast[, total := rowSums(vc.cast[, 2:ncol(vc.cast), with = FALSE])]
      data.table::setorder(vc.cast, -total)
      .vc_mat   <- as.matrix(vc.cast[, 2:ncol(vc.cast), with = FALSE])
      vc.mean   <- round(c(NA_real_, NA_real_, NA_real_, NA_real_, colMeans(.vc_mat)), 3)
      vc.median <- round(c(NA_real_, NA_real_, NA_real_, NA_real_, apply(.vc_mat, 2, median)), 3)
    }

    vt <- maf[, .N, .(Tumor_Sample_Barcode, Variant_Type)]
    vt.cast <- zyme_dcast_drop_false(vt, "Tumor_Sample_Barcode", "Variant_Type", "N",
                                     maf.tsbs, levels(maf$Variant_Type))

    if (any(colnames(vt.cast) %in% c('CNV'))) {
      vt.cast.cnv <- vt.cast[, c('Tumor_Sample_Barcode', colnames(vt.cast)[colnames(vt.cast) %in% c('CNV')]), with = FALSE]
      vt.cast <- vt.cast[, !colnames(vt.cast)[colnames(vt.cast) %in% c('CNV')], with = FALSE]
      if (ncol(vt.cast) > 1) vt.cast <- vt.cast[, total := rowSums(vt.cast[, 2:ncol(vt.cast), with = FALSE])] else vt.cast[, total := 0]
      vt.cast <- merge(vt.cast, vt.cast.cnv, by = 'Tumor_Sample_Barcode', all = TRUE)[order(total, CNV, decreasing = TRUE)]
    } else {
      vt.cast[, total := rowSums(vt.cast[, 2:ncol(vt.cast), with = FALSE])]
      data.table::setorder(vt.cast, -total)
    }

    hs <- maf[, .N, .(Hugo_Symbol, Variant_Classification)]
    hs.cast <- data.table::dcast(data = hs, formula = Hugo_Symbol ~ Variant_Classification,
                                 fill = 0, value.var = 'N')

    if (any(colnames(hs.cast) %in% c('Amp', 'Del'))) {
      hs.cast.cnv <- hs.cast[, c('Hugo_Symbol', colnames(hs.cast)[colnames(hs.cast) %in% c('Amp', 'Del')]), with = FALSE]
      hs.cast.cnv$CNV_total <- rowSums(x = hs.cast.cnv[, 2:ncol(hs.cast.cnv), with = FALSE], na.rm = TRUE)
      hs.cast <- hs.cast[, !colnames(hs.cast)[colnames(hs.cast) %in% c('Amp', 'Del')], with = FALSE]
      if (ncol(hs.cast) > 1) hs.cast[, total := rowSums(hs.cast[, 2:ncol(hs.cast), with = FALSE], na.rm = TRUE)] else hs.cast[, total := 0]
      hs.cast <- merge(hs.cast, hs.cast.cnv, by = 'Hugo_Symbol', all = TRUE)[order(total, CNV_total, decreasing = TRUE)]
    } else {
      hs.cast[, total := rowSums(hs.cast[, 2:ncol(hs.cast), with = FALSE])]
      data.table::setorder(hs.cast, -total)
    }

    has_cnv <- any(maf$Variant_Type == 'CNV')
    if (has_cnv) {
      numMutatedSamples <- maf[!Variant_Type %in% 'CNV',
                               .(MutatedSamples = data.table::uniqueN(Tumor_Sample_Barcode)),
                               by = Hugo_Symbol]
      numAlteredSamples <- maf[, .(AlteredSamples = data.table::uniqueN(Tumor_Sample_Barcode)),
                               by = Hugo_Symbol]
      numAlteredSamples <- merge(numMutatedSamples, numAlteredSamples, by = 'Hugo_Symbol', all = TRUE)
    } else {
      numAlteredSamples <- maf[, .(AlteredSamples = data.table::uniqueN(Tumor_Sample_Barcode)),
                               by = Hugo_Symbol]
      numAlteredSamples[, MutatedSamples := AlteredSamples]
    }
    hs.cast <- merge(hs.cast, numAlteredSamples, by = 'Hugo_Symbol', all = TRUE)
    data.table::setorder(hs.cast, -MutatedSamples, -total)
    hs.cast[is.na(AlteredSamples), AlteredSamples := 0L]
    hs.cast[is.na(MutatedSamples), MutatedSamples := 0L]

    .vc_colsums <- if (exists(".vc_mat", inherits = FALSE)) colSums(.vc_mat) else colSums(vc.cast[, 2:ncol(vc.cast), with = FALSE])
    summary <- data.table::data.table(
      ID = c('NCBI_Build', 'Center', 'Samples', 'nGenes', colnames(vc.cast)[2:ncol(vc.cast)]),
      summary = c(NCBI_Build, Center, nSamples, nGenes, .vc_colsums)
    )
    summary[, Mean := vc.mean]; summary[, Median := vc.median]

    if (nrow(hs.cast) > 10) {
      topten <- as.character(hs.cast[1:10, Hugo_Symbol])
      topten <- topten[topten %in% flags]
      if (chatty && length(topten) > 0) {
        cat('--Possible FLAGS among top ten genes:\n')
        for (temp in topten) cat(paste0("  ", temp, "\n"))
      }
    }
    if (chatty) cat("-Processing clinical data\n")

    if (is.null(anno)) {
      if (chatty) cat("--Missing clinical data\n")
      sample.anno <- data.table::data.table(Tumor_Sample_Barcode = maf.tsbs)
    } else if (is.data.frame(x = anno)) {
      sample.anno <- data.table::copy(x = anno); data.table::setDT(sample.anno)
      if (!'Tumor_Sample_Barcode' %in% colnames(sample.anno)) {
        message(paste0('Available fields in provided annotations..')); print(colnames(sample.anno))
        stop('Tumor_Sample_Barcode column not found in provided clinical data. Rename column containing sample names to Tumor_Sample_Barcode if necessary.')
      }
    } else if (file.exists(anno)) {
      sample.anno <- data.table::fread(anno, stringsAsFactors = FALSE, fill = TRUE)
      if (!'Tumor_Sample_Barcode' %in% colnames(sample.anno)) {
        message(paste0('Available fields in ', basename(anno), '..')); print(colnames(sample.anno))
        stop('Tumor_Sample_Barcode column not found in provided clinical data. Rename column name containing sample names to Tumor_Sample_Barcode if necessary.')
      }
    }

    colnames(sample.anno) <- gsub(' ', '_', colnames(sample.anno), fixed = TRUE)
    data.table::setDT(sample.anno)
    if (ncol(sample.anno) == 1) colnames(sample.anno)[1] <- "Tumor_Sample_Barcode"
    sample.anno <- sample.anno[!duplicated(Tumor_Sample_Barcode)]
    anno.tsbs <- sample.anno[, Tumor_Sample_Barcode]

    if (length(maf.tsbs[!maf.tsbs %in% anno.tsbs]) != 0 && chatty) {
      cat('--Annotation missing for below samples in MAF:\n')
      for (temp in maf.tsbs[!maf.tsbs %in% anno.tsbs]) cat(paste0("  ", temp, "\n"))
    }
    sample.anno <- sample.anno[Tumor_Sample_Barcode %in% maf.tsbs]

    list(
      variants.per.sample = tsb,
      variant.type.summary = vt.cast,
      variant.classification.summary = vc.cast,
      gene.summary = hs.cast,
      summary = summary,
      sample.anno = sample.anno
    )
  }

  # ============================================================
  # fast_validateMaf
  # ============================================================
  fast_validateMaf <- function(maf, rdup = TRUE, isTCGA = isTCGA, chatty = TRUE,
                               zyme = TRUE) {
    if (!isTRUE(zyme)) return(.orig_validateMaf(maf = maf, rdup = rdup,
                                                isTCGA = isTCGA, chatty = chatty))
    required.fields <- c('Hugo_Symbol', 'Chromosome', 'Start_Position', 'End_Position',
                         'Reference_Allele', 'Tumor_Seq_Allele2',
                         'Variant_Classification', 'Variant_Type', 'Tumor_Sample_Barcode')
    for (i in seq_along(required.fields)) {
      colId <- suppressWarnings(grep(paste("^", required.fields[i], "$", sep = ""),
                                     colnames(maf), ignore.case = TRUE))
      if (length(colId) > 0) colnames(maf)[colId] <- required.fields[i]
    }
    missing.fields <- required.fields[!required.fields %in% colnames(maf)]
    if (length(missing.fields) > 0) {
      missing.fields <- paste(missing.fields[1], sep = ',', collapse = ', ')
      stop(paste('missing required fields from MAF:', missing.fields))
    }

    if (rdup) {
      dup_keys <- c('Chromosome', 'Start_Position', 'Tumor_Sample_Barcode',
                    'Reference_Allele', 'Tumor_Seq_Allele2')
      dups <- duplicated(maf, by = dup_keys)
      n_dup <- sum(dups)
      if (n_dup > 0) {
        if (chatty) cat("--Removed", n_dup, "duplicated variants\n")
        maf <- maf[!dups]
      }
    }

    if (any(maf$Hugo_Symbol == "", na.rm = TRUE)) {
      blank_n <- sum(maf$Hugo_Symbol == "", na.rm = TRUE)
      if (chatty) {
        cat('--Found ', blank_n, ' variants with no Gene Symbols\n')
        cat("--Annotating them as 'Unknown' for convenience\n")
      }
      maf$Hugo_Symbol <- ifelse(maf$Hugo_Symbol == "", 'Unknown', maf$Hugo_Symbol)
    }

    na_n <- sum(is.na(maf$Hugo_Symbol))
    if (na_n > 0) {
      if (chatty) {
        cat('--Found ', na_n, ' variants with no Gene Symbols\n')
        cat("--Annotating them as 'Unknown' for convenience\n")
      }
      maf$Hugo_Symbol <- ifelse(is.na(maf$Hugo_Symbol), 'Unknown', maf$Hugo_Symbol)
    }

    if (isTCGA) maf$Tumor_Sample_Barcode <- substr(maf$Tumor_Sample_Barcode, 1, 12)

    silent <- c("3'UTR", "5'UTR", "3'Flank", "Targeted_Region", "Silent", "Intron",
                "RNA", "IGR", "Splice_Region", "5'Flank", "lincRNA",
                "De_novo_Start_InFrame", "De_novo_Start_OutOfFrame",
                "Start_Codon_Ins", "Start_Codon_SNP", "Stop_Codon_Del")
    vc.nonSilent <- c("Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site",
                      "Translation_Start_Site", "Nonsense_Mutation", "Nonstop_Mutation",
                      "In_Frame_Del", "In_Frame_Ins", "Missense_Mutation")
    vt <- c('SNP', 'DNP', 'TNP', 'ONP', 'INS', 'DEL')

    if (chatty) {
      maf.vcs <- unique(as.character(maf[, Variant_Classification]))
      maf.vts <- unique(as.character(maf[, Variant_Type]))
      if (length(maf.vcs[!maf.vcs %in% c(silent, vc.nonSilent)]) > 0) {
        cat("--Non MAF specific values in Variant_Classification column:\n")
        for (temp in maf.vcs[!maf.vcs %in% c(silent, vc.nonSilent)]) cat(paste0("  ", temp, "\n"))
      }
      if (length(maf.vts[!maf.vts %in% vt]) > 0) {
        cat("--Non MAF specific values in Variant_Type column:\n")
        for (temp in maf.vts[!maf.vts %in% vt]) cat(paste0("  ", temp, "\n"))
      }
    }

    if (!is.character(maf$Chromosome))      maf[, Chromosome := as.character(Chromosome)]
    if (!is.numeric(maf$Start_Position))    maf[, Start_Position := as.numeric(as.character(Start_Position))]
    if (!is.numeric(maf$End_Position))      maf[, End_Position := as.numeric(as.character(End_Position))]

    tsb <- maf$Tumor_Sample_Barcode
    if (!is.character(tsb)) tsb <- as.character(tsb)
    tsb_uniq   <- unique(tsb)
    tsb_levels <- tsb_uniq[.dt_forderv(tsb_uniq)]
    tsb_codes  <- data.table::chmatch(tsb, tsb_levels)
    data.table::set(maf, j = "Tumor_Sample_Barcode", value = tsb_codes)
    data.table::setattr(maf$Tumor_Sample_Barcode, "levels", tsb_levels)
    data.table::setattr(maf$Tumor_Sample_Barcode, "class", "factor")

    maf
  }

  # ============================================================
  # fast_read.maf
  # ============================================================
  fast_read.maf <- function(maf, clinicalData = NULL, rmFlags = FALSE,
                            removeDuplicatedVariants = TRUE, useAll = TRUE,
                            gisticAllLesionsFile = NULL, gisticAmpGenesFile = NULL,
                            gisticDelGenesFile = NULL, gisticScoresFile = NULL,
                            cnLevel = 'all', cnTable = NULL, isTCGA = FALSE,
                            vc_nonSyn = NULL, verbose = TRUE, zyme = TRUE) {
    # Defer unsupported / rare branches to upstream verbatim.
    if (!isTRUE(zyme) || !is.null(gisticAllLesionsFile) || !is.null(cnTable) ||
        isTCGA || !useAll) {
      return(.orig_read_maf(maf=maf, clinicalData=clinicalData, rmFlags=rmFlags,
        removeDuplicatedVariants=removeDuplicatedVariants, useAll=useAll,
        gisticAllLesionsFile=gisticAllLesionsFile, gisticAmpGenesFile=gisticAmpGenesFile,
        gisticDelGenesFile=gisticDelGenesFile, gisticScoresFile=gisticScoresFile,
        cnLevel=cnLevel, cnTable=cnTable, isTCGA=isTCGA, vc_nonSyn=vc_nonSyn, verbose=verbose))
    }

    if (is.data.frame(x = maf)) {
      maf <- data.table::as.data.table(maf)
    } else {
      if (verbose) cat('-Reading\n')
      # data.table >= 1.14 reads .gz natively via `file=`; the old `cmd=gunzip -c`
      # branch was Unix-only (no `gunzip` on Windows). Single path for both.
      maf <- data.table::fread(file=maf, sep="\t", stringsAsFactors=FALSE,
                               verbose=FALSE, data.table=TRUE,
                               showProgress=!grepl("\\.gz$", maf),
                               header=TRUE, fill=TRUE, skip="Hugo_Symbol", quote="")
    }
    if (verbose) cat("-Validating\n")
    # Resolve via namespace so the active maftools:::validateMaf (our patched
    # version when activated, original otherwise) is what runs.
    maf <- utils::getFromNamespace("validateMaf", "maftools")(
      maf = maf, isTCGA = isTCGA,
      rdup = removeDuplicatedVariants, chatty = verbose)
    if (is.null(vc_nonSyn)) {
      vc.nonSilent <- c("Frame_Shift_Del","Frame_Shift_Ins","Splice_Site",
                        "Translation_Start_Site","Nonsense_Mutation","Nonstop_Mutation",
                        "In_Frame_Del","In_Frame_Ins","Missense_Mutation")
    } else vc.nonSilent <- vc_nonSyn
    if (is.logical(rmFlags)) {
      if (rmFlags) {
        flags <- .orig_flags(top=20)
        cat("-Removing", length(flags), "FLAG genes\n")
        maf <- maf[!Hugo_Symbol %in% flags]
      }
    } else if (is.numeric(rmFlags)) {
      flags <- .orig_flags(top=rmFlags)
      cat("-Removing", length(flags), "FLAG genes\n")
      maf <- maf[!Hugo_Symbol %in% flags]
    }
    maf.silent <- maf[!Variant_Classification %in% vc.nonSilent]
    if (nrow(maf.silent) > 0) {
      maf <- maf[Variant_Classification %in% vc.nonSilent]
      if (verbose) cat(paste0('-Silent variants: ', nrow(maf.silent)), '\n')
    }
    if (nrow(maf) == 0) stop("No non-synonymous mutations found\nCheck `vc_nonSyn` argument in `read.maf` for details")
    .fast_factor_dt_col(maf, "Variant_Classification")
    .fast_factor_dt_col(maf, "Variant_Type")
    if (verbose) cat('-Summarizing\n')
    # maftools::MAF dispatches into summarizeMaf via the namespace chain, so
    # patched summarizeMaf is what runs when active.
    m <- .orig_MAF(nonSyn=maf, syn=maf.silent, clinicalData=clinicalData,
                   verbose=verbose)
    m
  }

  # ============================================================
  # Smoke recipe (load / call / save) — mirrors reference.R / pipeline/run.R
  # ============================================================
  # Fair-comparison boundary (per 4_package.md):
  #
  #   - read.maf's signature accepts a .maf.gz path; the fread / gunzip-pipe
  #     happens INSIDE the function we patch — both baseline and patched run
  #     the same I/O path on the same input. So the path-resolution step
  #     (yaml::read_yaml + path lookup) is user-side prep -> `load`, and the
  #     `read.maf(path, verbose = FALSE)` call itself -> `call`.
  #   - We don't pre-read the MAF into memory in `load`: the patch
  #     specifically accelerates read.maf's full pipeline (reader + validate +
  #     summarize), so passing a path is the documented API the user calls,
  #     and timing it is what the headline number claims.
  #   - data.table threading: NOT set in load. data.table::setDTthreads picks
  #     up DT_NUM_THREADS / OMP_NUM_THREADS from the env; both baseline and
  #     patched subprocesses inherit the parent's env identically, so the
  #     thread budget is matched. The patch itself does not override threads.

  .maftools_smoke_load <- function(task_dir, tier) {
    task <- yaml::read_yaml(file.path(task_dir, "task.yaml"))
    ds <- Filter(function(d) identical(d$tier, tier), task$datasets)
    if (length(ds) == 0L) {
      stop("no dataset for tier '", tier, "' in task.yaml")
    }
    data_path <- resolve_dataset_path(task_dir, ds[[1]]$path)
    list(data_path = data_path, tier = tier)
  }

  .maftools_smoke_call <- function(inputs) {
    # Single canonical invocation — matches pipeline/run.R / reference.R
    # (read.maf with verbose=FALSE on the same .maf.gz path).
    m <- maftools::read.maf(maf = inputs$data_path, verbose = FALSE)
    m
  }

  .maftools_smoke_save <- function(result, dir, tier = "tiny", ...) {
    # evaluate.R reads pipeline/maf.rds + reference_output_<tier>/maf.rds and
    # checks the seven slots below.
    saveRDS(list(
      data           = result@data,
      maf.silent     = result@maf.silent,
      variants.per.sample           = result@variants.per.sample,
      variant.type.summary          = result@variant.type.summary,
      variant.classification.summary = result@variant.classification.summary,
      gene.summary   = result@gene.summary,
      summary        = result@summary
    ), file.path(dir, "maf.rds"))
  }

  register_patch(
    name     = "maftools",
    upstream = "maftools",
    targets  = list(
      read.maf     = fast_read.maf,
      validateMaf  = fast_validateMaf,
      summarizeMaf = fast_summarizeMaf
    ),
    smoke = list(
      load = .maftools_smoke_load,
      call = .maftools_smoke_call,
      save = .maftools_smoke_save
    ),
    tested_against = "maftools 2.24.0",
    tested_upstream_versions = list(maftools = "2.24.0")
  )
}
