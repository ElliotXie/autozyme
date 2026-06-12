// C++ kernel for autozyme's maftools patch.
//
// Lifted from test_general_bio/test_maftools/pipeline/run.R (inline
// Rcpp::sourceCpp block, ~lines 22-35). The single exported kernel
// `zyme_fill_dcast` fills an N_rows x N_cols integer matrix from
// (row_code, col_code, count) triplets — substitutes data.table::dcast
// with drop=FALSE for the variant_classification and variant_type
// per-sample summaries in fast_summarizeMaf.
//
// Header-only (Rcpp), no parallelism — fill is O(nrow(triplets)).

// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
IntegerMatrix zyme_fill_dcast(IntegerVector row_codes,
                              IntegerVector col_codes,
                              IntegerVector counts,
                              int n_rows,
                              int n_cols) {
  IntegerMatrix M(n_rows, n_cols);
  int* mp = INTEGER(M);
  int n = row_codes.size();
  for (int i = 0; i < n; i++) {
    mp[(col_codes[i] - 1) * n_rows + (row_codes[i] - 1)] = counts[i];
  }
  return M;
}
