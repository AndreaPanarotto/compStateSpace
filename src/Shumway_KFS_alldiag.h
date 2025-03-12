# ifndef SHUMWAY_KFS_ALL_DIAG_H
# define SHUMWAY_KFS_ALL_DIAG_H

#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// mat rcpp_NearPD(mat m);

List k_filter(const mat z,
              const vec a0,
              const vec P0,
              const vec H, // don't know how to give them null defaults
              const vec TT,
              const vec Q);

List k_smoother(const List filtered,
                const vec TT);

#endif
