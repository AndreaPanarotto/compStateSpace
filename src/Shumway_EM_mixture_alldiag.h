# ifndef SHUMWAY_EM_MIXTURE_ALLDIAG_H
# define SHUMWAY_EM_MIXTURE_ALLDIAG_H

#include <RcppArmadillo.h>
#include <cstdlib>
#include "Shumway_KFS_alldiag.h"

using namespace Rcpp;
using namespace arma;

double relativeError(double x,
                     double y,
                     double tolerance = 1e-8);
mat relativeError(mat x,
                  mat y,
                  double tolerance = 1e-8);

List shumway_EM(mat z,
                vec a0,
                mat P0,
                mat TT,
                mat Q,
                mat H,
                double EM_toll = 1e-4,
                int EM_maxit = 1e2,
                int verbose = 0);

List shumway_EM_list(const List z_list,
                     const List a0_l,
                     const List P0_l,
                     vec TT,
                     vec Q,
                     vec H,
                     vec probs,
                     double EM_toll = 1e-4,
                     int EM_maxit = 1e2,
                     int verbose = 0);

#endif