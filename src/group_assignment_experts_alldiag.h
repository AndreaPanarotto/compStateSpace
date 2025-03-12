# ifndef GROUP_ASSIGNMENT_EXPERTS_ALLDIAG_H
# define GROUP_ASSIGNMENT_EXPERTS_ALLDIAG_H

#include <RcppArmadillo.h>
#include<iostream>
#include<string>
#include "Shumway_KFS_alldiag.h"

using namespace Rcpp;
using namespace arma;
using namespace std;


vec sample_prob(vec log_vec);


List group_assignment_forward(const List z_list,
                             const List groups_a0_list,
                             const List groups_P0_list,
                             const List groups_T,
                             const List groups_Q,
                             const List groups_H,
                             const List eta);

#endif