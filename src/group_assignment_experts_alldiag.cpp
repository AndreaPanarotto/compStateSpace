#include <RcppArmadillo.h>
// #include <cstdlib>
#include<iostream>
#include<string>
#include "Shumway_KFS_alldiag.h"

using namespace Rcpp;
using namespace arma;
using namespace std;

vec sample_prob(vec log_vec)
{
    log_vec = exp(log_vec - log_vec.max()) + 1e-10; // correction
    return((log_vec) / accu(log_vec));
}

List group_assignment_forward(const List z_list,
                             const List groups_a0_list,
                             const List groups_P0_list,
                             const List groups_T,
                             const List groups_Q,
                             const List groups_H,
                             const List eta)
{
    int n_series = z_list.size();
    mat z = as<mat>(z_list[0]);
    int d = z.n_cols + 1;
    
    List groups_probs_list(n_series);
    int K = groups_T.size();

    vec prob_series(groups_T.size());

    vec mu0(d-1);
    vec Sigma0(d-1);
    List filtered;
    vec TT(d - 1), H(d - 1), Q(d - 1);
    
    for (int ii = 0; ii < n_series; ii++)
    {
        prob_series.zeros();
        z = as<mat>(z_list[ii]);
        for (int kk = 0; kk < K; kk++) {
            mu0 = as<vec>(as<List>(groups_a0_list[kk])[ii]);
            Sigma0 = as<vec>(as<List>(groups_P0_list[kk])[ii]);
            TT = as<vec>(groups_T[kk]);
            Q  = as<vec>(groups_Q[kk]);
            H  = as<vec>(groups_H[kk]);
            filtered = k_filter(z,
                                mu0,
                                Sigma0,
                                TT,
                                Q,
                                H);
            prob_series(kk) = log(as<vec>(eta[ii]).eval()(kk)) + as<double>(filtered["log_lik"]);
        }
        
        groups_probs_list[ii] = sample_prob(prob_series);
            
    }
    return(groups_probs_list);
}