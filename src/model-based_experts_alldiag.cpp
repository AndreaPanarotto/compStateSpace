#include <RcppArmadillo.h>
#include <ctime>
#include <typeinfo>
#include "group_assignment_experts_alldiag.h"
#include "Shumway_EM_mixture_alldiag.h"

using namespace arma;
using namespace Rcpp;

//' Model-based clustering of time series based on a state space representation
//'
//' \code{mb_clust} performes the model-based clustering of time series, relying on a state-space representation and a mixture of experts model for the inclusion of series-specific covariates. The method was initially developed for clustering compositional time series according to their evolution in the simplex, by feeding the function their ilr-transformed version. 
//'
//' @param z_list A list of time series. Each element of the list should be a matrix where each row represents a time observation.
//' @param a0_l List of initial states (expectations).
//' @param P0_l List of variances of the initial states.
//' @param g_T List of initial group-specific matrices of state transformation. The matrices should be diagonal.
//' @param g_Q List of initial group-specific variance matrices of the state error. The matrices should be diagonal.
//' @param g_H List of initial group-specific variance matrices of the observation error. The matrices should be diagonal.
//' @param x_list List of series-specific covariates.
//' @param g_gamma List of initial values for the component weights parameters.
//' @param clust_maxit Maximum number of EM iterations.
//' @param clust_toll Tolerance for the relative difference of log-likelihoods criterion to stop the computation.
//' @param verbose A non-negative integer to indicate if and how frequently print the likelihood as the EM iterations proceed.
//' @details
//' For compositional time series, the ilr transformation of the series should be provided. \code{z_list <- lapply(y_list, comp_to_ilr)}, where \code{y_list} is a list of matrices with D (= number of parts) columns each, and rows depending on the length of the series, should be enough.
//' 
//' \code{g_T}, \code{g_Q} and \code{g_H} should be lists of length K, where K is the number of groups. Either these, or the values in \code{g_gamma}`, should all be different to ensure the avoidance of identical groups. We recommend providing different starting values for \code{g_T}, \code{g_Q} and \code{g_H}, rather than \code{g_gamma}. A suggested initialization involves the use of a quick clustering procedure (e.g. using \link[dtwclust]{tsclust} from the \code{dtwclust} package) and then some iterations of \link{shumway_EM_list} applied separately to the obtained clusters.
//' 
//' \code{a0_l} and \code{P0_l} should be one of the following: a list (of length K) of lists (each of length equal to the number of series N), or a list of length N. In the former case, the initial states are assumed to be provided for each series for each component. It is useful, for instance, to continue the computation after having obtained a result from the model, if some more iterations are needed. In the latter case, a starting point is assumed to be given for each series, and it is used for the computation for each component. For example, if the initialization with \link{shumway_EM_list} is employed, the obtained results for each series could be used.
//' 
//' \code{x_list} should be of length N, with each covariate vector being series-specific rather than "time-stamp-specific". Every element of the list should be a (P+1)-dimensional vector with 1 as first element, for the intercept.
//' 
//' \code{g_gamma} should be a list of length K, with each element a (P+1)-dimensional vector. The first element is never updated for identifiability reasons, so it should act as reference and a vector of zeros is suggested for it. If the elements in \code{g_T}, \code{g_Q}, and \code{g_H} have been differentiated with an initialization procedure, and no previous knowledge is involved, we suggest to use vectors of zeros for all the component weights parameters, in a sort of "non-informative prior" fashion.
//' 
//' If \code{verbose} is 0, nothing is printed during the computation. If it is a positive integer, the current likelihood and likelihood difference from the previous iteration is printed every \code{verbose} iterations.
//' 
//' 
//' @return A list containing:
//' \describe{
//' \item{groups_T}{A list with the final group-specific matrices of state transformation.}
//' \item{groups_Q}{A list with the final group-specific variance matrices of the state error.}
//' \item{groups_H}{A list with the final group-specific variance matrices of the observation error.}
//' \item{groups_a0_list}{A list of length K of lists, each of which shows the obtained initial states of each series for the corresponding component.}
//' \item{groups_P0_list}{A list of length K of lists, each of which shows the variance of the obtained initial states of each series for the corresponding component.}
//' \item{code}{A value indicating convergence (0) or not (1) of the algorithm.}
//' \item{clust_it}{Number of performed EM iterations.}
//' \item{total_lik}{Final complete log-likelihood.}
//' \item{clust_lik_diff}{Relative log-likelihood difference at the last iteration.}
//' \item{groups_probs_list}{A list with the probability of each series to belong to each component.}
//' \item{gamma}{A list with the final component weight parameters.}
//' \item{time}{Total computational time.}
//' }
//' @export
// [[Rcpp::export]]
List mb_clust(const List z_list,
             const List a0_l,
             const List P0_l,
             const List g_T,
             const List g_Q,
             const List g_H,
             const List x_list, // should contain 1 in the first component for intercept
             const List g_gamma,
             int clust_maxit = 1e3,
             double clust_toll = 1e-6,
             int verbose = 0)
    {
        const clock_t starttime = clock();
        List groups_T = clone(g_T), groups_Q = clone(g_Q), groups_H = clone(g_H); // needed or it would modify the original list
        List gamma = clone(g_gamma);
        int n_series = z_list.size();
        // int n, d = as<mat>(z_list[0]).n_cols + 1;
        int p = as<vec>(gamma[0]).n_elem - 1;
        int K = groups_T.size();
        
        double total_lik = -std::numeric_limits<float>::max();
        double new_lik, lik_diff = clust_toll + 1;
        
        int clust_it = 0;
        
        
        List groups_a0_list(K);
        List groups_P0_list(K);
        // if you give a list as long as the number of groups, just copy
        // in this case it should be a list of lists (K --> n_series)
        if(a0_l.size() == K)
        {
            groups_a0_list = clone(a0_l); // needed or it would modify the original list
            groups_P0_list = clone(P0_l);
        }
        else{
        // otherwise, the initial list is assumed the same for all groups
        // in this case, it should be a list of n_series
            for (int kk = 0; kk < K; kk++) {
                groups_a0_list[kk] = clone(a0_l); // needed or it would modify the original list
                groups_P0_list[kk] = clone(P0_l);
            }
            
        }
        
        // first probabilities computation
        List groups_probs_list(n_series);
        
        vec probs(n_series);
        vec covs_prod(K);
        List eta(n_series);
        List new_gamma(K);
        List group_result;
        List new_groups_T(K), new_groups_Q(K), new_groups_H(K), new_groups_a0_list(K), new_groups_P0_list(K);
        vec Q_grad(p + 1);
        mat Hess(p + 1, p + 1);
        double eta_ik;
        while (clust_it < clust_maxit) {

            // verbose stuff?
            if (verbose && clust_it % verbose == 0)
            {
                Rcout << "\n\nAfter outer iteration " << clust_it << "\nTotal logLikelihood = " << total_lik << endl;
                Rcout << "LikDiff = " << lik_diff << endl;
                // Rcout << "eta" << trans(eta) << endl;
            }
            clust_it++;
        
            // OUTER E-Phase
            for(int ii = 0; ii < n_series; ii++)
            {
                for(int kk = 0; kk < K; kk++)
                {
                    covs_prod(kk) = (trans(as<vec>(x_list[ii])) * as<vec>(gamma[kk])).eval()(0);
                }
                // softmax function from group assignment mixture
                eta[ii] = sample_prob(covs_prod);
            }
            // This is the computation of the z_ig in the me
            groups_probs_list = group_assignment_forward(z_list,
                                                         groups_a0_list,
                                                         groups_P0_list,
                                                         groups_T,
                                                         groups_Q,
                                                         groups_H,
                                                         eta);
            // OUTER Maximization phase: Shumway E-M
            new_lik = 0;
            new_groups_T = clone(groups_T);
            new_groups_Q = clone(groups_Q);
            new_groups_H = clone(groups_H);
            new_groups_a0_list = clone(groups_a0_list);
            new_groups_P0_list = clone(groups_P0_list);
            
            for(int kk = 0; kk < K; kk++)
            {
                // Rcout << "k" << kk << endl;
                Q_grad.zeros();
                Hess.zeros();
                for(int ii = 0; ii < n_series; ii++)
                {
                    // probabilities for all series to belong to group kk --> weights
                    probs[ii] = as<vec>(groups_probs_list[ii]).eval()(kk);
                    eta_ik = as<vec>(eta[ii]).eval()(kk);
                    // you can already update the eta part of the likelihood, since eta and z will be untouched by the EM
                    new_lik += log(eta_ik) * probs[ii];
                    // elements for gamma update can be computed here as well
                    if(kk>0)
                    {
                        Q_grad += as<vec>(x_list[ii]) * (probs[ii] - eta_ik);
                        Hess -= eta_ik * (1-eta_ik) * as<vec>(x_list[ii]) * trans(as<vec>(x_list[ii]));
                    }
                }
                group_result = shumway_EM_list(
                                    z_list,
                                    groups_a0_list[kk],
                                    groups_P0_list[kk],
                                    groups_T[kk],
                                    groups_Q[kk],
                                    groups_H[kk],
                                    probs,
                                    1, // EM_toll is actually useless for 1 iteration
                                    1,
                                    0); // verbose EM is 0 
                // returns TT, Q, H, a0_list, P0_list, EM_it, diff
                
                new_lik += as<double>(group_result["log_lik"]);
                // system matrices for group assignment and next loop updates
                new_groups_T[kk] = group_result["TT"];
                new_groups_H[kk] = group_result["H"];
                new_groups_Q[kk] = group_result["Q"];
                // initial elements for group assignment and next loop updates
                new_groups_a0_list[kk] = as<List>(group_result["a0_list"]);
                new_groups_P0_list[kk] = as<List>(group_result["P0_list"]);
                // gamma update
                if (kk>0) // gamma of the first group is fixed at 0
                {
                    try {
                        // Block of code to try
                        gamma[kk] = as<vec>(gamma[kk]) + inv_sympd(-Hess) * Q_grad;
                    }
                    catch (std::runtime_error& e) {
                        // Block of code to handle errorsList::create(Named("k") = kk,
                        return(List::create(Named("gamma") = gamma,
                            Named("eta") = eta,
                            Named("probs") = probs,
                            Named("Q_grad") = Q_grad,
                            Named("Hess") = Hess));
                    }
                }
            }
            
            lik_diff = relativeError(new_lik, total_lik, clust_toll / 10);
            total_lik = new_lik;
            if(lik_diff < clust_toll)
            {
                return(List::create(Named("groups_T") = groups_T,
                                    Named("groups_Q") = groups_Q,
                                    Named("groups_H") = groups_H,
                                    Named("groups_a0_list") = groups_a0_list,
                                    Named("groups_P0_list") = groups_P0_list,
                                    Named("code") = 0,
                                    Named("clust_it") = clust_it,
                                    Named("total_lik") = total_lik,
                                    Named("clust_lik_diff") = lik_diff,
                                    Named("groups_probs_list") = groups_probs_list,
                                    Named("gamma") = gamma,
                                    Named("time") = float( clock() - starttime ) /  CLOCKS_PER_SEC
                ));
            }
            groups_T = clone(new_groups_T);
            groups_Q = clone(new_groups_Q);
            groups_H = clone(new_groups_H);
            groups_a0_list = clone(new_groups_a0_list);
            groups_P0_list = clone(new_groups_P0_list);

        }
        
        return(List::create(Named("groups_T") = groups_T,
                            Named("groups_Q") = groups_Q,
                            Named("groups_H") = groups_H,
                            Named("groups_a0_list") = groups_a0_list,
                            Named("groups_P0_list") = groups_P0_list,
                            Named("code") = 1,
                            Named("clust_it") = clust_it,
                            Named("total_lik") = total_lik,
                            Named("clust_lik_diff") = lik_diff,
                            Named("groups_probs_list") = groups_probs_list,
                            Named("gamma") = gamma,
                            Named("time") = float( clock() - starttime ) /  CLOCKS_PER_SEC
        ));
    }