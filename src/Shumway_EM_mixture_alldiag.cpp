#include <RcppArmadillo.h>
#include <cstdlib>
#include "Shumway_KFS_alldiag.h"

using namespace Rcpp;
using namespace arma;

double relativeError(double x,
                     double y,
                     double tolerance = 1e-8) {
    double diff = std::abs(x - y);
    double mag = std::max(std::abs(x), std::abs(y));
    bool b = (mag > tolerance);
    return(diff / (mag * int(b) + (1 - int(b))));
}

mat relativeError(mat x,
                  mat y,
                  double tolerance = 1e-8) {
    mat diff = abs(x - y);
    mat mag = max(abs(x), abs(y));
    umat b = (mag > tolerance);
    return(diff / (mag % b + (1 - b)));
}

//' Estimation of the state-space model system matrices with the EM procedure
//' from Shumway and Stoffer (1982)
//'
//' \code{shumway_EM} and \code{shumway_EM_list} compute the estimates of the system matrices T, Q, H for time series modeled according to the following state-space model:
//' \deqn{\boldsymbol{\alpha}_t  = \mathbf{T} \boldsymbol{\alpha}_{t-1} + \boldsymbol{\xi}_t\,, \quad \boldsymbol{\xi}_t \sim \mathcal{N}\left(0,\  \mathbf{Q}\right),}
//' \deqn{\mathbf{z}_t  = \boldsymbol{\alpha}_t + \boldsymbol{\zeta}_t\,, \qquad \boldsymbol{\zeta}_t\sim \mathcal{N}\left(0,\ \mathbf{H}\right).} 
//' IMPORTANT: T, Q and H are assumed to be diagonal.
//' 
//' @name ShSt_EM

//' @describeIn ShSt_EM Computes the estimate for a single time series.
//' @param z A matrix representing time series. Each row represents a time
//' observation.
//' @param a0 Initial state (expectation).
//' @param P0 Diagonal of the variance matrix of the initial state. 
//' @param TT The diagonal of the matrix of state transformation.
//' @param Q  The diagonal of the variance matrix of the state error.
//' @param H  The diagonal of the variance matrix of the observation error.
//' @param EM_toll Tolerance for the relative difference of log-likelihoods criterion to stop the computation.
//' @param EM_maxit Maximum number of EM iterations.
//' @param verbose A non-negative integer to indicate if and how frequently print 
//' the current system matrices and likelihood as the EM iterations proceed.
//' 
//' @return \code{shumway_EM} returns a list containing:
//' \describe{
//' \item{TT}{Estimated matrix of state transformation.}
//' \item{Q}{Estimated variance matrix of the state error.}
//' \item{H}{Estimated variance matrix of the observation error.}
//' \item{a0}{Estimated initial state.}
//' \item{P0}{Estimated variance of the obtained initial state.}
//' \item{code}{A value indicating the exit condition of the algorithm. 0 means
//' convergence, 1 means that the maximum of iterations has been reached,
//' 2 means a decrease (instead of increase) of the log-likelihood, and 3 means
//' a \code{nan} likelihood.}
//' \item{EM_it}{Number of performed EM iterations.}
//' \item{log_lik}{Final marginal log-likelihood.}
//' \item{diff}{Relative log-likelihood difference at the last iteration.}
//' } 
//' @export
// [[Rcpp::export]]
List shumway_EM(arma::mat z,
                arma::vec a0,
                arma::vec P0,
                arma::vec TT,
                arma::vec Q,
                arma::vec H,
                double EM_toll = 1e-4,
                int EM_maxit = 1e2,
                int verbose = 0) {
    int n = z.n_rows, d = z.n_cols + 1;
    int EM_it = 0;
    double log_lik = -std::numeric_limits<float>::max();
    double lik_diff = -1;
    
    List filtered, smoothed;
    vec S11(d - 1), S10(d - 1), S00(d - 1); 
    vec old_TT(d - 1), old_Q(d - 1), old_H(d - 1);
    vec new_TT(d - 1), new_Q(d - 1), new_H(d - 1);
    vec old_a0;
    vec old_P0;
    vec positive_variances(d-1);
    positive_variances.fill(1e-8);
    while (EM_it < EM_maxit)
    {
        if (verbose && EM_it % verbose == 0)
        {
            Rcout << "\n\nAfter inner iteration " << EM_it << "\nT:\n" << TT;
            Rcout << ("\nQ:\n") << Q;
            Rcout << ("\nH:\n") << H;
            Rcout << "\nlogLikelihood = " << log_lik << endl;
            Rcout << "\nLikDiff = " << lik_diff << endl << endl;
        }
        EM_it++;
        // Expectation phase: KFS to get smoothed estimate for eta and smoothers stuff
        filtered = k_filter(z,
                            a0,
                            P0,
                            TT,
                            Q,
                            H);
        if(std::isnan(as<double>(filtered["log_lik"])))
        {
            return(List::create(Named("TT") = old_TT,
                                Named("Q") = old_Q,
                                Named("H") = old_H,
                                Named("a0") = old_a0,
                                Named("P0") = old_P0,
                                Named("code") = 3,
                                Named("EM_it") = EM_it-1,
                                Named("log_lik") = log_lik,
                                Named("diff") = lik_diff
            ));
        }
        if(as<double>(filtered["log_lik"]) < log_lik)
        {
            return(List::create(Named("TT") = old_TT,
                                Named("Q") = old_Q,
                                Named("H") = old_H,
                                Named("a0") = old_a0,
                                Named("P0") = old_P0,
                                Named("code") = 2,
                                Named("EM_it") = EM_it-1,
                                Named("log_lik") = log_lik,
                                Named("diff") = lik_diff
            ));
        }
        lik_diff = relativeError(filtered["log_lik"], log_lik, EM_toll / 10);
        log_lik = filtered["log_lik"];
        if(lik_diff < EM_toll)
        {
            return(List::create(Named("TT") = TT,
                                Named("Q") = Q,
                                Named("H") = H,
                                Named("a0") = a0,
                                Named("P0") = P0,
                                Named("code") = 0,
                                Named("EM_it") = EM_it,
                                Named("log_lik") = log_lik,
                                Named("diff") = lik_diff
            ));
        }
        smoothed = k_smoother(filtered, TT);
        
        old_a0 = a0;
        old_P0 = P0;
        a0 = trans(as<mat>(smoothed["alpha"]).row(0));
        P0 = trans(as<mat>(smoothed["V"]).row(0));
        
        S11.zeros();
        S10.zeros();
        S00.zeros();
        new_H.zeros();
        for (int t = 0; t < n; t++) {
            S11 += trans(as<mat>(smoothed["alpha"]).row(t+1) % as<mat>(smoothed["alpha"]).row(t+1) + as<mat>(smoothed["V"]).row(t+1));
            S10 += trans(as<mat>(smoothed["alpha"]).row(t+1) % as<mat>(smoothed["alpha"]).row(t) + as<mat>(smoothed["P_lag"]).row(t));
            S00 += trans(as<mat>(smoothed["alpha"]).row(t) % as<mat>(smoothed["alpha"]).row(t) + as<mat>(smoothed["V"]).row(t));
            new_H += trans((z.row(t) - as<mat>(smoothed["alpha"]).row(t+1)) % (z.row(t) - as<mat>(smoothed["alpha"]).row(t+1)) + as<mat>(smoothed["V"]).row(t+1));
        }
        // Maximization phase to get new T, Q, H
        new_TT = S10 / S00;
        
        new_Q = positive_variances + (S11 - new_TT % S10) / n;
        
        new_H = positive_variances + new_H / n;
        
        old_TT = TT;
        old_Q = Q; 
        old_H = H; 
        TT = new_TT;
        Q = new_Q;
        H = new_H;
        
    }
    return(List::create(Named("TT") = TT,
                        Named("Q") = Q,
                        Named("H") = H,
                        Named("a0") = a0,
                        Named("P0") = P0,
                        Named("code") = 1,
                        Named("EM_it") = EM_it,
                        Named("log_lik") = filtered["log_lik"],
                        Named("diff") = lik_diff
    ));
}



//' @describeIn ShSt_EM computes the estimates for a list of series assumed to 
//' share the same system matrices. A weight vector \code{probs} can be provided 
//' if the series do not contribute in the same way to the total likelihood,
//' for example in a mixture model.
//' @param z_list A list of time series. Each element of the list should be a
//' matrix where each row represents a time observation.
//' @param a0_l List of initial states (expectations). Check the `Details`
//' @param P0_l List of variances of the initial states. Check the `Details`
//' @param TT Matrix of state transformation. It should be diagonal.
//' @param Q Variance matrix of the state error. It should be diagonal.
//' @param H Variance matrix of the observation error. It should be diagonal.
//' @param probs Weight vector, containing a value in [0,1] for each of the time series. Check the `Details`.
//' @param EM_toll Tolerance for the relative difference of log-likelihoods criterion to stop the computation.
//' @param EM_maxit Maximum number of EM iterations.
//' @param verbose A non-negative integer to indicate if and how frequently print 
//' the current system matrices and likelihood as the EM iterations proceed.
//' @details
//' For compositional time series, the ilr transformation of the series should be provided. If \code{y_list} is a list of matrices representing compositional times series, with D (= number of parts) columns each, and rows depending on the length of the series, then \code{z_list <- lapply(y_list, \link{comp_to_ilr})}, should be enough.
//' 
//' \code{a0_l} and \code{P0_l} should be a list of length equal to the number of series N. Each element should contain a vector with the same size of a single time-stamp observation: the starting point for \code{a0_l} and the diagonal of the corresponding variance matrix for \code{P0_l}.
//' 
//' In the mixture case, \code{probs} should contain a value between 0 and 1 for each of the time series, corresponding to the probability of the series to belong to the currently considered component. This happens, for example, when \link{mb_clust} is called. If, instead, there are no mixtures involved and the series are all supposed to behave according to the same system matrices, \code{probs} should be a N-dimensional vector of ones.
//'  
//' @return \code{shumway_EM_list} returns a list containing:
//' \describe{
//' \item{TT}{Estimated matrix of state transformation.}
//' \item{Q}{Estimated variance matrix of the state error.}
//' \item{H}{Estimated variance matrix of the observation error.}
//' \item{a0_list}{Estimated initial state.}
//' \item{P0_list}{Estimated variance of the obtained initial state.}
//' \item{code}{A value indicating the exit condition of the algorithm. 0 means convergence, 1 means that the maximum of iterations has been reached, 2 means a decrease (instead of increase) of the log-likelihood, and 3 means a \code{nan} likelihood.}
//' \item{EM_it}{Number of performed EM iterations.}
//' \item{na_lik_series}{Index of the series that gives the \code{nan} likelihood, if \code{code} = 3. -1 otherwise}
//' \item{log_lik}{Final weighted marginal log-likelihood.}
//' \item{diff}{Relative log-likelihood difference at the last iteration.}
//' } 
//' @references
//' Shumway, R. H. and Stoffer, D. S. (1982) An approach to time series smoothing and forecasting using the EM algorithm. Journal of Time Series Analysis 3(4), 253–264.
//' 
//' Shumway, R. H. and Stoffer, D. S. (2000) Time Series Analysis and its Applications. Volume 3. Springer
//' @export
// [[Rcpp::export]]
List shumway_EM_list(const List z_list,
                     const List a0_l,
                     const List P0_l,
                     arma::vec TT,
                     arma::vec Q,
                     arma::vec H,
                     arma::vec probs,
                     double EM_toll = 1e-4,
                     int EM_maxit = 1e2,
                     int verbose = 0) {
    List a0_list = clone(a0_l), P0_list = clone(P0_l); // needed or it would modify the original a0_list
    // int n_series = z_list.size();
    
    mat z = as<mat>(z_list[0]);
    int n, d = z.n_cols + 1;
    int ii;
    
    int EM_it = 0;
    double log_lik = -std::numeric_limits<float>::max();
    double new_log_lik, lik_diff = -1;
    
    vec positive_probs = conv_to<vec>::from(find(probs > 0));
    double weighted_len = 0;
    for(vec::iterator it = positive_probs.begin(); it != positive_probs.end(); ++it)
    {
        weighted_len += as<mat>(z_list[(*it)]).n_rows * probs.eval()(*it);
    }
    List filtered, smoothed;
    vec S11(d - 1), S10(d - 1), S00(d - 1); 
    vec weighted_S11(d - 1), weighted_S10(d - 1), weighted_S00(d - 1); 
    vec old_TT(d - 1), old_Q(d - 1), old_H(d - 1);
    vec new_TT(d - 1), new_Q(d - 1), new_H(d - 1), series_H(d-1);
    // exp una alla volta, max tutte insieme
    List old_a0_list;
    List old_P0_list;
    vec positive_variances(d-1);
    positive_variances.fill(1e-8);
    while (EM_it < EM_maxit)
    {
        if (verbose && EM_it % verbose == 0)
        {
            Rcout << "\n\nAfter inner iteration " << EM_it << "\nT:\n" << TT;
            Rcout << ("\nQ:\n") << Q;
            Rcout << ("\nH:\n") << H;
            Rcout << "\nlogLikelihood = " << log_lik << endl;
            Rcout << "\nLikDiff = " << lik_diff << endl << endl;
        }
        EM_it++;
        
        // Initialize quantities to update
        weighted_S11.zeros();
        weighted_S10.zeros();
        weighted_S00.zeros();
        new_H.zeros();
        
        new_log_lik = 0;
        old_a0_list = clone(a0_list);
        old_P0_list = clone(P0_list);
        for(unsigned int it = 0; it < positive_probs.size(); it++)
        {
            ii = positive_probs.eval()(it); // series number
            z = as<mat>(z_list[ii]);
            n = z.n_rows;
            // Expectation phase: KFS to get smoothed estimate for eta
            filtered = k_filter(z,
                                as<vec>(a0_list[ii]),
                                as<vec>(P0_list[ii]),
                                TT,
                                Q,
                                H);
            if(std::isnan(as<double>(filtered["log_lik"])))
            {
                return(List::create(Named("TT") = old_TT,
                                    Named("Q") = old_Q,
                                    Named("H") = old_H,
                                    Named("a0_list") = old_a0_list,
                                    Named("P0_list") = old_P0_list,
                                    Named("code") = 3,
                                    Named("EM_it") = EM_it-1,
                                    Named("na_lik_series") = ii,
                                    Named("log_lik") = log_lik,
                                    Named("diff") = lik_diff
                ));
            }
            new_log_lik +=  probs.eval()(ii) * as<double>(filtered["log_lik"]);
            smoothed = k_smoother(filtered, TT);
            a0_list[ii] = as<mat>(smoothed["alpha"]).row(0);
            P0_list[ii] = as<mat>(smoothed["V"]).row(0);
            // alpha_hat_list[[i]] <- smoothed$alpha
            // alpha_EXP[[EM_it]] <- alpha_new
            S11.zeros();
            S10.zeros();
            S00.zeros();
            series_H.zeros();
            for (int t = 0; t < n; t++) {
                S11 += trans(as<mat>(smoothed["alpha"]).row(t+1) % as<mat>(smoothed["alpha"]).row(t+1) + as<mat>(smoothed["V"]).row(t+1));
                S10 += trans(as<mat>(smoothed["alpha"]).row(t+1) % as<mat>(smoothed["alpha"]).row(t) + as<mat>(smoothed["P_lag"]).row(t));
                S00 += trans(as<mat>(smoothed["alpha"]).row(t) % as<mat>(smoothed["alpha"]).row(t) + as<mat>(smoothed["V"]).row(t));
                series_H += trans((z.row(t) - as<mat>(smoothed["alpha"]).row(t+1)) % (z.row(t) - as<mat>(smoothed["alpha"]).row(t+1)) + as<mat>(smoothed["V"]).row(t+1));
            }
            weighted_S11 += probs.eval()(ii) * S11;
            weighted_S10 += probs.eval()(ii) * S10;
            weighted_S00 += probs.eval()(ii) * S00;
            new_H += probs.eval()(ii) * series_H;
        }
        if(new_log_lik < log_lik)
        {
            return(List::create(Named("TT") = old_TT,
                                Named("Q") = old_Q,
                                Named("H") = old_H,
                                Named("a0_list") = old_a0_list,
                                Named("P0_list") = old_P0_list,
                                Named("code") = 2,
                                Named("EM_it") = EM_it-1,
                                Named("na_lik_series") = -1,
                                Named("log_lik") = log_lik,
                                Named("diff") = lik_diff
            ));
        }
        lik_diff = relativeError(new_log_lik, log_lik, EM_toll / 10);
        log_lik = new_log_lik;
        if(lik_diff < EM_toll)
        {
            return(List::create(Named("TT") = TT,
                                Named("Q") = Q,
                                Named("H") = H,
                                Named("a0_list") = a0_list,
                                Named("P0_list") = P0_list,
                                Named("code") = 0,
                                Named("EM_it") = EM_it,
                                Named("na_lik_series") = -1,
                                Named("log_lik") = log_lik,
                                Named("diff") = lik_diff
            ));
        }
        // Maximization phase to get new T, Q, H
        new_TT = weighted_S10 / weighted_S00;
        new_Q = positive_variances + (weighted_S11 - new_TT % weighted_S10) / weighted_len;
        new_H = positive_variances + new_H / weighted_len;
        
        
        old_TT = TT;
        old_Q = Q; 
        old_H = H; 
        TT = new_TT;
        Q = new_Q;
        H = new_H;
    }
    return(List::create(Named("TT") = TT,
                        Named("Q") = Q,
                        Named("H") = H,
                        Named("a0_list") = a0_list,
                        Named("P0_list") = P0_list,
                        Named("code") = 1,
                        Named("EM_it") = EM_it,
                        Named("na_lik_series") = -1,
                        Named("log_lik") = log_lik,
                        Named("diff") = lik_diff
    ));
}