#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

//' Kalman filtering and smoothing of time series
//'
//' \code{k_filter} and \code{k_smoother} compute the filtered and smoothed estimates of time series modeled according to the following state-space model:
//' \deqn{\boldsymbol{\alpha}_t  = \mathbf{T} \boldsymbol{\alpha}_{t-1} + \boldsymbol{\xi}_t\;,\quad \boldsymbol{\xi}_t \sim \mathcal{N}\left(0,\  \mathbf{Q}\right),}
//' \deqn{\mathbf{z}_t  = \boldsymbol{\alpha}_t + \boldsymbol{\zeta}_t\,,\qquad \boldsymbol{\zeta}_t\sim \mathcal{N}\left(0,\ \mathbf{H}\right).} 
//' IMPORTANT: T, Q and H are assumed to be diagonal.
//' @name ShSt_KFS

 
//' @describeIn ShSt_KFS Computes the filtered estimates for a time series.
//' @param z A matrix representing time series. Each row represents a time
//' observation.
//' @param a0 Initial state (expectation).
//' @param P0 Diagonal of the variance matrix of the initial state. 
//' @param TT The diagonal of the matrix of state transformation.
//' @param Q  The diagonal of the variance matrix of the state error.
//' @param H  The diagonal of the variance matrix of the observation error.
//' @details The predicted estimates and their variances are defined as
//' \deqn{\mathbf{a}_{t}^{t-1} = \mathbb{E}\left(\boldsymbol{\alpha}_{t} \mid \mathbf{Z}_{1:{t-1}}\right),\quad \mathbf{P}_{t}^{t-1} = \mathbb{V}\text{ar}\left(\boldsymbol{\alpha}_{t} \mid \mathbf{Z}_{1:{t-1}}\right),}
//' where \eqn{\mathbf{Z}_{1:{t-1}}} denotes the series observed up to time \eqn{t-1}. 
//' The filtered estimates and their variances are defined as
//' \deqn{\mathbf{a}_{t}^{t} = \mathbb{E}\left(\boldsymbol{\alpha}_{t} \mid \mathbf{Z}_{1:{t}}\right),\quad \mathbf{P}_{t}^{t} = \mathbb{V}\text{ar}\left(\boldsymbol{\alpha}_{t} \mid \mathbf{Z}_{1:{t}}\right).}
//' @return A list containing:
//' \describe{
//' \item{alpha_next}{Matrix of predicted estimates \eqn{\mathbf{a}_t^{t-1}}.}
//' \item{alpha_current}{Matrix of filtered estimates \eqn{\mathbf{a}_t^t}.}
//' \item{P_next}{Estimated variances of the predicted estimates.}
//' \item{P_current}{Estimated variances of the filtered estimates.}
//' \item{log_lik}{Marginal log-likelihood.}
//' \item{K_n}{Final Kalman gain - useful for smoothing.}
//' } 
//' @export
// [[Rcpp::export]]
List k_filter(const arma::mat z,
              const arma::vec a0,
              const arma::vec P0,
              const arma::vec TT,
              const arma::vec Q,
              const arma::vec H) 
{
    int n = z.n_rows, d = z.n_cols + 1;
    mat alpha_current(n + 1, d - 1);
    alpha_current.row(0) = trans(a0); // starting point is given for "current"
    mat alpha_next(n, d - 1);
    
    mat P_current(n+1, d-1);
    P_current.row(0) = trans(P0);
    mat P_next(n, d-1);
    double log_lik = 0;
    
    rowvec K;
    
    for(int t = 0; t < n; t++)
    {
        alpha_next.row(t) = alpha_current.row(t) % trans(TT);
        P_next.row(t) = trans(TT) % P_current.row(t) % trans(TT) + trans(Q);
        log_lik += log(prod(P_next.row(t) + trans(H))) +
            sum((z.row(t) - alpha_next.row(t)) / (P_next.row(t) + trans(H)) % (z.row(t) - alpha_next.row(t)));
        K = P_next.row(t) / (P_next.row(t) + trans(H));
        alpha_current.row(t+1) = alpha_next.row(t) + (z.row(t) - alpha_next.row(t)) % K;
        P_current.row(t+1) = (ones<rowvec>(d-1) - K) % P_next.row(t);
    }
    return(
        List::create(
            Named("alpha_next") = alpha_next,
            Named("alpha_current") = alpha_current,
            Named("P_next") = P_next,
            Named("P_current") = P_current,
            Named("log_lik") = -0.5 * log_lik,
            Named("K_n") = K
        )
    );
}


//' @describeIn ShSt_KFS Computes the smoothed estimates for a time series, given
//' the filtered estimates.
//' @param filtered A list of results from `k_filter`.
//' @param TT Matrix of state transformation.
//' @details The smoothed estimates and their variances are defined as
//' \deqn{\mathbf{a}_{t}^{n} = \mathbb{E}\left(\boldsymbol{\alpha}_{t} \mid \mathbf{Z}\right),\quad \mathbf{P}_{t}^{n} = \mathbb{V}\text{ar}\left(\boldsymbol{\alpha}_{t} \mid \mathbf{Z}\right),}
//' that is, having knowledge of the whole observed series.
//' @return \code{k_smoother} returns a list containing:
//' \describe{
//' \item{alpha}{Matrix of smoothed estimates \eqn{\mathbf{a}_t^{n}}.}
//' \item{V}{Estimated variances of the smoothed estimates.}
//' \item{P_lag}{Lag-One covariance smoother - useful to estimate the system matrices.}
//' } 
//' @references 
//' Shumway, R. H. and Stoffer, D. S. (2000) Time Series Analysis and its Applications. 
//' Volume 3. Springer
//' @export
// [[Rcpp::export]]
List k_smoother(const List filtered,
                const arma::vec TT)
{
    mat alpha_next = filtered["alpha_next"];
    mat alpha_current = filtered["alpha_current"];
    mat P_next = filtered["P_next"];
    mat P_current = filtered["P_current"];
    rowvec K_n = filtered["K_n"];
    
    int n = alpha_next.n_rows, d = alpha_next.n_cols + 1;
    
    mat alpha(n + 1, d - 1);
    alpha.row(n) = alpha_current.row(n);
    mat V(n+1, d-1);
    V.row(n) = P_current.row(n);
    mat P_lag(n, d-1);
    P_lag.row(n-1) = (ones<rowvec>(d-1) - K_n) % trans(TT) % P_current.row(n-1);
    rowvec J, J_old;
    for (int t = n-1; t>=0; t--) {
        J = P_current.row(t) % trans(TT) / P_next.row(t);
        alpha.row(t) = alpha_current.row(t) + (alpha.row(t+1) - alpha_next.row(t)) % J;
        V.row(t) = P_current.row(t) + J % (V.row(t+1) - P_next.row(t)) % J;
        if (t < (n-1))
        {
            P_lag.row(t) = P_current.row(t+1) % J + J_old % (P_lag.row(t+1) - trans(TT) % P_current.row(t+1)) % J;
        }
        J_old = J;
    }
    return(List::create(
            Named("alpha") = alpha,
            Named("V") = V,
            Named("P_lag") = P_lag
    ));
}