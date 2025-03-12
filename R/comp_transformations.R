#' Logratio and inverse transformations
#'
#' Collection of functions to return the clr- or ilr-transformed version of a compositional time series, or viceversa. Transformations between ilr and clr forms are included.
#'
#' @return Each function returns a matrix with the same number of rows as the input, with row-wise transformed elements.
#' @details Vectors are treated as 1-row matrices, and the returned value is as such.
#' @name comp_transformations
NULL

#' @describeIn comp_transformations clr transformation of a compositional series.
#' @param y A T x D matrix, where T is the series length and D is the simplex dimension. Each row should be a composition.
#' @export
comp_to_clr <- function(y)
{
    if(is.null(dim(y)))
    {
        # 1 vector case
        y <- t(matrix(y)) # now it's 1xd
    }
    return(t(apply(y, 1, function(x) log(x / compositions::geometricmean(x)))))
    
}

#' @describeIn comp_transformations inverse clr transformation of a series (returns a matrix of compositions).
#' @param zc A T x D matrix, where T is the series length and D is the simplex dimension. Each row should sum to 0.
#' @export
clr_to_comp <- function(zc)
{
    if(is.null(dim(zc)))
    {
        # 1 vector case
        zc <- t(matrix(zc)) # now it's 1xd
    }
    return(t(apply(zc, 1, function(x) exp(x)/sum(exp(x)))))
}

#' @describeIn comp_transformations ilr transformation of a clr series.
#' @param zc A T x D matrix, where T is the series length and D is 
#' the simplex dimension. Each row should sum to 0.
#' @export
clr_to_ilr <- function(zc)
{
    if (is.null(dim(zc)))
    {
        # 1 vector case
        zc <- t(matrix(zc)) # now it's 1xd
    }
    d <- dim(zc)[2]
    V <- matrix(0, d, d - 1)
    for (i in 1:d - 1) {
        u <- sqrt(i / (i + 1)) * c(rep(1 / i, i), -1)
        V[1:(i + 1) , i] <- u
    }
    return(t(apply(zc, 1, function(x)
        t(V) %*% x)))
}

#' @describeIn comp_transformations clr transformation of a ilr series.
#' @param zi A T x (D-1) matrix, where T is the series length and D is the simplex dimension.
#' @export
ilr_to_clr <- function(zi)
{
    if(is.null(dim(zi)))
    {
        # 1 vector case
        zi <- t(matrix(zi)) # now it's 1xd
    }
    d <- dim(zi)[2] + 1
    V <- matrix(0, d, d-1)
    for (i in 1:d-1) {
        u <- sqrt(i/(i+1)) * c(rep(1/i, i), -1)
        V[1:(i+1) ,i] <- u
    }
    return(t(apply(zi, 1, function(x) V %*% x)))
}

#' @describeIn comp_transformations ilr transformation of a compositional series.
#' @param y A T x D matrix, where T is the series length and D is 
#' the simplex dimension. Each row should be a composition.
#' @export
comp_to_ilr <- function(y)
{
    return(clr_to_ilr(comp_to_clr(y)))
}


#' @describeIn comp_transformations inverse ilr transformation of a series (returns a matrix of compositions).
#' @param zi A T x (D-1) matrix, where T is the series length and D is the simplex dimension.
#' @export
ilr_to_comp <- function(zi)
{
    return(clr_to_comp(ilr_to_clr(zi)))
}
