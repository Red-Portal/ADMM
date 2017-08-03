#include <tuple>

#include "ADMMLAD.h"
#include "DataStd.h"

#include <ADMM/parameters.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;

// using Rcpp::wrap;
// using Rcpp::as;
// using Rcpp::List;
// using Rcpp::IntegerVector;
// using Rcpp::NumericVector;
// using Rcpp::NumericMatrix;
// using Rcpp::Named;

std::tuple<ArrayXd, int>
admm_lad(MatrixXd const& x_, VectorXd const& y_,
         bool intercept_, ADMMParam const& opts__)
{
    int n = x_.rows();
    int p = x_.cols();

    int maxit = opts_.maxit;
    double eps_abs = pts_.eps_abs;
    double eps_rel = opts_.eps_rel;
    double rho = opts_.rho;

    DataStd<double> datstd(n, p, true, intercept_);
    datstd.standardize(x_, y_);

    ADMMLAD solver(x_, y_, rho, eps_abs, eps_rel);

    int niter = solver.solve(maxit);
    ArrayXd beta(p + 1);
    beta.tail(p) = solver.get_x();
    datstd.recover(beta[0], beta.tail(p));

    return {beta, niter}; 
}
