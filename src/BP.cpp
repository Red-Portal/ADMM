#include <tuple>

#include "ADMMBP.h"

#include <ADMM/parameters.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
// using Eigen::ArrayXd;

// using Rcpp::wrap;
// using Rcpp::as;
// using Rcpp::List;
// using Rcpp::IntegerVector;
// using Rcpp::NumericVector;
// using Rcpp::NumericMatrix;
// using Rcpp::Named;

typedef Eigen::Map<const MatrixXd> MapMat;
typedef Eigen::Map<const VectorXd> MapVec;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::SparseMatrix<double> SpMat;

std::tuple<SpMat, int> 
admm_bp(MapMat const& x_, MapVec const& y_, ADMMParam const& opts_)
{
    int maxit = opts_.maxit;
    double eps_abs = opts_.eps_abs;
    double eps_rel = opts_.eps_rel;
    double rho = opts_.rho;

    ADMMBP solver(x_, y_, rho, eps_abs, eps_rel);

    int niter = solver.solve(maxit);
    SpMat beta(x_.cols(), 1);
    beta.col(0) = solver.get_z();
    beta.makeCompressed();

    // return List::create(Named("beta") = beta,
    //                     Named("niter") = niter);
    return {beta, niter};
}
