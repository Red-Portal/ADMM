#define EIGEN_DONT_PARALLELIZE

#include <ADMM/parameters.h>
#include <vector>

#include "ADMMLassoTall.h"
#include "ADMMLassoWide.h"
#include "DataStd.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::ArrayXf;
using Eigen::ArrayXd;
using Eigen::ArrayXXf;

// using Rcpp::wrap;
// using Rcpp::as;
// using Rcpp::List;
// using Rcpp::Named;
// using Rcpp::IntegerVector;

typedef Eigen::SparseVector<float> SpVec;
typedef Eigen::SparseMatrix<float> SpMat;

inline void write_beta_matrix(SpMat &betas,
                              int col, float beta0,
                              SpVec &coef)
{
    betas.insert(0, col) = beta0;

    for(SpVec::InnerIterator iter(coef); iter; ++iter)
    {
        betas.insert(iter.index() + 1, col) = iter.value();
    }
}

// SEXP admm_lasso(SEXP x_, SEXP y_, SEXP lambda_,
//                 SEXP nlambda_, SEXP lmin_ratio_,
//                 SEXP standardize_, SEXP intercept_,
//                 SEXP opts_)

std::tuple<ArrayXd, SpMat, std::vector<int>>
admm_lasso(MatrixXf x_,VectorXf y_,
           ArrayXd lambda_,
           double lmin_ratio_,
           bool const standardize_, bool const intercept_,
           ADMMParam const& opts_)
{
    // In glmnet, we minimize
    //   1/(2n) * ||y - X * beta||^2 + lambda * ||beta||_1
    // which is equivalent to minimizing
    //   1/2 * ||y - X * beta||^2 + n * lambda * ||beta||_1
    int const n = x_.rows();
    int const p = x_.cols();

    int nlambda = lambda_.size();

    const int maxit        = opts_.maxit;
    const double eps_abs   = opts_.eps_abs;
    const double eps_rel   = opts_.eps_rel;
    const double rho       = opts_.rho;

    DataStd<float> datstd(n, p,
                          standardize_, intercept_);
    datstd.standardize(x_, y_);

    ADMMLassoTall *solver_tall;
    ADMMLassoWide *solver_wide;

    if(x_.rows() > y_.cols())
        solver_tall = new ADMMLassoTall(x_, y_, eps_abs, eps_rel);
    else
        solver_wide = new ADMMLassoWide(x_, y_, eps_abs, eps_rel);

    if(nlambda < 1)
    {
        double lmax = 0.0;
        if(n > p)
            lmax = solver_tall->get_lambda_zero()
                / n * datstd.get_scaleY();
        else
            lmax = solver_wide->get_lambda_zero()
                / n * datstd.get_scaleY();

        double lmin = lmin_ratio_ * lmax;
        lambda_.setLinSpaced(nlambda,
                             std::log(lmax), std::log(lmin));
        lambda_ = lambda_.exp();
        nlambda = lambda_.size();
    }

    SpMat beta(p + 1, nlambda);
    beta.reserve(Eigen::VectorXi::Constant(nlambda, std::min(n, p)));

    std::vector<int> niter(nlambda);
    double ilambda = 0.0;

    for(int i = 0; i < nlambda; i++)
    {
        ilambda = lambda_[i] * n / datstd.get_scaleY();
        if(n > p)
        {
            if(i == 0)
                solver_tall->init(ilambda, rho);
            else
                solver_tall->init_warm(ilambda);

            niter[i] = solver_tall->solve(maxit);
            SpVec res = solver_tall->get_z();
            float beta0 = 0.0;
            datstd.recover(beta0, res);
            write_beta_matrix(beta, i, beta0, res);
        }
        else
        {
            if(i == 0)
                solver_wide->init(ilambda, rho);
            else
                solver_wide->init_warm(ilambda);

            niter[i] = solver_wide->solve(maxit);
            SpVec res = solver_wide->get_x();
            float beta0 = 0.0;
            datstd.recover(beta0, res);
            write_beta_matrix(beta, i, beta0, res);
        }
    }

    if(n > p)
        delete solver_tall;
    else
        delete solver_wide;

    beta.makeCompressed();

    return {lambda_, beta, niter};
}
