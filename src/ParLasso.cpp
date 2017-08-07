#include <ADMM/ADMM.h>

#include "PADMMLasso.h"
#include "DataStd.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// using Eigen::MatrixXf;
// using Eigen::VectorXf;
// using Eigen::ArrayXf;
// using Eigen::ArrayXd;
// using Eigen::ArrayXXf;

// using Rcpp::wrap;
// using Rcpp::as;
// using Rcpp::List;
// using Rcpp::Named;
// using Rcpp::IntegerVector;

// typedef Eigen::SparseVector<float> SpVec;
// typedef Eigen::SparseMatrix<float> SpMat;

template<typename T>
inline void write_beta_matrix(SpMat<T> &betas,
                              int col, float beta0,
                              SpVec<T> &coef)
{
    betas.insert(0, col) = beta0;

    for(typename SpVec<T>::InnerIterator iter(coef); iter; ++iter)
    {
        betas.insert(iter.index() + 1, col) = iter.value();
    }
}


std::tuple<ArrayXd, SpMat<float>, std::vector<int>>
ADMM::
admm_parlasso(MatrixXf x_, VectorXf y_,
              ArrayXd lambda_,
              double lmin_ratio_,
              bool const standardize_, bool const intercept_,
              int nthread_, ADMMParam opts_)
{
    // Rcpp::NumericMatrix xx(x_);
    // Rcpp::NumericVector yy(y_);

    const int n = x_.rows();
    const int p = y_.cols();

    // In glmnet, we minimize
    //   1/(2n) * ||y - X * beta||^2 + lambda * ||beta||_1
    // which is equivalent to minimizing
    //   1/2 * ||y - X * beta||^2 + n * lambda * ||beta||_1
    int nlambda = lambda_.size();

    const int maxit        = opts_.maxit;
    const double eps_abs   = opts_.eps_abs;
    const double eps_rel   = opts_.eps_rel;
    const double rho       = opts_.rho;

    DataStd<float> datstd(n, p, standardize_, intercept_);
    datstd.standardize(x_, y_);

    PADMMLasso_Master solver(x_, y_, nthread_,
                             eps_abs, eps_rel);

    if(nlambda < 1)
    {
        double lmax = solver.get_lambda_zero()
            / n * datstd.get_scaleY();
        double lmin = lmin_ratio_ * lmax;
        lambda_.setLinSpaced(nlambda,
                             std::log(lmax), std::log(lmin));
        lambda_ = lambda_.exp();
        nlambda = lambda_.size();
    }

    SpMat<float> beta(p + 1, nlambda);
    beta.reserve(
        Eigen::VectorXi::Constant(
            nlambda, std::min(n, p)));

    std::vector<int> niter(nlambda);
    double ilambda = 0.0;

    for(int i = 0; i < nlambda; i++)
    {
        ilambda = lambda_[i] * n / datstd.get_scaleY();
        if(i == 0)
            solver.init(ilambda, rho);
        else
            solver.init_warm(ilambda);

        niter[i] = solver.solve(maxit);
        SpVec<float> res = solver.get_z();
        float beta0 = 0.0;
        datstd.recover(beta0, res);
        write_beta_matrix(beta, i, beta0, res);
    }

    beta.makeCompressed();

    return {lambda_, beta, niter};
}
