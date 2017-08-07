#ifndef _ADMM_HPP_
#define _ADMM_HPP_

#include <Eigen/SparseCore>
#include <Eigen/Cholesky>

#include <vector>
#include <tuple>

#include "parameters.h"

using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::VectorXd;
using Eigen::ArrayXd;

typedef Eigen::Map<const MatrixXd> MapMat;
typedef Eigen::Map<const VectorXd> MapVec;

template<typename T>
using SpVec = Eigen::SparseVector<T>;
template<typename T>
using SpMat = Eigen::SparseMatrix<T>;

namespace ADMM
{
    std::tuple<SpMat<double>, int> 
    admm_bp(MapMat const& x_, MapVec const& y_,
            ADMMParam const& opts_);
    
    std::tuple<ArrayXd, SpMat<float>, std::vector<int>>
    admm_enet(MatrixXf x_, VectorXf y_,
              ArrayXd lambda_,
              double lmin_ratio_,
              bool const standardize_, bool const intercept_,
              double const alpha_,
              ADMMParam const& opts_);

    
    std::tuple<ArrayXd, int>
    admm_lad(MatrixXd x_, VectorXd y_,
             bool intercept_, ADMMParam const& opts_);

    std::tuple<ArrayXd, SpMat<float>, std::vector<int>>
    admm_lasso(MatrixXf x_,VectorXf y_,
               ArrayXd lambda_,
               double lmin_ratio_,
               bool const standardize_, bool const intercept_,
               ADMMParam const& opts_);


    std::tuple<ArrayXd, SpMat<float>, std::vector<int>>
    admm_parlasso(MatrixXf x_, VectorXf y_,
                  ArrayXd lambda_,
                  double lmin_ratio_,
                  bool const standardize_, bool const intercept_,
                  int nthread_, ADMMParam opts_);

}
#endif
