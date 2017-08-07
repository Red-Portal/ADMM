#ifndef ADMMLASSOTALL_H
#define ADMMLASSOTALL_H

#include <Eigen/SparseCore>
#include <Eigen/Cholesky>

#include "FADMMBase.h"
#include "Linalg/BlasWrapper.h"
#include "Spectra/SymEigsSolver.h"
#include "ADMMMatOp.h"

// minimize  1/2 * ||y - X * beta||^2 + lambda * ||beta||_1
//
// In ADMM form,
//   minimize f(x) + g(z)
//   s.t. x - z = 0
//
// x => beta
// z => -X * beta
// A => X
// b => y
// f(x) => 1/2 * ||Ax - b||^2
// g(z) => lambda * ||z||_1
class ADMMLassoTall:
    public FADMMBase<Eigen::VectorXf,
                     Eigen::SparseVector<float>,
                     Eigen::VectorXf>
{
protected:
    typedef float Scalar;  // Using float is much faster than double
    typedef Eigen::Matrix<Scalar,
                          Eigen::Dynamic,
                          Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map<const Vector> MapVec;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::SparseVector<Scalar> SparseVector;
    typedef Eigen::LLT<Matrix> LLT;

    MapMat datX;                  // data matrix
    MapVec datY;                  // response vector
    Vector XY;                    // X'Y
    LLT solver;                   // matrix factorization

    Scalar lambda;                // L1 penalty
    Scalar lambda0;               // minimum lambda to make coefficients all zero



    // x -> Ax
    void A_mult (Vector &res, Vector &x)  { res.swap(x); }
    // y -> A'y
    void At_mult(Vector &res, Vector &y)  { res.swap(y); }
    // z -> Bz
    void B_mult (Vector &res, SparseVector &z) { res = -z; }
    // ||c||_2
    double c_norm() { return 0.0; }



    static void soft_threshold(SparseVector &res, const Vector &vec, const double &penalty)
    {
        int v_size = vec.size();
        res.setZero();
        res.reserve(v_size);

        const Scalar *ptr = vec.data();
        for(int i = 0; i < v_size; i++)
        {
            if(ptr[i] > penalty)
                res.insertBack(i) = ptr[i] - penalty;
            else if(ptr[i] < -penalty)
                res.insertBack(i) = ptr[i] + penalty;
        }
    }
    void next_x(Vector &res)
    {
        Vector rhs = XY - adj_y;
        // rhs += rho * adj_z;

        // manual optimization
        for(SparseVector::InnerIterator iter(adj_z); iter; ++iter)
            rhs[iter.index()] += rho * iter.value();

        res.noalias() = solver.solve(rhs);
    }
    virtual void next_z(SparseVector &res)
    {
        Vector vec = main_x + adj_y / rho;
        soft_threshold(res, vec, lambda / rho);
    }
    void next_residual(Vector &res)
    {
        // res = main_x;
        // res -= aux_z;

        // manual optimization
        std::copy(main_x.data(), main_x.data() + dim_main, res.data());
        for(SparseVector::InnerIterator iter(aux_z); iter; ++iter)
            res[iter.index()] -= iter.value();
    }
    void rho_changed_action() {}
    void update_rho() {}



    // Calculate ||v1 - v2||^2 when v1 and v2 are sparse
    static double diff_squared_norm(const SparseVector &v1, const SparseVector &v2)
    {
        const int n1 = v1.nonZeros(), n2 = v2.nonZeros();
        const Scalar *v1_val = v1.valuePtr(), *v2_val = v2.valuePtr();
        const int *v1_ind = v1.innerIndexPtr(), *v2_ind = v2.innerIndexPtr();

        Scalar r = 0.0;
        int i1 = 0, i2 = 0;
        while(i1 < n1 && i2 < n2)
        {
            if(v1_ind[i1] == v2_ind[i2])
            {
                Scalar val = v1_val[i1] - v2_val[i2];
                r += val * val;
                i1++;
                i2++;
            } else if(v1_ind[i1] < v2_ind[i2]) {
                r += v1_val[i1] * v1_val[i1];
                i1++;
            } else {
                r += v2_val[i2] * v2_val[i2];
                i2++;
            }
        }
        while(i1 < n1)
        {
            r += v1_val[i1] * v1_val[i1];
            i1++;
        }
        while(i2 < n2)
        {
            r += v2_val[i2] * v2_val[i2];
            i2++;
        }

        return r;
    }

    // Faster computation of epsilons and residuals
    double compute_eps_primal()
    {
        double r = std::max(main_x.norm(), aux_z.norm());
        return r * eps_rel + std::sqrt(double(dim_dual)) * eps_abs;
    }
    double compute_eps_dual()
    {
        return dual_y.norm() * eps_rel + std::sqrt(double(dim_main)) * eps_abs;
    }
    double compute_resid_dual()
    {
        return rho * std::sqrt(diff_squared_norm(aux_z, old_z));
    }
    double compute_resid_combined()
    {
        // SparseVector tmp = aux_z - adj_z;
        // return rho * resid_primal * resid_primal + rho * tmp.squaredNorm();

        // manual optmization
        return rho * resid_primal * resid_primal + rho * diff_squared_norm(aux_z, adj_z);
    }

public:
    ADMMLassoTall(ConstGenericMatrix &datX_, ConstGenericVector &datY_,
                  double eps_abs_ = 1e-6,
                  double eps_rel_ = 1e-6) :
        FADMMBase<Eigen::VectorXf, Eigen::SparseVector<float>, Eigen::VectorXf>
                  (datX_.cols(), datX_.cols(), datX_.cols(),
                  eps_abs_, eps_rel_),
        datX(datX_.data(), datX_.rows(), datX_.cols()),
        datY(datY_.data(), datY_.size()),
        XY(datX.transpose() * datY),
        lambda0(XY.cwiseAbs().maxCoeff())
    {}

    double get_lambda_zero() const { return lambda0; }

    // init() is a cold start for the first lambda
    void init(double lambda_, double rho_)
    {
        main_x.setZero();
        aux_z.setZero();
        dual_y.setZero();

        adj_z.setZero();
        adj_y.setZero();

        lambda = lambda_;
        rho = rho_;

        Matrix XX;
        Linalg::cross_prod_lower(XX, datX);

        if(rho <= 0)
        {
            Spectra::DenseSymMatProd<Scalar> op(XX);
            Spectra::SymEigsSolver< Scalar, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<Scalar> > eigs(&op, 1, 3);
            eigs.init();
            eigs.compute(10, 0.1);
            Vector evals = eigs.eigenvalues();
            rho = std::pow(evals[0], 1.0 / 3) * std::pow(lambda, 2.0 / 3);
        }

        XX.diagonal().array() += rho;
        solver.compute(XX.selfadjointView<Eigen::Lower>());

        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;

        adj_a = 1.0;
        adj_c = 9999;

        rho_changed_action();
    }
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    void init_warm(double lambda_)
    {
        lambda = lambda_;

        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;

        // adj_a = 1.0;
        // adj_c = 9999;
    }
};



#endif // ADMMLASSOTALL_H
