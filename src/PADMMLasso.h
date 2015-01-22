#ifndef PADMMLASSO_H
#define PADMMLASSO_H

#include "PADMMBase.h"

class PADMMLasso_Worker: public PADMMBase_Worker
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::Map<MatrixXd> MapMat;
    typedef Eigen::Map<VectorXd> MapVec;
    typedef Eigen::HouseholderQR<MatrixXd> QRdecomp;

    const VectorXd XY;            // X'y
    MatrixXd XQR;                 // QR decomposition of X
    
    // res = x in next iteration
    virtual void next_x(VectorXd &res)
    {
        VectorXd b = XY + rho * (*aux_z) - dual_y;
        VectorXd tmp = XQR.triangularView<Eigen::Upper>() * main_x;
        VectorXd r = b - XQR.triangularView<Eigen::Upper>().transpose() * tmp - rho * main_x;
        double rsq = r.squaredNorm();
        tmp.noalias() = XQR.triangularView<Eigen::Upper>() * r;
        double alpha = rsq / (rho * rsq + tmp.squaredNorm());
        res = main_x + alpha * r;
    }
public:
    PADMMLasso_Worker(const MapMat &datX_, const MapVec &datY_,
                      VectorXd &aux_z_) :
        PADMMBase_Worker(datX_.cols(), aux_z_),
        XY(datX_.transpose() * datY_)
    {
        QRdecomp decomp(datX_);
        XQR = decomp.matrixQR().topRows(std::min(datX_.cols(), datX_.rows()));
    }
    
    virtual ~PADMMLasso_Worker() {}
    
    // init() is a cold start for the first lambda
    virtual void init(double rho_)
    {
        main_x.setZero();
        dual_y.setZero();
        rho = rho_;

        rho_changed_action();
    }
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    virtual void init_warm() {}
};

class PADMMLasso_Master: public PADMMBase_Master
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::Map<MatrixXd> MapMat;
    typedef Eigen::Map<VectorXd> MapVec;
    typedef Rcpp::List RList;
    typedef Rcpp::NumericMatrix RMat;
    typedef Rcpp::NumericVector RVec;
    
    double lambda;  // L1 penalty

    virtual void next_z(VectorXd &res)
    {
        res.setZero();
        for(int i = 0; i < n_comp; i++)
        {
            worker[i]->add_y_to(res);
        }
        res /= rho;
        for(int i = 0; i < n_comp; i++)
        {
            worker[i]->add_x_to(res);
        }
        res /= n_comp;
        soft_threshold(res, lambda / rho / n_comp);
    }
    virtual void rho_changed_action() {}
    
public:
    PADMMLasso_Master(RList &datX, RList &datY, int dim_par_,
                      double eps_abs_ = 1e-6,
                      double eps_rel_ = 1e-6) :
        PADMMBase_Master(dim_par_, datX.length(), eps_abs_, eps_rel_)
    {
        for(int i = 0; i < n_comp; i++)
        {
            RMat X = datX[i];
            RVec Y = datY[i];
            MapMat mX(Rcpp::as<MapMat>(X));
            MapVec mY(Rcpp::as<MapVec>(Y));
            worker[i] = new PADMMLasso_Worker(mX, mY, aux_z);
        }
    }
        
    virtual ~PADMMLasso_Master()
    {
        PADMMLasso_Worker *w;
        for(int i = 0; i < n_comp; i++)
        {
            w = (PADMMLasso_Worker *) worker[i];
            delete w;
        }
    }

    virtual double lambda_max() { return 1.0; }

    // init() is a cold start for the first lambda
    virtual void init(double lambda_, double rho_)
    {
        aux_z.setZero();
        lambda = lambda_;
        PADMMLasso_Worker *w;
        for(int i = 0; i < n_comp; i++)
        {
            w = (PADMMLasso_Worker *) worker[i];
            w->init(rho_);
        }
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;

        rho_changed_action();
    }
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    virtual void init_warm(double lambda_)
    {
        lambda = lambda_;
        update_z();
        update_y();
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
    }

    static void soft_threshold(VectorXd &vec, const double &penalty)
    {
        double *ptr = vec.data();
        for(int i = 0; i < vec.size(); i++)
        {
            if(ptr[i] > penalty)
                ptr[i] -= penalty;
            else if(ptr[i] < -penalty)
                ptr[i] += penalty;
            else
                ptr[i] = 0;
        }
    }
};

#endif // PADMMLASSO_H