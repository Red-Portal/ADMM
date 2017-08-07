#ifndef _PARAMETERS_HPP_
#define _PARAMETERS_HPP_

struct ADMMParam
{
    inline ADMMParam(int maxit_,
                     double eps_abs_,
                     double eps_rel_,
                     double rho_)
        : maxit(maxit_),
          eps_abs(eps_abs_),
          eps_rel(eps_rel_), 
          rho(rho_)
    {}

    int maxit;
    double eps_abs;
    double eps_rel;
    double rho;
};

#endif
