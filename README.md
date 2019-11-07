# ACQUIRE

## A MATLAB package for the solution of optimization problems modeling the restoration of images corrupted by Poisson noise.

## Authors
Daniela di Serafino, University of Campania "Luigi Vanvitelli", Caserta, Italy, daniela.diserafino[at]unicampania.it
Germana Landi, University of Bologna, Bologna, Italy, germana.landi[at]unibo.it    
Marco Viola, University of Campania "Luigi Vanvitelli", Caserta, Italy, marco.viola[at]unicampania.it

## Last Update
Version 1.0 - November 1, 2019

## Description
ACQUIRE is a MATLAB implementation of the "Algorithm based on Consecutive
QUadratic and Iteratively REweighted norm approximations" for the solution
of constrained optimization problems of the form

              min  F(x) := KL(A*x + b, y) + lambda*TV(x)
              s.t. x in Omega,

modeling, e.g., the restoration of images corrupted by Poisson noise.

The objective function is the sum of a data-fidelity term consisting of
the generalized Kullback-Leibler (KL) divergence of the blurred image
(A*x + b) from the observed image (y) (see eq. (2) in [1]) and a
regularization term consisting of the discrete isotropic Total
Variation (TV) (see eq. (3) in [1]) with weight lambda. Here A is a
linear blurring operator and b is the background noise.

The feasible set Omega represents either nonnegativity constraints, i.e.,

      Omega = {x : x_i>=0 for all i}

or nonnegativity contraints plus a linear constraint imposing total flux
conservation, i.e.,

      Omega = {x : (x_i>=0, for all i) & (sum(x) = sum_i (y_i - b)}.

ACQUIRE is a line-search method that considers a smoothed version of TV,
based on a Huber-like function, and computes the search directions by
minimizing quadratic approximations of the problem, built by exploiting
some second-order information. The KL divergence is approximated by a
classical second-order Taylor expansion plus a strong convexity term.
An Iteratively Reweighted Norm (IRN) approach is used to approximate the
smoothed TV. See Section 3 in [1] for details.

The minimization of the quadratic approximations is performed by using a
Scaled Gradient Projection (SGP) method [S. Bonettini, R. Zanella and
L. Zanni, Inverse Problems 25 (2009), 015002].

### References
[1] D. di Serafino, G. Landi and M. Viola,
*ACQUIRE: an inexact iteratively reweighted norm approach for TV-based Poisson image restoration*,
Applied Mathematics and Computation, volume 364, 2020, article 124678, DOI: 10.1016/j.amc.2019.124678.
Preprint available from [ArXiv](https://arxiv.org/abs/1807.10832) and [Optimization Online](http://www.optimization-online.org/DB_HTML/2018/07/6745.html).

## Software requirements
ACQUIRE runs under MATLAB. It has been tested under MATLAB 2018b.

## Contents of the package
Here's the list of ACQUIRE files in alphabetical order:
- `acquire.m`     : main function;
- `DFproj_pg.m`   : Dai-Fletcher algorithm for the projection onto
                    simplex-like constraints used for the computation of
                    projected gradients;
- `DFproj_sgp.m`  : Dai-Fletcher algorithm for the projection onto
                    simplex-like constraints used in the SGP algorithm to
                    project points onto the feasible set;
- `ForwardD.m`    : function computing forward first-order finite differences
                    with periodic boundary conditions;
- `ForwardDXT.m`  : function evaluating the adjoint operator of the forward
                    first-order column-difference operator;
- `ForwardDYT.m`  : function evaluating the adjoint operator of the forward
                    first-order row-difference operator;
- `KLfunction.m`  : function evaluationg the Kullback-Leibler (KL)
                    divergence (see eq (2) in [1]);
- `KLhessprod.m`  : function computing the product between the Hessian
                    of the quadratic approximation of the KL function
                    and a vector;
- `KLTVscaling.m` : function computing the scaling matrix for the SGP
                    algorithm by means of the splitting technique
                    described, e.g., in [Zanella et al., Inverse
                    Problems 25 (2009), 045010];
- `line_search.m` : function performing the backtracking Armijo line search;
- `projgrad.m`    : function computing the projected gradient at a given
                    point by using `DFproj_pg`;
- `sgp_subp.m`    : SGP algorithm for the minimization of quadratic subproblems;
- `TVapprox.m`    : function computing the IRN approximation of the
                    smoothed TV;
- `TVfunction.m`  : function evaluating the smoothed TV approximation.

See the documentation inside each file for further details.

## Example of use
- `example.m`                 : example of use of ACQUIRE;
- `TEST_phantom_gauss.mat`    : "phantom" image with Gaussian blur and SNR 40;
- `TEST_cameraman_motion.mat` : "cameraman" image with motion blur and SNR 40.

## License
[![GNU GPL v3.0](http://www.gnu.org/graphics/gplv3-127x51.png)](http://www.gnu.org/licenses/gpl.html)
