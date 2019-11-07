% -------------------------------------------------------------------------
% Copyright (C) 2019 by D. di Serafino, G. Landi, M. Viola.
%
%                           COPYRIGHT NOTIFICATION
%
% This file is part of ACQUIRE.
%
% ACQUIRE is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% ACQUIRE is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with ACQUIRE. If not, see <http://www.gnu.org/licenses/>.
% -------------------------------------------------------------------------

function [xk,exitflag,iter,errvect,fvect,times] = ...
    acquire(A,AisPSF,y,b,lambda,lincon,x0,xtrue,options)

%==========================================================================
%
% Authors:
%   Daniela di Serafino (daniela.diserafino [at] unicampania.it)
%   Germana Landi       (germana.landi [at] unibo.it )
%   Marco Viola         (marco.viola [at] unicampania.it)
%
% Version: 1.0
% Last Update: 1 November 2019
%
%==========================================================================
%
% This function implements the "Algorithm based on Consecutive QUadratic
% and Iteratively REweighted norm approximations" (ACQUIRE), for the
% solution of constrained optimization problems of the form
%
%               min  F(x) := KL(A*x + b, y) + lambda*TV(x)
%               s.t. x in Omega,
%
% modeling, e.g., the restoration of images corrupted by Poisson noise.
% The objective function is the sum of a data-fidelity term consisting of
% the generalized Kullback-Leibler (KL) divergence of the blurred image
% (A*x + b) from the observed image (y) (see eq (2) in [1]) and a
% regularization term consisting of the discrete isotropic Total
% Variation (TV) (see eq (3) in [1]) with weight lambda.
% Here A is a linear blurring operator and b is the background noise.
% The feasible set Omega represents either nonnegativity constraints,
% i.e.,
%       Omega = {x : x_i>=0 for all i}
% or nonnegativity contraints plus a linear constraint imposing total
% flux conservation, i.e.,
%       Omega = {x : (x_i>=0, for all i) & (sum(x) = sum_i (y_i - b)}.
%
% ACQUIRE is a line-search method that considers a smoothed version of TV,
% based on a Huber-like function, and computes the search directions by
% minimizing quadratic approximations of the problem, built by exploiting
% some second-order information. The KL divergence is approximated by
% a classical second-order Taylor expansion plus a strong convexity term.
% An Iteratively Reweighted Norm (IRN) approach is used to approximate the
% smoothed TV. See Section 3 in [1] for details.
% The minimization of the quadratic approximation is performed by using a
% Scaled Gradient Projection (SGP) method [S. Bonettini, R. Zanella and
% L. Zanni, Inverse Problems 25 (2009), 015002].
%
%==========================================================================
%
% REFERENCES:
% [1] D. di Serafino, G. Landi and M. Viola,
%     "ACQUIRE: an inexact iteratively reweighted norm approach for TV-based
%      Poisson image restoration", Applied Mathematics and Computation,
%      volume 364, 2020, article 124678, DOI: 10.1016/j.amc.2019.124678.
%
% Preprint available from ArXiv
%     https://arxiv.org/abs/1807.10832
% and Optimization Online
%     http://www.optimization-online.org/DB_HTML/2018/07/6745.html
%
%==========================================================================
%
% INPUT ARGUMENTS
%
% A       = double matrix, representing the blurring operator (applied as
%           A*x(:)); periodic boundary conditions are assumed; A must have
%           nonnegative entries and its columns must sum-up to 1;
%           for efficiency, it is recommended to provide A as a point
%           spread function (PSF), so that the product A*x(:) will be
%           computed as a pointwise product in the Fourier space;
% AisPSF  = logical, true if A is a PSF;
% y       = double matrix, containing the observed image, affected by blur
%           and Poisson noise;
% b       = double, background noise; note that b is a scalar, i.e., we
%           assume the background noise to be the same for each pixel of y;
% lambda  = double, weight of the TV regularization term;
% lincon  = logical, false if the problem has nonnegativity constraints
%           only, true if the problem has nonnegativity constraints plus a
%           single linear constraint (flux conservation);
% x0      = [optional] double matrix, starting guess,
%           if empty then x0 is set according to the value of
%           options.StartPt (see next);
% xtrue   = [optional] double matrix, true image;
%           if available then the relative error w.r.t. the true image is
%           stored at each iteration, otherwise the relative distance
%           between two consecutive iterates is stored;
% options = [optional] struct array with the following (possible) entries,
%           to be specified as pairs ('propertyname', propertyvalue);
%           the string 'propertyname' can be
%           ScalePb     = logical, if true then the problem is scaled by
%                         the maximum value of y [default true];
%           StartPt     = integer, values 0 or 1 [default 0],
%                          0: x0 is set equal to y,
%                          1: x0 is a constant image with the same total
%                             flux as y;
%           StopCrit    = integer, identifies the stopping criterion used
%                         by ACQUIRE [default 0],
%                          0: maximum number of iterations or maximum time,
%                          1: relative distance between successive iterates,
%                             i.e., the algorithm stops if
%                               ||x_{k+1} - x_k|| < Tolerance*||x_k||,
%                             where || || is the Frobenius norm,
%                          2: relative variation of objective function value,
%                             i.e., the algorithm stops if
%                               |F(x_{k+1}) - F(x_k)| < Tolerance*|F(x_k)|;
%           Tolerance   = double, tolerance for stopping criterion 1 or 2 [default 1e-4];
%           MaxIter     = integer, maximum number of iterations [default 1000];
%           MaxTime     = double, maximum execution time (in seconds) [default 25];
%           LineSearch  = integer, values 0, 1 or >1, specifying the line
%                         search to be performed at each iteration (see
%                         lines 7-9 of Algorithm 1 in [1]) [default 1]:
%                          0: no line search is used,
%                          1: backtracking Armijo line search,
%                         >1: GLL backtracking nonmonotone line search
%                             [Grippo-Lampariello-Lucidi, 1986] with memory
%                             equal to the value of LineSearch.
%
% OUTPUT ARGUMENTS
%
% xk       = double matrix, computed solution;
% exitflag = integer, values 0, 1, 2
%             0: selected stopping criterion satisfied,
%             1: maximum number of iterations reached without satisying the
%                selected stopping criterion;
%             2: maximum execution time reached without satisying the
%                selected stopping criterion;
% iter     = integer, number of iterations performed;
% errvect  = double vector, history of the error (Frobenius) norm
%             - if xtrue is empty then
%                   errvect(k) = ||x_k - x_{k-1}|| / ||x_{k-1}||,
%             - otherwise
%                   errvect(k) = ||x_{k-1} - xtrue|| / ||xtrue||
%               (note that errvect(1) is associated with the starting
%               guess x0 and errvect(k) is associated with the
%               approximation of the solution at iteration k-1);
% fvect    = double vector, history of the objective function values
%            (note that fvect(k) = F(x_{k-1}), where x_0 is the starting
%            guess);
% times    = double vector, times(k) is the elapsed time (in seconds)
%            from the beginning of the computation to the end of the
%            {k-1}-th iteration.
%
%==========================================================================

if nargin < 6
    error('Error: the first 6 input arguments must be provided!');
end
if isempty(AisPSF) || isempty(AisPSF)
    error('Error: incorrect information on the blurring operator!');
end
if min(size(y))<=1
    error('Error: images must be provided as 2-dimensional arrays.');
end
if nargin < 7
    x0 = [];
end
if nargin < 8
    xtrue = [];
end
if nargin < 9
    options = struct([]);
end

%% Initializing external parameters (can be passed to ACQUIRE with the "options" structure)
ScalePb     = true;    % choose whether to scale the problem
StartPt     = 0;       % choose which starting point to use
                       %  0: use y; 1: use constant image
StopCrit    = 0;       % choose the stopping criterion for ACQUIRE:
                       %  0: NONE (only maxit and maxtime),
                       %  1: relative distance between consecutive iterates,
                       %  2: relative variation of objective function
Tolerance   = 1e-4;
MaxIter     = 1000;    % max num iterations of ACQUIRE
MaxTime     = 25;      % max execution time for ACQUIRE
LineSearch  = 1;       % Armijo backtracking line-search is used
if isempty(xtrue)
    errflag = 0;
else
    errflag = 1;
end

%% Initializing parameters for the SGP subproblm solver
SmoothParTV = 1e-2;    % coefficient for the smoothed TV functional
StrConv     = 1e-5;    % coefficient for the strong convexity of the approximant for the Kullback-Leibler
TolRed      = 0.1;     % algorithm tolerance variation (parameter theta in (14) [1])
SubpScaling = 2;       % scaling used by the SGP method in the subproblem solution
                       % 0: NONE, 1: X=x, 2: X = KLTV-specific
MaxPG       = 10;      % maximum number of SGP iterations for each call
MinPG       = 1;       % minimum number of SGP iterations for each call
MemorySGP   = 1;       % specifies how to deal with the PABBmin memory
                       %  0: at each call SGP starts from scratch PABBmin
                       % >0: starting from iteration "MemorySGP", SGP uses informations from the previous call
if lincon
    sgpomega = 'nonneg_eq';
else
    sgpomega = 'nonneg';
end

%% Grabbing personalized settings from options
optionnames = fieldnames(options);
for iter=1:numel(optionnames)
    switch upper(optionnames{iter})
        case 'SCALEPB'
            ScalePb = options.(optionnames{iter});
        case 'STARTPT'
            StartPt = options.(optionnames{iter});
        case 'STOPCRIT'
            StopCrit = options.(optionnames{iter});
        case 'TOLERANCE'
            Tolerance = options.(optionnames{iter});
        case 'MAXITER'
            MaxIter = options.(optionnames{iter});
        case 'MAXTIME'
            MaxTime = options.(optionnames{iter});
        case 'SMOOTHPARTV'
            SmoothParTV = options.(optionnames{iter});
        case 'LINESEARCH'
            LineSearch = options.(optionnames{iter});
        otherwise
            error(['Unrecognized option: ''' optionnames{iter} '''']);
    end
end

%% Starting the clock
tstart = tic;

%% Problem preprocessing
% adding background noise to observed image if its minimum value is smaller than b
if min(min(y)) < b
    y = y+b;
end
if min(y(:))<0
    fprintf('Error: negative data in y!\n');
    return
end

% scaling the problem
if ScalePb
    scalefactor = max(max(y));
    if errflag
        xtrue = xtrue/scalefactor;
    end
    y = y/scalefactor;
    b = b/scalefactor;
end

norm_x_true = norm(xtrue,'fro');

% shifting the PSF
if (AisPSF)
    A = fftshift(A);
end

% precomputing Fast Fourier Transforms for applying A and A'
if AisPSF
    A = fft2(A);
    AT = conj(A);
else
    AT = A';
end

sizexk = size(y);

% computing total flux
flux = sum(sum(y));
N = numel(y);
flux = flux - N*b;

if AisPSF
    tmp = (flux/(flux+N*b)).*real(ifft2(AT.*fft2(y)));
else
    tmp = (flux/(flux+N*b)).*reshape(AT*y(:),sizexk);
end
X_low_bound = min(tmp(tmp>0));         	% lower bound for the scaling matrix
X_upp_bound = max(max(tmp));            % upper bound for the scaling matrix
if X_upp_bound/X_low_bound < 50
    X_low_bound = X_low_bound/10;
    X_upp_bound = X_upp_bound*10;
end

%% Setting starting guess if empty
if isempty(x0)
    switch StartPt
        case 0
            x0 = y;
        case 1
            x0 = flux/numel(y)*ones(size(y));
        otherwise
            error('Option StartPt can only have value 0 or 1!');
    end
end

if lincon
    a0 = flux;
    x0 = DFproj_sgp(a0,x0,ones(size(x0)));
else
    a0 = [];
end

xk = x0;

%% ACQUIRE starts

iter = 0;
errvect = [];
fvect = [];
times = [];
OptMeasure = 1+Tolerance;
kltv_old = 0;
exitflag = -1;

if errflag
    errvect = [errvect, norm(xtrue-xk,'fro')/norm_x_true];
end
times=[times, toc(tstart)];

InfoSGP = [];

if SubpScaling == 2
    if AisPSF
        auxvector = fft2(ones(sizexk));
        auxvector = real(ifft2(AT.*auxvector));
    else
        auxvector = ones(numel(xk),1);
        auxvector = reshape(AT*auxvector,sizexk);
    end
end

while OptMeasure > Tolerance && iter < MaxIter && times(end)<=MaxTime
    iter = iter+1;
    
    % computing Kullback-Leibler divergence at x_k following the
    % formulation by Bertero et al.
    if AisPSF
        Axk = real(ifft2(A.*fft2(xk)));
    else
        Axk = reshape(A*xk(:),sizexk);
    end
    temp = Axk + b;
    
    KL = sum(sum( y.*log(y./temp)+temp-y ));
    W = y./(temp.^2);
    rhstemp = -(temp-y)./temp;
    if AisPSF
        rhstemp = real(ifft2(AT.*fft2(rhstemp)));
    else
        rhstemp = reshape(AT*rhstemp(:),sizexk);
        W = W(:);
    end
    
    % creating the handle to the matrix-vector product with the KL Hessian at xk
    prod_handle = @(v)KLhessprod(W,A,AT,AisPSF,v,sizexk,StrConv);
    
    rhstemp = rhstemp + prod_handle(xk);
    
    if StrConv>0
        rhs = StrConv*xk + rhstemp;
    else
        rhs = rhstemp;
    end
    
    % evaluating the smooth TV at xk
    [TV, Omega] = TVfunction(xk,SmoothParTV);
    % Omega consists of the weights in the IRN approximation
    % of the Huber-like TV (see Sec. 3 in [1])
    qk = @(v)TVapprox(Omega,TV,v);
    
    if iter==1
        kltv_old = KL+lambda*TV;
    end
    fvect = [fvect, kltv_old];
    
    if iter==1 % initializing the tolerance for the subproblem projected gradient
        grad = prod_handle(xk)-rhs + lambda*qk(xk);
        ivar = zeros(size(xk));
        ivar(xk<=0) = -1;
        PGrad = projgrad(grad,ivar,lincon);
        PGnorm0 = norm(PGrad,'fro');
        tol_subp = TolRed * PGnorm0;
    else       % updating tolerance
        tol_subp = TolRed * tol_subp;
    end
    
    % scaling selection for the SGP algorithm
    switch SubpScaling
        case 0
            Scaling = [];
        case 1
            Scaling = @(v) v;
        case 2
            Scaling = @(v) KLTVscaling(auxvector,v,Omega,lambda);
    end
    
    if ~MemorySGP || (iter<MemorySGP)
        InfoSGP = []; % will be used to store information on the BB steps for the previous call
    end
    
    %% Calling SGP for subproblem solution
    [ xk_new, ~,InfoSGP] = sgp_subp(prod_handle,rhs,...
        'LAMBDA', lambda,...
        'REG',qk,...
        'XSCALING', SubpScaling,...
        'X_LB',X_low_bound,...
        'X_UB',X_upp_bound,...
        'VREG',Scaling,...
        'MAXIT', MaxPG+1, ...
        'MINIT', MinPG, ...
        'MAXTIME',MaxTime,...
        'TSTART',tstart,...
        'VERBOSE', 0, ...
        'STOPCRITERION', 4, ...
        'TOL', tol_subp,...
        'INITIALIZATION', xk, ...
        'InfoSGP',InfoSGP,...
        'OMEGA',sgpomega,...
        'FLUX',a0);
    
    ivar = zeros(size(xk_new));
    ivar(xk_new <= 0) = -1;
    
    gradk = prod_handle(xk)-rhs; % KL gradient at previous step
    
    if LineSearch
        pk = xk_new-xk;
        gradk = gradk + lambda*qk(xk);
        if LineSearch == 1
            fref = kltv_old;
        elseif length(fvect)>LineSearch
            tempf = fvect(end-LineSearch+1:end);
            fref = max(tempf);
        else
            tempf = fvect;
            fref = max(tempf);
        end
        [xk_new,~,kltv_new] = line_search(xk,pk,gradk,@(v)(KLfunction(A,AisPSF,v,sizexk,b,y)+lambda*TVfunction(v,SmoothParTV)),fref);
    else
        kltv_new = KLfunction(A,AisPSF,xk,sizexk,b,y)+lambda*TVfunction(xk,SmoothParTV);
    end
    
    %% Evaluate stopping criterion
    switch StopCrit
        case 0 % no stop
            OptMeasure = Tolerance +1;
        case 1 % relative distance between iterates
            OptMeasure = norm(xk_new-xk,'fro')/norm(xk,'fro');
        case 2 % relative variation of objective function value
            OptMeasure = abs(kltv_old - kltv_new)/abs(kltv_new);
        otherwise
            error('Error: stopping criterion is not existent!');
    end
    
    if errflag
        errvect = [errvect, norm(xtrue-xk_new,'fro')/norm_x_true];
    else
        errvect = [errvect, norm(xk_new-xk,'fro')/norm(xk,'fro')];
    end
    times=[times, toc(tstart)];
    
    xk = xk_new;
    kltv_old = kltv_new;
    
end
fvect = [fvect, kltv_new];

if StopCrit == 0
    exitflag = 0;
else
    if OptMeasure <= Tolerance
        exitflag = 0;
    elseif iter >= MaxIter
        exitflag = 1;
    elseif times(end)>MaxTime
        exitflag = 2;
    end
end

end