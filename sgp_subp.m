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

function [x, iter, InfoSGP] = sgp_subp(A, b, varargin)

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
% This code is a modified version of the "sgp_deblurring" code from Zanella
% and Cavicchioli available at: http://www.unife.it/prisma/software
% This modified version allows to use the SGP method for the solution of
% the quadratic subproblems arising in ACQUIRE.
% The problem is assumed to have the form
%                         min  0.5 * x'*A*x - b'*x + lambda*qreg(x)
%                         s.t. sum(x) = flux [optional]
%                              x >= 0.
% Here A and b represent respectively the Hessian and the linear term of
% the quadratic approximation of the KL functional, whereas qreg is the IRN
% approximation of the smoothed TV regularization (see Section 3 in [1]).
%
%==========================================================================
%
% REFERENCES:
% [1] D. di Serafino, G. Landi and M. Viola,
%     "ACQUIRE: an inexact iteratively reweighted norm approach for TV-based
%      Poisson image restoration", Applied Mathematics and Computation,
%      volume 364, 2020, article 124678, DOI:https://doi.org/10.1016/j.amc.2019.124678.
%
% Preprint available from ArXiv
%     https://arxiv.org/abs/1807.10832
% and Optimization Online
%     http://www.optimization-online.org/DB_HTML/2018/07/6745.html
%
%==========================================================================
%
% MANDATORY INPUTS ARGUMENTS
% 
% A         = function handle used to compute the Matrix-vector product
%             between the Hessian of the Kullback-Leibler quadratic
%             approximation and a given vector;
% b         = double matrix, linear term of the Kullback-Leibler quadratic
%             approximation;
%
% OPTIONAL INPUT ARGUMENTS 
% 
% The following options must be provided as keyword/value pairs. The
% keyword can be
%
% OMEGA     = string, constraints type [default 'nonneg']
%              'nonneg'   : non negativity,
%              'nonneg_eq': non negativity and flux conservation, flux
%                           needs to be specified;
% FLUX      = double, total flux (r.h.s of the linear constraint);
% LAMBDA    = double, weight of the regularization term;
% REG       = function handle to qreg, i.e., the quadratic approximation of
%             the regularization function; the function must return the two
%             output arguments [grad_qreg, fval_qreg], corresponding
%             respectively to the gradient and the function value of qreg;
% INFOSGP   = struct, contains information for the computation of ABBmin
%             step lengths from the previous call of SGP;
% XSCALING  = integer, scaling to be used in SGP, values 0 ,1 or 2 [default 1]
%              0: NONE, 1: X = x, 2: X = KLTV-specific scaling;
% VREG      = function handle to a function computing the scaling matrix in
%             the case XSCALING = 2;
% X_LB      = double, lower bound for the scaling matrix [default 1e-5];
% X_UB      = double, upper bound for the scaling matrix [default 1e5];
% INITIALIZATION = integer or double matrix, choice for starting point [default 0]
%                    0: all zero starting point,
%                    1: random starting point,
%                   x0: double matrix, user-provided starting point;
% MINIT     = integer, minimum number of iterations [default 4];
% MAXIT     = integer, maximum number of iterations [default 1000];
% TSTART    = unsigned integer, starting time of the calling function; if
%             left empty the time cost of the current SGP call is computed;
% MAXTIME   = double, maximum allowed computational time (in seconds)
%              either for the calling function of for SGP [default 25];
% STOPCRITERION = integer, stopping rule selection, values from 1 to 4 [default 1]
%                  1: maximum number of iterations,
%                  2: relative distance between successive iterates,
%                  3: relative function value variation,
%                  4: absolute tolerance on projected gradient norm.
% TOL       = double, tolerance used in the stopping criterion [default 1e-4];
% M         = integer, nonmonotone lineasearch memory, if m = 1 the
%             algorithm is monotone [default 1];
% GAMMA     = double, linesearch sufficient-decrease parameter [default 1e-4];
% BETA      = double, linesearch backtracking parameter [default 0.4];
% ALPHA_MIN = double, lower bound for Barzilai-Borwein step lengths [default 1e-5];
% ALPHA_MAX = double, upper bound for Barzilai-Borwein step lengths [default 1e5];
% MALPHA    = integer, memory length for alphaBB2 [default 3];
% TAUALPHA  = double, parameter for Barzilai-Borwein steplength [default 0.5];
% INITALPHA = double, initial value for Barzilai-Borwein steplength [default 1.3];
% VERBOSE   = integer, values 0 or 1, verbosity level: [default 0]
%              0: silent,
%              1: print some information at each iteration.
%
% OUTPUT ARGUMENTS
% 
% x         = double matrix, computed solution;
% iter      = integer, number of SGP iterations;
% InfoSGP   = struct, contains information for the computation of ABBmin
%             step lengths to be usedin the next call of SGP.
%
%==========================================================================
%
%                       INFO ON THE ORIGINAL VERSION
%
% This software is developed within the research project
%
%        PRISMA - Optimization methods and software for inverse problems
%                           http://www.unife.it/prisma
%
% funded by the Italian Ministry for University and Research (MIUR), under
% the PRIN2008 initiative, grant n. 2008T5KA4L, 2010-2012. This software is
% part of the package "IRMA - Image Reconstruction in Microscopy and Astronomy"
% currently under development within the PRISMA project.
%
% Version: 1.0
% Date:    July 2011
%
% Authors:
%   Riccardo Zanella, Gaetano Zanghirati
%    Dept. of Mathematics, University of Ferrara, Italy
%    riccardo.zanella@unife.it, g.zanghirati@unife.it
%   Roberto Cavicchioli, Luca Zanni
%    Dept. of Pure Appl. Math., Univ. of Modena and Reggio Emilia, Italy
%    roberto.cavicchioli@unimore.it, luca.zanni@unimore.it
%
% Software homepage: http://www.unife.it/prisma/software
%
% Copyright (C) 2011 by R. Cavicchioli, R. Zanella, G. Zanghirati, L. Zanni.
% -------------------------------------------------------------------------
% COPYRIGHT NOTIFICATION
%
% Permission to copy and modify this software and its documentation for
% internal research use is granted, provided that this notice is retained
% thereon and on all copies or modifications. The authors and their
% respective Universities makes no representations as to the suitability
% and operability of this software for any purpose. It is provided "as is"
% without express or implied warranty. Use of this software for commercial
% purposes is expressly prohibited without contacting the authors.
%
% This program is free software; you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% Free Software Foundation; either version 3 of the License, or (at your
% option) any later version.
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
% See the GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License along
% with this program; if not, either visite http://www.gnu.org/licenses/
% or write to
% Free Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
% =========================================================================

% start the clock
t0 = tic;

% test for number of required parametres
if (nargin-length(varargin)) ~= 2
    error('Wrong number of required parameters');
end

%%%%%%%%%%%%%%%%%%%%%%%%
% SGP default parameters
%%%%%%%%%%%%%%%%%%%%%%%%
MAXIT = 1000;                       % maximum number of iterations
MINIT = 4;                          % minimum number of iterations (added by Daniela)
MAXTIME = 25;
gamma = 1e-4;                   	% for sufficient decrease
beta = 0.4;                     	% backtracking parameter
M = 1;                          	% memory in obj. function value (if M = 1 monotone)
alpha_min = 1e-5;             		% alpha lower bound
alpha_max = 1e5;					% alpha upper bound
Malpha = 3;                     	% alfaBB1 memory
tau = 0.5;                      	% alternating parameter
initalpha = 1.3;                  	% initial alpha
initflag = 0;                       % 0 -> initial x all zeros
verb = 0;                           % 0 -> silent
stopcrit = 1;                       % 1 -> number of iterations
tol=1e-4;
omega = 'nonneg';                   % non negativity constraints
flux = [];
lambda = 0;
reg = [];
xscaling = 1;
Vreg = [];
InfoSGP = [];
X_low_bound = 1e-5;
X_upp_bound = 1e5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read the optional parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'OMEGA'
                omega = varargin{i+1};
            case 'FLUX'
                flux = varargin{i+1};
            case 'LAMBDA'
                lambda = varargin{i+1};
            case 'REG'
                reg = varargin{i+1};
            case 'INFOSGP'
                InfoSGP = varargin{i+1};
            case 'XSCALING'
                xscaling = varargin{i+1};
            case 'VREG'
                Vreg = varargin{i+1};
            case 'X_LB'
                X_low_bound = varargin{i+1};
            case 'X_UB'
                X_upp_bound = varargin{i+1};
            case 'INITIALIZATION'
                if numel(varargin{i+1}) > 1   % initial x provided by user
                    initflag = 999;
                    x = varargin{i+1};
                else
                    initflag = varargin{i+1};
                end
            case 'MINIT'
                MINIT = varargin{i+1};
            case 'MAXIT'
                MAXIT = varargin{i+1};
            case 'TSTART'
                t0 = varargin{i+1};
            case 'MAXTIME'
                MAXTIME = varargin{i+1};
            case 'STOPCRITERION'
                stopcrit = varargin{i+1};
            case 'TOL'
                tol = varargin{i+1};
            case 'M'
                M = varargin{i+1};
            case 'GAMMA'
                gamma = varargin{i+1};
            case 'BETA'
                beta = varargin{i+1};
            case 'ALPHA_MIN'
                alpha_min = varargin{i+1};
            case 'ALPHA_MAX'
                alpha_max = varargin{i+1};
            case 'MALPHA'
                Malpha = varargin{i+1};
            case 'TAUALPHA'
                tau = varargin{i+1};
            case 'INITALPHA'
                initalpha = varargin{i+1};
            case 'VERBOSE'
                verb = varargin{i+1};
            otherwise
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end

%%%%%%%%%%%%%%%%
% starting point
%%%%%%%%%%%%%%%%
switch initflag
    case 0          % all zeros
        x = zeros(size(b));
    case 1          % random
        x = randn(size(b));
    case 999        % x is explicitly given, check dimension
        if  not( size(x) == size(b) )
            error('Invalid size of the initial point.');
        end
    otherwise
        error('Unknown initialization option.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% every image is treated as a 2D matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%
% stop criterion
%%%%%%%%%%%%%%%%
if not( ismember(stopcrit, [1 2 3 4]) )
    error('Unknown stopping criterion: ''%d''',num2str(stopcrit));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% settings for BB-like stelengths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iter = 1;                               % iteration counter
alpha = initalpha;                      % initial alpha
if isempty(InfoSGP)
    Valpha = alpha_max * ones(Malpha,1);    % memory buffer for alpha
else
    Valpha = InfoSGP.Valpha;
    alpha  = min(Valpha);
    tau    = InfoSGP.tau;
end
Fold = -1e30 * ones(M, 1);              % memory buffer for obj. func.

%%%%%%%%%%%%%%%%%
% projection type
%%%%%%%%%%%%%%%%%
switch (omega)
    case 'nonneg'
        pflag = 0;
    case 'nonneg_eq'
        pflag = 1;
    otherwise
        error('projection %s is not implemented',omega);
end

%%%%%%%%%%%%%%
% start of SGP
%%%%%%%%%%%%%%
% projection of the initial point
switch (pflag)
    case 0 % non negativity
        x( x < 0 ) = 0;
    case 1 % non negativity and flux conservation
        % we have no diagonal scaling matrix yet, so
        % we project using euclidean norm
        x = DFproj_sgp(flux, x, ones(size(x)));
end

% objective function value
Ax = A(x);
g = Ax - b;
fv = x(:)'*(0.5*Ax(:) - b(:));

if lambda>0
    [reggrad,regfv] = reg(x);
    g = g+lambda*reggrad;
    fv = fv + lambda*regfv;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bounds for the scaling matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% scaling matrix
if initflag == 0
    X = ones(size(x));
else
    if xscaling == 0
        X = ones(size(x));
    elseif xscaling == 1
        X = x;
        % bounds
        X( X < X_low_bound ) = X_low_bound;
        X( X > X_upp_bound ) = X_upp_bound;
    else
        X = Vreg(x);
        % bounds
        X( X < X_low_bound ) = X_low_bound;
        X( X > X_upp_bound ) = X_upp_bound;
    end
end

if pflag == 1
    D = 1./X;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tolerance for stop criterion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch stopcrit
    case 2
        if verb > 0
            fprintf('it %03d || x_k - x_(k-1) ||^2 / || x_k ||^2 %e \n',iter-1,0);
        end
        % instead of using sqrt each iteration
        tol = tol*tol;
    case 3
        if verb > 0
            fprintf('it %03d | f_k - f_(k-1) | / | f_k | %e \n',iter-1,0);
        end
end

%%%%%%%%%%%
% main loop
%%%%%%%%%%%
loop = true;
while loop
    % store alpha and objective function values
    Valpha(1:Malpha-1) = Valpha(2:Malpha);
    Fold(1:M-1) = Fold(2:M);
    Fold(M) = fv;
    
    % compute descent direction
    y = x - alpha*X.*g;
    
    switch (pflag) % projection onto the feasible set
        case 0 % non negativity
            y( y < 0 ) = 0;
        case 1 % non negativity and flux conservation
            y = DFproj_sgp(flux, y.*D, D);
    end
    
    d = y - x;
    
    % backtracking loop for linesearch
    gd = d(:)'*g(:);
    
    lam = 1;
    
    fcontinue = 1;
    Ad = A(d);    % exploiting linearity
    fr = max(Fold);
    
    while fcontinue
        xplus = x + lam*d;
        
        Ax_try = Ax + lam*Ad;
        fv = 0.5*xplus(:)'*Ax_try(:) - xplus(:)'*b(:);
        if lambda>0
            [reggrad,regfv] = reg(xplus);
            fv = fv + lambda*regfv;
        end
        
        if ( fv <= fr + gamma * lam * gd || lam < 1e-12)
            x = xplus; clear xplus;
            sk = lam*d;
            Ax = Ax_try; clear Ax_try;
            gtemp = Ax - b;
            if lambda>0
                gtemp = gtemp+lambda*reggrad;
            end
            
            yk = gtemp - g;
            g = gtemp; clear gtemp;
            fcontinue = 0;
        else
            lam = lam * beta;
        end
    end
    if (fv >= fr) && (verb > 0)
        disp('Warning: fv >= fr');
    end
    % alpha
    % lam
    % update the scaling matrix and the steplength
    if xscaling == 0
        X = ones(size(x));
    elseif xscaling == 1
        X = x;
        % bounds
        X( X < X_low_bound ) = X_low_bound;
        X( X > X_upp_bound ) = X_upp_bound;
    else
        X = Vreg(x);
        % bounds
        X( X < X_low_bound ) = X_low_bound;
        X( X > X_upp_bound ) = X_upp_bound;
    end
    
    D = 1./X;
    sk2 = sk.*D; yk2 = yk.*X;
    
    bk = sk2(:)'*yk(:);  ck = yk2(:)'*sk(:);
    
    if (bk <= 0)
        alpha1 = min(10*alpha,alpha_max);
    else
        alpha1BB = (sk2(:)'*sk2(:))/bk;
        alpha1 = min(alpha_max, max(alpha_min, alpha1BB));
    end
    if (ck <= 0)
        alpha2 = min(10*alpha,alpha_max);
    else
        alpha2BB = ck/(yk2(:)'*yk2(:));
        alpha2 = min(alpha_max, max(alpha_min, alpha2BB));
    end
    %     alpha1, alpha2
    Valpha(Malpha) = alpha2;
    
    if (iter <= 2) && isempty(InfoSGP)
        alpha = min(Valpha);
    elseif (alpha2/alpha1 < tau)
        alpha = min(Valpha);
        tau = tau*0.9;
    else
        alpha = alpha1;
        tau = tau*1.1;
    end
    %     tau
    alpha = double(single(alpha));
    
    iter = iter + 1;
    
    %%%%%%%%%%%%%%%
    % stop criteria
    %%%%%%%%%%%%%%%
    switch stopcrit
        case 1
            if verb > 0
                fprintf('it %03d of  %03d\n',iter-1,MAXIT);
            end
        case 2
            normstep = (sk(:)'*sk(:)) / (x(:)'*x(:));
            loop = (normstep > tol);
            if verb > 0
                fprintf('it %03d || x_k - x_(k-1) ||^2 / || x_k ||^2 %e tol %e\n',iter-1,normstep,tol);
            end
        case 3
            reldecrease = abs(fv - Fold(M)) / abs(fv);
            loop = (reldecrease > tol);
            if verb > 0
                fprintf('it %03d | f_k - f_(k-1) | / | f_k | %e tol %e\n',iter-1,reldecrease,tol);
            end
        case 4
            Ivar = zeros(size(x));
            Ivar(x <= 0) = -1;
            pgrad = projgrad(g,Ivar,pflag);
            PGnorm = norm(pgrad,'fro');
            loop = (PGnorm > tol);
    end
    
    if iter > MAXIT
        loop = false;
    end
    if iter <= MINIT
        loop = true;
    end
    if toc(t0) > MAXTIME
        loop = false;
    end
end

InfoSGP = struct('tau',tau,'iter',iter,'Valpha',Valpha);
iter = iter - 1;

end