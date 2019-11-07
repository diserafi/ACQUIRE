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

function [x] = DFproj_pg(c,l,u,pflag)

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
% This code is a modified version of the "projectDF" code from Zanella
% and Cavicchioli available at: http://www.unife.it/prisma/software it is
% used to evaluate the projected gradient at a given point x, i.e. the
% projection of the antigradient onto the tangent cone.
% "projectDF" is an implementation of the secant-based Dai-Fletcher
% algorithm [Dai-Fletcher, MathProg 2006] to solve the separable, singly
% linearly and nonnegatively constrained quadratic programming problem
%                         min  0.5 * x'*x - c'*x
%                         s.t. sum(x) = 0 [optional]
%                              l <= x <= u
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
% c     = double array, coefficient vector of the objctive's linear term;
% l     = double array, lower bound;
% u     = double array, upper bound;
% pflag = logical, if true the problem is subject to the linear constraint.
%
% OUTPUT ARGUMENTS
% 
% x     = double array, solution vector.
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
% Software homepage: http://www.unife.it/irma
%                    http://www.unife.it/prisma/software
%
% Copyright (C) 2011 by R. Cavicchioli, R. Zanella, G. Zanghirati, L. Zanni.
% ------------------------------------------------------------------------------
%
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

if pflag == 1
    
    lambda = 0;                 % initial lambda
    dlambda = 1;                % initial step
    tol_r = 1e-11;          % tolerance on the function
    tol_lam = 1e-11;            % tolerance on the step
    biter = 0;                  % bracketing phase iterations
    siter = 0;                  % secant phase iterations
    maxprojections = 1000;      % maximum number of iterations
    
    % Bracketing Phase
    x = min(max(l,(c+lambda)),u); r = sum(x(:));
    
    % check abs(r) < tol
    if ( abs(r) < tol_r )
        return;
    end
    
    if r < 0
        lambdal = lambda;
        rl = r;
        lambda = lambda+dlambda;
        x = min(max(l,(c+lambda)),u); r = sum(x(:));
        while r < 0
            biter = biter+1;
            lambdal = lambda;
            s = max(rl/r-1, 0.1);
            dlambda = dlambda+dlambda/s;
            lambda = lambda+dlambda;
            rl = r;
            x = min(max(l,(c+lambda)),u); r = sum(x(:));
        end
        lambdau = lambda;
        ru = r;
    else
        lambdau = lambda;
        ru = r;
        lambda = lambda-dlambda;
        x = min(max(l,(c+lambda)),u); r = sum(x(:));
        while r > 0
            biter = biter+1;
            lambdau = lambda;
            s = max(ru/r-1, 0.1);
            dlambda = dlambda+dlambda/s;
            lambda = lambda-dlambda;
            ru = r;
            x = min(max(l,(c+lambda)),u);  r = sum(x(:));
        end
        lambdal = lambda;
        rl = r;
    end
    
    % check ru and rl
    if (abs(ru) < tol_r)
        x = min(max(l,(c+lambdau)),u);
        return;
    end
    if (abs(rl) < tol_r)
        x = min(max(l,(c+lambdal)),u);
        return;
    end
    
    % Secant Phase
    s = 1-rl/ru;
    dlambda = dlambda/s;
    lambda = lambdau-dlambda;
    x = min(max(l,(c+lambda)),u); r = sum(x(:));
    maxit_s = maxprojections - biter;
    
    % Main loop
    while ( abs(r) > tol_r & ...
            dlambda > tol_lam * (1 + abs(lambda)) & ...
            siter < maxit_s )
        siter = siter + 1;
        if r > 0
            if s <= 2
                lambdau = lambda;
                ru = r;
                s = 1-rl/ru;
                dlambda = (lambdau-lambdal)/s;
                lambda = lambdau - dlambda;
            else
                s = max(ru/r-1, 0.1);
                dlambda = (lambdau-lambda) / s;
                lambda_new = max(lambda - dlambda, 0.75*lambdal+0.25*lambda);
                lambdau = lambda;
                ru = r;
                lambda = lambda_new;
                s = (lambdau - lambdal) / (lambdau-lambda);
            end
        else
            if s >= 2
                lambdal = lambda;
                rl = r;
                s = 1-rl/ru;
                dlambda = (lambdau-lambdal)/s;
                lambda = lambdau - dlambda;
            else
                s = max(rl/r-1, 0.1);
                dlambda = (lambda-lambdal) / s;
                lambda_new = min(lambda + dlambda, 0.75*lambdau+0.25*lambda);
                lambdal = lambda;
                rl = r;
                lambda = lambda_new;
                s = (lambdau - lambdal) / (lambdau-lambda);
            end
        end
        x = min(max(l,(c+lambda)),u); r = sum(x(:));
    end
    
else
    x = min(max(l,c),u);
end

end