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

function [xmin,it,fmin] = line_search(xk,dk,gk,fun,fr)

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
% This function performs an Armijo backtracking line search along the
% direction dk, starting from a given point xk.
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
% xk   = double matrix, starting point;
% dk   = double matrix, search direction;
% gk   = double matrix, gradient of the objective function at xk;
% fun  = function handle to the objective function;
% fr   = double, reference value.
%
% OUTPUT ARGUMENTS
% 
% xmin = double matrix, point satisfying the Armijo condition;
% it   = integer, number of iterations,
% fmin = double, function value at the returned point.
%
%==========================================================================

dphi0 = dk(:)'*gk(:);

sigma = 1/2;
mu = 1.e-5;
it = 0;
alfa = 1;
xmin = xk+dk;

fmin = fun(xmin);
gdx = dphi0;

while (fmin > fr + mu*gdx) && (it < 10)
    alfa1 = sigma*alfa;
    xmin = xk + alfa1*dk;
    fmin = fun(xmin);
    gdx = gk(:)'*(xmin(:)-xk(:));
    alfa = alfa1;
    it = it+1;
end

end