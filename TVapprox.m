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

function [grad,fval] = TVapprox(W,TV,u)

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
% This function evaluates, at a given point u, the function value and the
% gradient of the IRN approximation of the smoothed TV centered at a point
% xk. See Section 3 in [1] for further details.
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
% W   = double matrix, weights for the TV approximation;
% TV  = double, smoothed TV function value in the center of the approximation;
% u   = double matrix, given point.
%
% OUTPUT ARGUMENTS
% 
% grad = double matrix, gradient of the TV approximation at u;
% fval = double, value of the TV approximation at u.
%
%==========================================================================

[D1u,D2u] = ForwardD(u);
OmegaD1 = W.*D1u; OmegaD2 = W.*D2u;

% Gradient evaluation
tempD1 = ForwardDXT(OmegaD1); tempD2 = ForwardDYT(OmegaD2);
grad = tempD1+tempD2;

% Function value evaluation
if nargout>1
    fval = 0.5*( D1u(:)'*OmegaD1(:) + D2u(:)'*OmegaD2(:) + TV );
end

end
