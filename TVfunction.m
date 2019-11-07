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

function [TV, Omega] = TVfunction(v,tau)

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
% This function evaluates the Huber-like (with parameter tau) smoothed
% Total Variation for a given image v. We assume periodic boundary
% conditions for the finite difference operators. See Section 2 in [1].
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
% v      = double matrix, given image;
% tau    = double, coefficient for the Huber-like smoothed TV.
%
% OUTPUT ARGUMENTS
% 
% TV     = double, smoothed TV function value;
% Omega  = double matrix, weights for the Iteratively Reweight Norm
%          approximation of the smoothed TV centered in the given point
%          (see Section 3 in [1]).
%
%==========================================================================

[D1x,D2x] = ForwardD(v);
phi = sqrt(D1x.^2+D2x.^2);

Omega = phi;
index = phi<tau;
phi(index) = 0.5.*(phi(index).^2/tau+tau);
TV = sum(sum(phi));

Omega(index) = 1/tau;
Omega(~index) = 1./Omega(~index);

end