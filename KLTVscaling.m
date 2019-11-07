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

function [scaling] = KLTVscaling(auxvector,u,Omega,lambda)

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
% This function evaluates the scaling matrix, at a given point u, for the
% Scaled Gradient Projection algorithm in the case of KLTV functional.
% The scaling is obtained by means of the splitting technique described,
% e.g., in [Zanella et al., Inverse Problems, 25(4), article 045010, 2009]
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
%     https://arxiv.org/abs/1807.10832% and Optimization Online
%     http://www.optimization-online.org/DB_HTML/2018/07/6745.html
%
%==========================================================================
%
% INPUT ARGUMENTS
% 
% auxvector = double matrix, auxiliary vector corresponding to AT*ones(size(u));
% u         = double matrix, current point;
% Omega     = double matrix, weights for the IRN approximation of the TV
%             functional;
% lambda    = double, weight of the TV regularization term.
%
% OUTPUT ARGUMENTS
%
% scaling   = double matrix, scaling matrix for SGP.
%
%==========================================================================

tmp1 = [Omega(:,1:end-1)+Omega(:,2:end), Omega(:,1) + Omega(:,end)];
tmp2 = [Omega(1:end-1,:)+Omega(2:end,:); Omega(1,:) + Omega(end,:)];

scaling = (tmp1 + tmp2).*u;scaling = lambda*scaling + auxvector;

scaling = u./scaling;

end