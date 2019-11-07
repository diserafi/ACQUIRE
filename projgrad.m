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

function [PG] = projgrad(g,Ivar,pflag)

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
% This function computes the projected gradient associated with the problem
%               min  f(x)
%               s.t. x in Omega
% The feasible set Omega represents either nonnegativity constraints,
% i.e.,
%       Omega = {x : x_i>=0 for all i}
% or nonnegativity contraints plus a linear constraint, i.e.,
%       Omega = {x : (x_i>=0, for all i) & (sum(x) = a0}.
% The shape of the feasible set is specified by "pflag".
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
% g      = double matrix, gradient at the given point;
% Ivar   = integer matrix, information on the activity of the bound
%          constraints at the given point, i.e. for each (i,j) Ivar(i,j) is
%           -1: if the (i,j)-th component of the given point is on the
%               lower bound, 
%            0: if the (i,j)-th component of the given point is free,
%            1: if the (i,j)-th component of the given point is on the
%               upper bound; 
% pflag  = logical, if true the problem is subject to the linear equality
%          constraint. 
%
% OUTPUT ARGUMENTS
% 
% PG     = double matrix, projected gradient at the given point.
%
%==========================================================================

l = -Inf*ones(size(g));
u = Inf*ones(size(g));
l(Ivar == -1) = 0;
u(Ivar == 1) = 0;
PG = DFproj_pg(-g,l,u,pflag);

end