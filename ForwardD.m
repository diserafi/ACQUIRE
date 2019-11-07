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

function [DXU,DYU] = ForwardD(U)

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
% This function is used to compute the forward first-order finite differences
% for a double matrix U along both columns and rows. The outputs will be used
% for the evaluation of the Total Variation and its derivatives.
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
% INPUT ARGUMENTS
% 
% U   = double matrix.
%
% OUTPUT ARGUMENTS
% 
% DXU = double matrix, forward column differences; each entry (i,j) contains
%              DXU(i,j)   = U(i,j+1)-U(i,j),
%       we assume periodic boundary conditions, i.e.,
%              DXU(i,end) = U(i,1)-U(i,end);
% DYU = double matrix, forward row differences; each entry (i,j) contains
%              DYU(i,j)   = U(i+1,j)-U(i,j),
%       we assume periodic boundary conditions, i.e.,
%              DYU(end,j) = U(1,j)-U(end,j);
%
%==========================================================================

DXU = [diff(U,1,2), U(:,1) - U(:,end)];
DYU = [diff(U,1,1); U(1,:) - U(end,:)];

end