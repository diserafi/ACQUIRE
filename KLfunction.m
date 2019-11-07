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

function [ KL ] = KLfunction(A,AisPSF,x,sizex,b,y)

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
% This function evaluates the Kullback-Leibler divergence of the blurred image
% (A*x + b) from the observed image (y) (see eq (2) in [1] for details).
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
% A      = double matrix, representing the blurring operator (applied as
%          A*x(:)); periodic boundary conditions are assumed; A must have
%          nonnegative entries and its columns must sum-up to 1;
%          for efficiency, it is recommended to provide A as a point
%          spread function (PSF), so that the product A*x(:) will be
%          computed as a pointwise product in the Fourier space;
% AisPSF = logical, if true matrix A is a PSF;
% x      = double matrix, point into which the function is evaluated;
% sizex  = integer array, size of x;
% b      = double, background noise;
% y      = double matrix, obseved image.
%
% OUTPUT ARGUMENTS
% 
% KL     = double, KL function value.
%
%==========================================================================

if AisPSF
    Axk = real(ifft2(A.*fft2(x)));
else
    Axk = reshape(A*x(:),sizex);
end

temp = Axk + b;
KL = sum(sum( y.*log(y./temp)+temp-y )); % Bertero et al.

end