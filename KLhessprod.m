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

function [y] = KLhessprod(w,A,AT,AisPSF,v,sizev,delta)

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
% This function computes the mat-vec product y between the Hessian of the
% quadratic approximation of the KL function and a vector v, i.e.
%                           y = AT*W*A*v + delta*v,
% where W is the diagonal matrix with diagonal entries w.
% See Section 3 in [1] for further details.
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
% w      = double matrix, diagonal elements of the weight matrix W;
% A      = double matrix, representing the blurring operator (applied as
%          A*x(:)); periodic boundary conditions are assumed; A must have
%          nonnegative entries and its columns must sum-up to 1;
%          for efficiency, it is recommended to provide A as a point
%          spread function (PSF), so that the product A*x(:) will be
%          computed as a pointwise product in the Fourier space;
%          for efficiency, if A is a PSF then the user is asked to provide
%          as input fft(A) in place of A;
% AT     = double matrix, conjugate of A; for efficiency, if A is a PSF
%          then the user is asked to provide as input fft(AT) in place of AT;
% AisPSF = logical, if true matrix A is a PSF;
% v      = double matrix, "vector" for the matrix-vector product;
% sizev  = integer array, size of v;
% delta  = double, strong convexity constant for the KL approximation.
%
% OUTPUT ARGUMENTS
% 
% y      = double matrix, result of the product reshaped as a matrix with
%          the same size as v.
%
%==========================================================================

if AisPSF
    y = w.*(real(ifft2(A.*fft2(v))));
    y = real(ifft2(AT.*fft2(y)));
else
    y = w.*(A*v(:));
    y = AT*y;
    y = reshape(y,sizev);
end
if nargin>5 && delta>0
    y = y + delta*v;
end

end