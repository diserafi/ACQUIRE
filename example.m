%==========================================================================
%                        EXAMPLE OF USE OF ACQUIRE
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

%% Select the problem
pbind = 1;  % 1 - PHANTOM with Gaussian blur and SNR approx 40
            % 2 - CAMERAMAN with motion blur and SNR approx 40

if pbind==1
    load TEST_phantom_gauss.mat
else
    load TEST_cameraman_motion.mat
end

%% Run ACQUIRE with default settings and tolerance 1e-3 on the relative distance between consecutive iterates
options = struct('StopCrit',1,'Tolerance',1e-3);
[xk,flag,i,errvect,fvect,times] = acquire(psf,true,gn,bg,lambda,false,[],obj,options);

%% Plot relative error and objective function values versus elapsed time
figure()
semilogy(times,errvect,'LineWidth',2)
title('RELATIVE ERROR vs TIME')

figure()
semilogy(times,fvect,'LineWidth',2)
title('OBJ FUNCTION VALUE vs TIME')

%% Plot corrupted and restored image
figure()
imagesc(gn)
colormap gray
axis square
title('Corrupted image')

figure()
imagesc(xk)
colormap gray
axis square
title('Restored image')