%%%% General matlab commands for both 1d and md
% module avail
% matlab run under: ../gmm-barycenter 2/src/barycenter
% .mat data is put under: ../gmm-barycenter 2/src


% module purge
% module load gcc
% module load matlab/R2016a
% matlab

clear;clc;
s1 = '../mouse_2'; % path for the data

setparam;
load (strcat(s1,'.mat')) 

%%
%N: number of data batches
%m: number of states, i.e., the number of mixture components for MAW barycenter
%d: data dimension

m = 15; %% user should specify the number of components for MAW barycenter
stride=stride';
N =length(stride);
instanceW=ones(1,N); 
%% compute GMM barycenter with B-ADMM method allowing the change of component weights
% initiate centroid from an instance
c0.supp=supp(:, 1:m);
c0.w=ww(1:m)/sum(ww(1:m));
% set the number of iterations and badmm_rho (no need to change)
options.badmm_max_iters=1000;
options.badmm_rho=10;
%options.tau = 50;
% compute the centroid
tic;
[c, X]=centroid_sphBregman_GMM(stride, instanceW, supp, ww, c0, options);
toc;
%%%%%%%%%
save(strcat(s1,'_OT.mat'), 'c', 'X')

