%%%% General matlab commands for both 1d and md
% module avail
% matlab run under: ../gmm-barycenter 2/src/barycenter
% .mat data is put under: ../gmm-barycenter 2/src


% module purge
% module load gcc
% module load matlab/R2016a
% matlab

clear;clc;
s1 = '../mouse_2';

setparam;
load (strcat(s1,'.mat')) 

%%
%N: number of instances
%m: number of states
%d: dimension

%m = 10;
N = 4;
d=2;

stride = stride';
%stride=m*ones(1,N); 
%stride = [7,10,8,8];
instanceW=ones(1,N);
%instanceW=rand(1,N); % not necessarily sum2one
%% compute GMM barycenter with B-ADMM method allowing the change of component weights
% initiate centroid from an instance
%c0.supp=supp(:, 1:5*m);
%c0.w=ww(1:5*m)/sum(ww(1:5*m));
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
save(strcat(s1,'_18_OT.mat'), 'c', 'X')

