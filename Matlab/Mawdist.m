function [dist,gammaij]=Mawdist(dim, supp1, supp2,w1,w2);
% Compute the MAW distance between two GMM with Gusassian component parameters specified in supp1 and supp2 and prior specified to w1 and w2

pairdist=GaussWasserstein(dim, supp1, supp2);
[dist,gammaij]=ot(pairdist,w1,w2);
