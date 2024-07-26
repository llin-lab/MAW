clear;clc;

load("<data folder>/mouse_2.mat");

supp1=supp(:,1:stride(1));
supp2=supp(:,(stride(1)+1):(stride(1)+stride(2)));
w1=ww(1:stride(1));
w2=ww((stride(1)+1):stride(1)+stride(2));
[dist,gammaij] = Mawdist(2, supp1, supp2, w1, w2);
save("<path to result>/mouse_2_outcome.mat", "dist", "gammaij")
