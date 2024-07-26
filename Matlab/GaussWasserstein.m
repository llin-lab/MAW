function pairdist=GaussWasserstein(dim, supp1, supp2);
  % Compute the pairwise squared Wasserstein distance between each component in supp1 and each component in supp2
% numcomponents in a distribution is size(supp1.2).
%Suppose numcmp1=size(supp1,2), numcmp2=size(supp2,2)
% Squared Wasserstein distance between two Gaussian:
% \|\mu_1-\mu_2\|^2+trace(\Sigma_1+\Sigma_2-2*(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2})
% For commutative case when \Sigma_1*\Sigma_2=\Sigma_2*\Sigma_1 (true for symmetric matrices)
% The distance is equivalent to
% \|\mu_1-\mu_2\|^2+\|Sigma_1^{1/2}-\Sigma_2^{1/2}\|_{Frobenius}^2
% Frobenius norm of matrices is the Euclidean (L2) norm of the stacked
% vector converted from the matrix
% We use the commutative case formula to avoid more potential precision
% errors

numcmp1=size(supp1,2);
numcmp2=size(supp2,2);
pairdist=zeros(numcmp1,numcmp2);

for ii=1:numcmp1
	 for jj=1:numcmp2
		  sigma1=reshape(supp1(dim+1:dim+dim*dim,ii),[dim,dim]);
		  sigma2=reshape(supp2(dim+1:dim+dim*dim,jj),[dim,dim]);

% b1=sqrtm_eig(sigma1); %use eigen value decomposition to solve squre root          
% b2=sqrtm_eig(sigma2);
b1=sqrtm(sigma1);
b2=sqrtm(sigma2);

mudif=supp1(1:dim,ii)-supp2(1:dim,jj);
pairdist(ii,jj)=sum(mudif.*mudif)+sum(sum((b1-b2).*(b1-b2)));

%a=sigma1+sigma2-2*sqrtm(b1*sigma2*b1);
%pairdist(ii,jj)=sum(mudif.*mudif)+sum(diag(a));

end; % jj
end; %ii




