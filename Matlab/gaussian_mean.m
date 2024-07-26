function Sigma=gaussian_mean(V, w, Sigma0)
% size(V) = [d*d, n]
% size(w) = [m, n]
% size(Sigma) = [d*d, m]

d=sqrt(size(V,1));
assert(size(V,1) == d*d);
n=size(V,2);
assert(n == size(w, 2))
m=size(w,1);
w = bsxfun(@times, w, 1./sum(w, 2));

Sigma=Sigma0;
if d > 1
Sigma=reshape(Sigma, [d d m]);
old_Sigma=zeros(d, d, m);
V=reshape(V, [d d n]);
while (max(abs(old_Sigma(:) - Sigma(:))) > 1E-5 * max(abs(Sigma(:))))
old_Sigma = Sigma;
Sigma=zeros(d, d, m);

for j=1:m
    mem = sqrtm_old(old_Sigma(:,:,j));
    %for i=1:n    
    %    Sigma(:,:,j) = Sigma(:,:,j) + w(j,i) * sqrtm_old(mem * V(:,:,i) * mem);
    %end
    Sigma(:,:,j) = sqrtm_batch_it(V, w(j,:), mem);
end
end
elseif d == 1
    V=V(:);
    Sigma=(w * sqrt(V)).^2; 
end

Sigma=reshape(Sigma, [d*d, m]);