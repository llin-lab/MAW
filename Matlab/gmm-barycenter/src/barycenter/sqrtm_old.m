function X = sqrtm_old(A)
n=size(A,1);
[Q, T] = schur(A);
R = zeros(n);
for j=1:n
    R(j,j) = sqrt(T(j,j));
    for i=j-1:-1:1
        s=0;
        for k=i+1:j-1
            s = s + R(i,k)*R(k,j);
        end
        R(i,j)=(T(i,j)-s)/(R(i,i)+R(j,j));
    end
end
X=Q*R*Q';