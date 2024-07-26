function [c, X] = centroid_sphBregman_GMM(stride, instanceW, supp, w, c0, options) 
  % The algorithmic prototype of Wasserstein Barycenter using Bregman ADMM
  % 
  % This code has been created by Jianbo Ye (jxy198 [AT] ist.psu.edu).
  % 
  d = floor(sqrt(size(supp,1)));
  n = length(stride);
  m = length(w);
  posvec=[1,cumsum(stride)+1];

  if isempty(c0)
    error('Please give a GMM barycenter initialization.');
    c=centroid_init(stride, supp, w, options);
  else
    c=c0;
  end
  support_size=length(c.w);  
  %load(['cstart' num2str(n) '-' num2str(support_size) '.mat']);

  X = zeros(support_size, m);
  Y = zeros(size(X)); Z = X;
  spIDX_rows = zeros(support_size * m,1);
  spIDX_cols = zeros(support_size * m,1);
  for i=1:n
      [xx, yy] = meshgrid((i-1)*support_size + (1:support_size), posvec(i):posvec(i+1)-1);
      ii = support_size*(posvec(i)-1) + (1:(support_size*stride(i)));
      spIDX_rows(ii) = xx';
      spIDX_cols(ii) = yy';
  end
  spIDX = repmat(speye(support_size), [1, n]);
  
  
  % initialization
  suppW=zeros(1,m);
  for i=1:n
      suppW(posvec(i):posvec(i+1)-1)=instanceW(i);
      Z(:,posvec(i):posvec(i+1)-1) = 1/(support_size*stride(i));
  end
  C = pdist2(c.supp(1:d,:)', supp(1:d,:)', 'euclidean').^2;
  C = C + gaussian_wd(c.supp((d+1):end,:), supp((d+1):end,:));
  nIter = 2000;     
  if isfield(options, 'badmm_max_iters')
      nIter=options.badmm_max_iters;
  end
  
  if isfield(options, 'badmm_rho')
      rho = options.badmm_rho*mean(median(C));
  else
      rho = 10.*mean(median(C));
  end
  
  C = bsxfun(@times, C, suppW);
  rho = rho * median(instanceW);
  
  if isfield(options, 'tau')
      tau=options.tau;
  else
      tau=10;
  end
  
  if isfield(options, 'badmm_tol')
      badmm_tol=options.badmm_tol;
  else
      badmm_tol=1E-4;
  end
  for iter = 1:nIter
      % update X
      X = Z .* exp((C+Y)/(-rho)) + eps;
      X = bsxfun(@times, X, w./sum(X,1));
      
      % update Z
      Z0 = Z;
      Z = X .* exp(Y/rho) + eps;
      spZ = sparse(spIDX_rows, spIDX_cols, Z(:), support_size * n, m);
      tmp = full(sum(spZ, 2)); tmp = reshape(tmp, [support_size, n]);
      dg = bsxfun(@times, 1./tmp, c.w'); 
      dg = sparse(1:support_size*n, 1:support_size*n, dg(:));
      Z = full(spIDX * dg * spZ);
      
      % update Y      
      Y = Y + rho * (X - Z);
      
      % update c.w
      tmp = bsxfun(@times, tmp, 1./sum(tmp));
      sumW = sum(sqrt(tmp),2)'.^2; % (R2)
      %sumW = sum(tmp,2)'; % (R1)
      c.w = sumW / sum(sumW);
      %c.w = Fisher_Rao_center(tmp');
      
      % update c.supp and compute C (lazy)
      if mod(iter, tau)==0 && ~isfield(options, 'support_points')
        tmpX = bsxfun(@times, X, suppW);
        c.supp(1:d,:) = supp(1:d,:) * tmpX' ./ repmat(sum(tmpX,2)', [d, 1]);
        c.supp((d+1):end,:) = gaussian_mean(supp((d+1):end,:), tmpX, c.supp((d+1):end,:));
        C = pdist2(c.supp(1:d,:)', supp(1:d,:)', 'euclidean').^2;
        C = C + gaussian_wd(c.supp((d+1):end,:), supp((d+1):end,:));
        C = bsxfun(@times, C, suppW);
      end
      
      % The constraint X=Z are not necessarily strongly enforced
      % during the update of w, which makes it suitable to reset
      % lagrangian multipler after a few iterations
      if (mod(iter, 100) == 0)
%          Y(:,:) = 0;
%           if primres > 10*dualres
%             rho = 2 * rho;
%             fprintf(' *2');
%           elseif 10*primres < dualres
%             rho = rho / 2;
%             fprintf(' /2');
%           end
      end
      
      % output
      if (mod(iter, 100) == 0 || iter == 10)
          primres = norm(X-Z,'fro')/norm(Z,'fro');
          dualres = norm(Z-Z0,'fro')/norm(Z,'fro');
          fprintf('\t %d %f %f %f ', iter, ...
              sum(sum(bsxfun(@times, C.*X, suppW))) / sum(instanceW), ...
              primres, dualres);
          fprintf('\n');       
          if sqrt(dualres * primres)<badmm_tol
              break;
          end
      end
  end
end