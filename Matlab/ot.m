function [objval,gammaij]=ot(cost,p1,p2);
%function ot() performs standard Optimal Transport

%cost[m1,m2] is the cost distance between any pair of clusters.
% When Wasserstein-2 metric is computed, cost() is the square of the Eulcidean distance between 
% two support points across the two distributions. The solution is in res.
% objval is the optimized objective function value. If Wasserstein-2 metric
% is to be solved (using the right cost defintion, see above), sqrt(objval)
% is the Wasserstein-2 metric.
%p1[m1] is the marginal for the first distribution, column vector
  %p2[m2] is the marginal for the second distribution, column vector

  
  %  cost=[0.1,0.2,1.0;0.8,0.8,0.1]; p1=[0.4;0.6]; p2=[0.2;0.3;0.5];

clear prob;

m1=size(cost,1); %number of support points in the first distribution
m2=size(cost,2); %number of support points in the second distribution

if (sum(p1)==0.0 || sum(p2)==0.0) 
  fprintf('Warning: total probability is zero: %f, %f\n',sum(p1),sum(p2));
  return;
end;

% Normalization
p1=p1/(sum(p1));
p2=p2/sum(p2);

coststack=reshape(cost,m1*m2,1); %column-wise scan
prob.c=coststack;
prob.blx=zeros(m1*m2,1);
prob.ulx=[inf*ones(m1*m2,1)];

	     prob.a=zeros(m1+m2,m1*m2);
	     prob.blc=zeros(m1+m2,1);
	     prob.buc=zeros(m1+m2,1);

% Generate subscript matrix for easy reference
	     wijsub=zeros(m1,m2);
	     k=1;
	   for j=1:m2
	   for i=1:m1
	     wijsub(i,j)=k;
	     k=k+1;
	     end;
	     end;
	     
% Set up the constraints
	   for i=1:m1
	   for j=1:m2
	     prob.a(i,wijsub(i,j))=1.0;
	     end;
	     prob.buc(i)=p1(i);
	     prob.blc(i)=p1(i);
	     end;

	   for j=1:m2
	   for i=1:m1
	     prob.a(j+m1,wijsub(i,j))=1.0;
	     end;
	     prob.buc(j+m1)=p2(j);
	     prob.blc(j+m1)=p2(j);
	     end;
	     

       param.MSK_IPAR_OPTIMIZER = 'MSK_OPTIMIZER_INTPNT';
       param.MSK_IPAR_LOG = 0;
       [r,res]=mosekopt('minimize echo(0)',prob,param);

       % To extract the optimized objective function value
       objval=res.sol.itr.pobjval;

       % To extract the matching weights solved
xx=res.sol.itr.xx;
	   gammaij=reshape(xx(1:m1*m2),[m1,m2]);
       


