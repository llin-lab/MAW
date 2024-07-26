import torch
import mosek
from scipy.linalg import sqrtm

def ot(cost,p1,p2):
    #function ot() performs standard Optimal Transport
    
    #cost[m1,m2] is the cost distance between any pair of clusters.
    # When Wasserstein-2 metric is computed, cost() is the square of the Eulcidean distance between 
    # two support points across the two distributions. The solution is in res.
    # objval is the optimized objective function value. If Wasserstein-2 metric
    # is to be solved (using the right cost defintion, see above), sqrt(objval)
    # is the Wasserstein-2 metric.
    #p1[m1] is the marginal for the first distribution, column vector
      #p2[m2] is the marginal for the second distribution, column vector

    #  cost=[0.1,0.2,1.0;0.8,0.8,0.1]; p1=[0.4;0.6]; p2=[0.2;0.3;0.5];

    
    m1 = cost.shape[0] #number of support points in the first distribution
    m2 = cost.shape[1] #number of support points in the second distribution

    if (torch.sum(p1) == 0.0 or torch.sum(p2) == 0.0):
      print("Warning: total probability is zero: %f, %f\n" % (torch.sum(p1), torch.sum(p2)))
      return

    # Normalization
    p1 = p1 / torch.sum(p1)
    p2 = p2 / torch.sum(p2)
    
    with mosek.Task() as task:
    
        coststack = torch.reshape(cost,m1 * m2, order='F')
        c = coststack
        blx = torch.zeros(m1 * m2)
        ulx = torch.inf * np.ones(m1 * m2)
        bkx = [mosek.boundkey.lo] * (m1 * m2)
        a = np.zeros((m1 + m2, m1 * m2))
        blc = torch.zeros(m1 + m2)
        buc = torch.zeros(m1 + m2)
        bkc = [mosek.boundkey.fx] * (m1 + m2)

        # Generate subscript matrix for easy reference
        wijsub = torch.zeros((m1, m2))
        k = 0
        for j in range(m2):
            for i in range(m1):
                wijsub[i,j] = k
                k = k + 1

        # Set up the constraints
        for i in range(m1):
            for j in range(m2):
                a[i, int(wijsub[i,j])] = 1.0
            buc[i] = p1[i]
            blc[i] = p1[i]


        for j in range(m2):
            for i in range(m1):
                a[j+m1, int(wijsub[i,j])] = 1.0
            buc[j+m1] = p2[j]
            blc[j+m1] = p2[j]
    
        numvar = len(blx)
        numcon = len(blc)

    
        # Append 'numcon' empty constraints.
        # The constraints will initially have no bounds.
        task.appendcons(numcon)

        # Append 'numvar' variables.
        # The variables will initially be fixed at zero (x=0).
        task.appendvars(numvar)

        for j in range(numvar):
            # Set the linear term c_j in the objective.
            task.putcj(j, c[j])

            # Set the bounds on variable j
            # blx[j] <= x_j <= bux[j]
            task.putvarbound(j, bkx[j], blx[j], ulx[j])
        
            asub = []
            aval = []
            for i in range(a.shape[0]):
                if a[i, j] != 0:
                    asub.append(i)
                    aval.append(a[i, j])

            # Input column j of A
            task.putacol(j,                  # Variable (column) index.
                         asub,            # Row index of non-zeros in column j.
                         aval)            # Non-zero Values of column j.

        for i in range(numcon):
            task.putconbound(i, bkc[i], blc[i], buc[i])
    
        task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.intpnt)
        task.putintparam(mosek.iparam.log, 0)
    
        # Input the objective sense (minimize/maximize)
        task.putobjsense(mosek.objsense.minimize)
    
        # Solve the problem
        task.optimize()

        # To extract the optimized objective function value
        objval = task.getprimalobj(mosek.soltype.itr)

        # To extract the matching weights solved
        xx = task.getxx(mosek.soltype.itr)
        gammaij = torch.reshape(np.array(xx)[0:m1*m2], (m1, m2), order='F')
       
    return({"objval": objval, "gammaij": gammaij})


def GaussWasserstein(d, supp1, supp2):
    # Compute the pairwise squared Wasserstein distance between each component in supp1 and each component in supp2
    # numcomponents in a distribution is size(supp1.2).
    #Suppose numcmp1=size(supp1,2), numcmp2=size(supp2,2)
    # Squared Wasserstein distance between two Gaussian:
    # \|\mu_1-\mu_2\|^2+trace(\Sigma_1+\Sigma_2-2*(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2})
    # For commutative case when \Sigma_1*\Sigma_2=\Sigma_2*\Sigma_1 (true for symmetric matrices)
    # The distance is equivalent to
    # \|\mu_1-\mu_2\|^2+\|Sigma_1^{1/2}-\Sigma_2^{1/2}\|_{Frobenius}^2
    # Frobenius norm of matrices is the Euclidean (L2) norm of the stacked
    # vector converted from the matrix
    # We use the commutative case formula to avoid more potential precision
    # errors

    numcmp1 = supp1.shape[1]
    numcmp2 = supp2.shape[1]
    pairdist = torch.zeros((numcmp1, numcmp2))

    for ii in range(numcmp1):
        for jj in range(numcmp2):
            sigma1 = torch.reshape(supp1[d:d+d*d,ii], (d,d), order='F')
            sigma2 = torch.reshape(supp2[d:d+d*d,jj], (d,d), order='F')

            b1=sqrtm_eig(sigma1); %use eigen value decomposition to solve squre root          
            b2=sqrtm_eig(sigma2);
            
            mudif = supp1[0:dim,ii] - supp2[0:dim,jj]
            pairdist[ii,jj] = torch.sum(mudif * mudif) + torch.sum((b1 - b2) * (b1 - b2))

    return(pairdist)

def Mawdist(d, supp1, supp2,w1,w2):
    # Compute the MAW distance between two GMM with Gusassian component parameters specified in supp1 and supp2 and prior specified to w1 and w2

    pairdist = GaussWasserstein(d, supp1, supp2)
    result = ot(pairdist, w1, w2)
    return({"dist": result["objval"], "gammaij": result["gammaij"]})


def sqrtm_eig(sigma):
    # Use Eigen-decomposition to compute the square root of a matrix
    evals, evecs = torch.eig(sigma, eigenvectors = True)
    if (torch.min(evals) < 0):
        print("Warning: the matrix is not positive semi-definite with the minimum eigenvalue: %f\n" % torch.min(evals))
        
    result = torch.matmul(evecs, torch.matmul(torch.diag(torch.sqrt(evals)), torch.inverse(evecs)))
    result = torch.view_as_real(result)
    return(result)

