Mawdist <- function(d, supp1, supp2, w1, w2, method = "Schur") {
  # Compute the MAW distance between two GMM with Gusassian component parameters specified in supp1 and supp2 and prior specified to w1 and w2
  pairdist <- GaussWasserstein(d, supp1, supp2, method)
  result <- ot(pairdist,w1,w2)
  dist <- result$objval
  gammaij <- result$gammaij
  return(list(dist = dist, gammaij = gammaij))
  
}


GaussWasserstein <- function(d, supp1, supp2, method = "Schur") {
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
  # For method, it specifies the algorithm to compute the square root of a function
  # Schur as default use Schur decomposition, other two options is PRACMA which uses algorithm in pracma package
  # and Eigen uses eigendecomposition.
  
  numcmp1 <- dim(supp1)[2]
  numcmp2 <- dim(supp2)[2]
  pairdist <- matrix(0, nrow = numcmp1,ncol = numcmp2)
  
  if (method == "Schur" | is.null(method)) m <- sqrtm_old
  else if (method == "PRACMA") m <- sqrtm_pracma
  else if (method == "Eigen") m <- sqrtm_eigen 
  else stop("Please indicate a correct method to compute matrix square root.")
  
  for (ii in 1:numcmp1) {
    sigma1 <- pracma::Reshape(supp1[(d+1):(d+d*d),ii],d,d)
    # b1=sqrtm_eig(sigma1); #use eigen value decomposition to solve squre root          
    b1 <- m(sigma1)
    for (jj in 1:numcmp2) {
      sigma2 <- pracma::Reshape(supp2[(d+1):(d+d*d),jj],d,d)
      
      # b2=sqrtm_eig(sigma2);
      b2 <- m(sigma2)
      
      mudif <- supp1[1:d,ii] - supp2[1:d,jj]
      pairdist[ii,jj] = sum(mudif*mudif) + sum((b1-b2)*(b1-b2))
      
      #a=sigma1+sigma2-2*sqrtm(b1*sigma2*b1);
      #pairdist(ii,jj)=sum(mudif.*mudif)+sum(diag(a));
    }
  }
  return(pairdist)
}




ot <- function(cost,p1,p2) {
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
  
  m1 <- dim(cost)[1] #number of support points in the first distribution
  m2 <- dim(cost)[2] #number of support points in the second distribution
  if (sum(p1) == 0.0 || sum(p2) == 0.0) {
    printf('Warning: total probability is zero: %f, %f',sum(p1),sum(p2))
    return()
  }
  # Normalization
  p1 <- p1/(sum(p1))
  p2 <- p2/sum(p2)
  
  coststack <- c(cost) #column-wise scan
  prob <- list()
  prob$c <- coststack
  blx <- rep(0, m1*m2)
  ulx <- Inf*rep(1, m1*m2)
  
  prob$A = matrix(0, nrow = m1+m2, ncol = m1*m2)
  blc <- rep(0, m1+m2)
  buc <- rep(0, m1+m2)
  
  
  # Generate subscript matrix for easy reference
  wijsub <- matrix(0, nrow = m1, ncol = m2)
  k <- 1
  for (j in 1:m2) {
    for (i in 1:m1) {
      wijsub[i,j] = k;
      k <- k+1
    }
  }
  
  
  # Set up the constraints
  for (i in 1:m1) {
    for (j in 1:m2) {
      prob$A[i,wijsub[i,j]] <- 1.0
    }
    buc[i] <- p1[i]
    blc[i] <- p1[i]
  }
  
  for (j in 1:m2) {
    for (i in 1:m1) {
      prob$A[j+m1,wijsub[i,j]] <- 1.0
    }
    buc[j+m1] <- p2[j]
    blc[j+m1] <- p2[j]
  }
  
  prob$bx <- rbind(blx = blx, bux = ulx)
  prob$bc <- rbind(blc = blc, buc = buc)
  
  prob$iparam <- list(OPTIMIZER = "OPTIMIZER_INTPNT", LOG = 0)
  prob$sense <- "minimize"
  r <- Rmosek::mosek(prob, list(soldetail=1))
  
  # To extract the optimized objective function value
  objval <- r$sol$itr$pobjval
  
  # To extract the matching weights solved
  xx <- r$sol$itr$xx
  gammaij <- pracma::Reshape(xx[1:(m1*m2)],m1,m2)
  
  
  return(list(objval = objval,gammaij = gammaij))
}


sqrtm_old <- function(A) {
  n <- dim(A)[1]
  result <- Matrix::Schur(A)
  Q <- result$Q
  Tr <- result$`T`
  R <- matrix(0, nrow = n, ncol = n)
  
  if (sum(diag(Tr) == 0) <= 1) {
    if (sum(diag(Tr) < 0) == 0) {
      # Compute the square root of an upper triangular matrix with at most 1 zero element on diagonal
      for (j in 1:n) {
        R[j,j] <- sqrt(Tr[j,j])
        if (j > 1) {
          for (i in (j-1):1) {
            s <- 0
            if ((i+1) <= (j-1)) {
              for (k in (i+1):(j-1)) {
                s <- s + R[i,k] * R[k,j]
              }
            }
            if (R[i,i] + R[j,j] == 0) R[i,j] <- (Tr[i,j] - s) / (R[i,i] + R[j,j] + pracma::eps(x = 1))
            else R[i,j] <- (Tr[i,j] - s) / (R[i,i] + R[j,j])
          }
        }
        
      }
      X <- Q %*% R %*% t(Q)
    }
    else {
      # Compute the square root of an upper triangular matrix with at most 1 zero element but some small negative values (close to 0) on diagonal.
      #if (Tr[j,j] < 0 & Tr[j,j] > -1e-4) {
      #warning("Due to the computational limit, the covariance matrix is positive semi-definite and the eigenvalues are negative in format.")
      #}
      for (j in 1:n) {
        R[j,j] <- sqrt(as.complex(Tr[j,j]))
        if (j > 1) {
          for (i in (j-1):1) {
            s <- 0
            if ((i+1) <= (j-1)) {
              for (k in (i+1):(j-1)) {
                s <- s + R[i,k] * R[k,j]
              }
            }
            R[i,j] <- (Tr[i,j] - s) / (R[i,i] + R[j,j])
          }
        }
        
      }
      X <- Re(Q %*% R %*% t(Q))
    }
  } 
  else stop(paste("In sqrm_old function, the symmetric matrix may be negative definite,",
                  "or the upper triangular matrix in Schur decomposition has at least 2 zero elements on diagonal,", 
                  "which may indicate that the square root of the matrix may not exist."))
  
  return(X)
}


sqrtm_eig <- function(A) {
  n <- dim(A)[1]
  result <- eigen(A, only.values = FALSE)
  if (sum(result$values < 0) == 0) return(result$vectors %*% diag(sqrt(result$values)) %*% solve(result$vectors))
  else {
    lambda <- diag(sqrt(as.complex(result$values)))
    return(Re(result$vectors %*% lambda %*% solve(result$vectors)))
  }
}

sqrtm_pracma <- function(A) {
  n <- dim(A)[1]
  result <- pracma::sqrtm(A)
  return(result$B)
}



