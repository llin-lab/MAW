import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm

def centroid_sphBregman_GMM(stride, instanceW, supp, w, c0, options):
    d = int(np.sqrt(supp.shape[0]))
    n = len(stride)
    m = len(w)
    posvec = np.concatenate(([0], np.cumsum(stride) + 1))
    
    if c0 is None:
        raise ValueError('Please give a GMM barycenter initialization.')
        c = centroid_init(stride, supp, w, options)
    else:
        c = c0
        
    support_size = len(c['w'])
    
    X = np.zeros((support_size, m))
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    
    spIDX_rows = np.zeros((support_size * m,), dtype=int)
    spIDX_cols = np.zeros_like(spIDX_rows)
    
    for i in range(n):
        xx, yy = np.meshgrid(np.arange((i) * support_size, (i + 1) * support_size),
                             np.arange(posvec[i], posvec[i + 1]))
        ii = support_size * (posvec[i] - 1) + np.arange(support_size * stride[i])
        spIDX_rows[ii] = xx.flatten()
        spIDX_cols[ii] = yy.flatten()
    
    spIDX = np.kron(np.eye(support_size), np.ones((n, n)))
    
    suppW = np.zeros(m)
    for i in range(n):
        suppW[posvec[i]:posvec[i + 1]] = instanceW[i]
        Z[:, posvec[i]:posvec[i + 1]] = 1 / (support_size * stride[i])
    
    if d > 1:
        C = cdist(c['supp'][:d].T, supp[:d].T, 'sqeuclidean')
    else:
        C = cdist(c['supp'][:d].T, supp[:d].T, 'sqeuclidean')
    
    C += gaussian_wd(c['supp'][d:], supp[d:])
    
    nIter = options.get('badmm_max_iters', 2000)
    rho = options.get('badmm_rho', 10) * np.median(C, axis=0).mean()
    
    tau = options.get('tau', 10)
    badmm_tol = options.get('badmm_tol', 1e-4)
    
    C *= suppW
    rho *= np.median(instanceW)
    
    for iter in range(nIter):
        X = Z * np.exp((C + Y) / (-rho)) + np.finfo(float).eps
        X /= np.sum(X, axis=0, keepdims=True)
        X *= w
        
        Z0 = Z.copy()
        Z = X * np.exp(Y / rho) + np.finfo(float).eps
        
        spZ = coo_matrix((Z.flatten(), (spIDX_rows, spIDX_cols)),
                         shape=(support_size * n, m))
        tmp = np.asarray(spZ.sum(axis=1)).flatten()
        tmp = tmp.reshape(support_size, n)
        dg = 1 / tmp * c['w']
        dg = coo_matrix(np.diag(dg.flatten()))
        Z = np.asarray(spIDX @ dg @ spZ.tocsr())
        
        Y += rho * (X - Z)
        
        tmp = tmp / np.sum(tmp, axis=1, keepdims=True)
        sumW = np.sum(np.sqrt(tmp), axis=1) ** 2
        c['w'] = sumW / np.sum(sumW)
        
        if iter % tau == 0 and 'support_points' not in options:
            tmpX = X * suppW[:, None]
            c['supp'][:d] = (supp[:d] @ tmpX.T) / np.sum(tmpX, axis=0)
            c['supp'][d:] = gaussian_mean(supp[d:], tmpX, c['supp'][d:])
            
            if d > 1:
                C = cdist(c['supp'][:d].T, supp[:d].T, 'sqeuclidean')
            else:
                C = cdist(c['supp'][:d].T, supp[:d].T, 'sqeuclidean')
            
            C += gaussian_wd(c['supp'][d:], supp[d:])
            C *= suppW
        
        primres = np.linalg.norm(X - Z, 'fro') / np.linalg.norm(Z, 'fro')
        dualres = np.linalg.norm(Z - Z0, 'fro') / np.linalg.norm(Z, 'fro')
        
        if iter % 100 == 0 or iter == 10:
            print(f'\t {iter} {np.sum(C @ X * suppW) / np.sum(instanceW)} {primres} {dualres}')
            if np.sqrt(dualres * primres) < badmm_tol:
                break
    
    return {'c': c, 'X': X}

def gaussian_mean(V, w, Sigma0):
    if np.ndim(V) == 1:
        V = V[:, np.newaxis]
    
    d = int(np.sqrt(V.shape[0]))
    n = V.shape[1]
    m = w.shape[0]
    w = w / np.sum(w, axis=0, keepdims=True)
    
    Sigma = Sigma0.copy()
    
    if d > 1:
        Sigma = Sigma.reshape((d, d, m))
        old_Sigma = np.zeros_like(Sigma)
        V = V.reshape((d, d, n))
        
        while np.max(np.abs(old_Sigma - Sigma)) > 1e-5 * np.max(np.abs(Sigma)):
            old_Sigma = Sigma.copy()
            Sigma = np.zeros_like(Sigma)
            
            for j in range(m):
                mem = sqrtm(old_Sigma[:, :, j])
                for i in range(n):
                    Sigma[:, :, j] += w[j, i] * sqrtm(mem @ V[:, :, i] @ mem)
    elif d == 1:
        V = V.flatten()
        Sigma = (w @ np.sqrt(V)) ** 2
    
    return Sigma

def gaussian_wd(V1, V2):
    if np.ndim(V1) == 1:
        V1 = V1[:, np.newaxis]
    if np.ndim(V2) == 1:
        V2 = V2[:, np.newaxis]
    
    d = int(np.sqrt(V1.shape[0]))
    n1 = V1.shape[1]
    n2 = V2.shape[1]
    
    D = np.zeros((n1, n2))
    V1 = V1.reshape((d, d, n1))
    V2 = V2.reshape((d, d, n2))
    
    for i in range(n1):
        if d > 1:
            d1 = sqrtm(V1[:, :, i])
            t1 = np.sum(np.diag(V1[:, :, i]))
        else:
            d1 = np.sqrt(V1[:, :, i])
            t1 = V1[:, :, i]
        
        for j in range(n2):
            if d > 1:
                v2 = V2[:, :, j]
                d2 = sqrtm(d1 @ v2 @ d1)
                D[i, j] = t1 + np.sum(np.diag(v2)) - 2 * np.sum(np.diag(d2))
            else:
                v2 = V2[:, j]
                d2 = np.sqrt(d1 * v2 * d1)
                D[i, j] = t1 + v2 - 2 * d2
    
    return D

# Placeholder for centroid_init function which you need to implement
def centroid_init(stride, supp, w, options):
    # Implement the centroid_init function as per your requirements
    pass

