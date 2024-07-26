# MATLAB, R and Python code for computing MAW distance and barycenter
This is a toolkit box that computes the MAW distance and barycenter under different coding languages.

To cite the reference, please use Lin L, Shi W, Ye J, Li J. (2023) Multisource single-cell data integration by MAW barycenter for Gaussian mixture models. Biometrics, 79, 866–877. [https://doi.org/10.1111/biom.13630]

## Instructions
### MATLAB code
1. Create a directory called "code", unpack the downloaded files and put them under the directory "code". In the discussions below, we assume "code" is the current directory. 

2. Need to obtain a valid mosek installation/license. If you already have one, you can put the mosek license in the folder code/mosek. See (https://docs.mosek.com/9.2/toolbox/install-interface.html) for details.

   After mosek package is downloaded and unpacked, a directory called mosek will be created usually under the home directory. Move the directory to ../gmm-barycenter/src/barycenter

3. Then compile *.c source files. For Mac OS (and probably need to install Accelerate framework from Apple Store, if haven't). Open a Matlab console under the directory: code/gmm-barycenter/src/barycenter
   Then run the following command in the Matlab console:
   
```
mex -v CLIBS="$CLIBS -lstdc++ -framework Accelerate" sqrtm_batch_it.c
mex -v CLIBS="$CLIBS -lstdc++ -framework Accelerate" sqrtm_batch_ud.c
```

The above commands will compile C source files: sqrtm_batch_it.c and sqrtm_batch_ud.c. 
#### For computing barycenter
4. Run Matlab under directory (working directory): ../gmm-barycenter/src/barycenter
   
   Data files named "*.mat" are put under directory: ../gmm-barycenter/src

6. In Matlab console, run barycenter_md.m located in ../gmm-barycenter/src/barycenter. The data file has to be put in the same directory ../gmm-barycenter/src.

DATA format assumed for the input data file "*.mat" of barycenter_md.m:
The *.mat data file should contain three objects: ww, supp, stride
ww: concatenates the prior proportions for all the GMMs estimated from all batches. Thus, ww is a vector of length J_1 + J_2 + ...+ J_N, where J_k is the number of mixture components for each data batch, and N is the total number of data batches.

supp: stores the mean and covariance for all the GMMs. More specifically, supp is a matrix of dimension (d+d*d)x(J_1+J_2+...+J_N), where d is the dimension of the data. For each column of supp, the first d rows store the mean vector, and the last d*d rows store the vectorized covariance matrix for each component. 

stride: It is an integer vector of the form (J_1,...,J_N). 

After running barycenter_md.m, it will output a matlab data file. 

The object in the output file named "c.*" contains all the information of the compuated MAW barycenter. More specifically, "c.w" contains the prior proportion for the barycenter, and "c.supp" is a matrix of size (d+d*d)x J, where J is the number of components specified for the barycenter. Similarly, each column of "c.supp" stores the mean vector and vectorized covariance for each component. 





