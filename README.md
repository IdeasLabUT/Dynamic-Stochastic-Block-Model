# Dynamic Stochastic Block Models MATLAB Toolbox

This toolbox contains MATLAB implementations of two stochastic block models (SBMs) for analyzing dynamic network data in the form of network snapshots at discrete time. The first model, the dynamic SBM (DSBM), also referred to as the hidden Markov SBM (HM-SBM), was proposed by Xu & Hero (2014) and makes a hidden Markov assumption on the edge dynamics. The second model, the stochastic block transition model (SBTM) was proposed by Xu (2015) and offers additional model flexibility compared to the DSBM at the cost of a larger number of parameters and slower inference procedure.

## Contents

The toolbox currently includes the following:

- Implementations of the DSBM and SBTM inference procedures for both a priori (known classes) and a posteriori (estimated classes) block models.
- Demo of DSBM applied to simulated network datasets.
- Demo of DSBM and SBTM applied to dynamic email network constructed from Enron corpus (Priebe et al., 2009).

## Download

1. Download or clone the Git repository.
2. Add the root folder to your MATLAB path.

## Usage

The inputs and outputs for each function are documented in the function header. For most variables, the last dimension in the array denotes the time step. Adjacency matrices are stored as `n x n x T` arrays, where `n` denotes the number of nodes, and `T` denotes the number of discrete time steps. Class memberships are stored in an `n x T` matrix.

### Dynamic Stochastic Block Model (DSBM)

#### A Priori Block Models (Known Classes)

In the a priori block model setting, a sufficient statistic for the DSBM is the time series of block density matrices, which can be obtained using the `calcBlockDens()` function as follows:

`blockDens = calcBlockDens(adj,class)`.

Near-optimal estimates of the edge probability matrices in the maximum a posteriori (MAP) sense can be obtained using the `ekfDsbm()` function, which applies the extended Kalman filter (EKF) on the time series of block density matrices as follows:

`[psi,psiCov] = ekfDsbm(blockDens,stateTrans,transCov,obsCov,initMean,initCov)`

where the output `psi` is the logit of the estimated edge probability matrices at each time step, and `psiCov` contains the covariances of the estimates at each time step. See the manual for additional details, including information on input parameters.

#### A Posteriori Block Models (Estimated Classes)

In the a posteriori block model setting, class memberships are estimated along with the edge probability matrices. The estimates of the class memberships and edge probability matrices can be obtained using the `ekfDsbmLocalSearch()` function, which alternates between applying a local search over the class memberships and applying the EKF on the block densities for the current class membership estimates as follows:

`[class,psi,psiCov] = ekfDsbmLocalSearch(adj,k,stateTrans,transCov,obsCov,initMean,initCov)`

Unlike `ekfDsbm()`, the `ekfDsbmLocalSearch()` function operates directly on the `n x n x T` array of adjacency matrices at each time step. It also requires the desired number of classes `k` to be specified.

### Stochastic Block Transition Model (SBTM)

Unlike the DSBM, the SBTM requires the `n x n x T` array of adjacency matrices in both the a priori and a posteriori settings. In the a priori setting, the estimated edge transition probabilities are obtained using the `ekfSbtm()` function as follows:

`[psi,psiCov] = ekfSbtm(adj,class,stateTrans,transCov,obsCov,initMean,initCov)`.

`psi` now corresponds to estimated transition probabilities rather than edge probabilities (as in the DSBM) and has 2x as many rows as in the DSBM.

In the a posteriori setting, the estimated class memberships and edge transition probabilities are obtained using the `ekfSbtmLocalSearch()` function as follows:

`[class,psi,psiCov] = ekfSbtmLocalSearch(adj,k,stateTrans,transCov,obsCov,initMean,initCov)`.

### Recommended Toolboxes

- Parallel Computing Toolbox: required to conduct local search over SBM classes in parallel for a posteriori block modeling. This toolbox is highly recommended to speed up the computation time for a posteriori block model inference.
- Statistics and Machine Learning Toolbox: required to use spectral clustering to estimate the initial class estimates. If the Statistics and Machine Learning Toolbox is not available, substitute an alternate function for k-means clustering in `spectralClusterSbm()`.

## Additional Information

Please consult the toolbox manual `Manual.pdf` for additional information on usage, including descriptions of optional inputs and outputs and selection of hyperparameters. See also the documentation in the headers of the respective functions (or type `help` followed by the function name, e.g. `help ekfDsbm` in the MATLAB command window) for more details on usage.

## References

Priebe, C. E., Conroy, J. M., Marchette, D. J., & Park, Y. (2009). Scan statistics on Enron graphs. Retrieved from http://cis.jhu.edu/~parky/Enron/enron.html

Xu, K. S. (2015). Stochastic block transition models for dynamic networks. In Proceedings of the 18th International Conference on Artificial Intelligence and Statistics (pp. 1079–1087). Retrieved from http://arxiv.org/abs/1411.5404

Xu, K. S., & Hero III, A. O. (2014). Dynamic stochastic blockmodels for time-evolving social networks. IEEE Journal of Selected Topics in Signal Processing, 8(4), 552–562. Retrieved from http://arxiv.org/abs/1403.0921

## License

Distributed with a BSD license; see `LICENSE.txt`