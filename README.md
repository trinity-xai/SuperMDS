# SuperMDS

Parallelized implementation of MultiDimensional Scaling algorithms with support for 
weights, landmarks, stress sampling for performance and Out of Sample Extension (OSE).
Provides an approximation inverse transform to go from lower dimensional space up to high dimensional using a Multilateration method. (works but needs work)
Provides a Conditional Variational AutoEncoder (CVAE) parallelized at the minibatch level and designed to be trained on an MDS transform in order to provide inverse transform estimations.
