package com.github.trinity.supermds;

/**
 * Configuration class for controlling the behavior of the inverse MDS multilateration process.
 * <p>
 * This class encapsulates tunable parameters used during the optimization phase of inverse
 * multidimensional scaling (MDS), specifically when using multilateration to reconstruct
 * high-dimensional coordinates from a lower-dimensional embedding.
 * </p>
 *
 * <ul>
 *   <li><b>maxIterations</b>: The maximum number of iterations the optimizer will perform.</li>
 *   <li><b>maxEvaluations</b>: The maximum number of function evaluations allowed during optimization.</li>
 *   <li><b>regularizationLambda</b>: The strength of the L2 regularization term applied to discourage
 *       large reconstructed coordinate values and prevent overfitting to poor local minima.</li>
 *   <li><b>useAnchorCentroidStart</b>: Whether to initialize the optimization starting point as the
 *       centroid of the anchor points.</li>
 * </ul>
 *
 * <p>This configuration provides flexibility for tuning optimization performance and accuracy
 * depending on dataset characteristics and desired trade-offs.</p>
 *
 * @author Sean Phillips
 */
public class MultilaterationConfig {
    /**
     * Maximum number of optimization iterations
     */
    public int maxIterations = 10000;

    /**
     * Maximum number of function evaluations
     */
    public int maxEvaluations = 10000;

    /**
     * Regularization strength (lambda) for L2 penalty
     */
    public double regularizationLambda = 0.1;

    /**
     * Whether to center the initial guess using anchor centroid
     */
    public boolean useAnchorCentroidStart = true;

    public enum OptimizerType {
        LEVENBERG_MARQUARDT,
        GAUSS_NEWTON,
        BOBYQA,
        CMAES
    }

    public OptimizerType optimizer = OptimizerType.LEVENBERG_MARQUARDT;

    // Constructor with defaults
    public MultilaterationConfig() {
    }

    // Constructor for convenience
    public MultilaterationConfig(int maxIters, int maxEvals, double lambda) {
        this.maxIterations = maxIters;
        this.maxEvaluations = maxEvals;
        this.regularizationLambda = lambda;
    }
}
