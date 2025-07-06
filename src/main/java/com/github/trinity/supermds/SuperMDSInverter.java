package com.github.trinity.supermds;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.fitting.leastsquares.GaussNewtonOptimizer;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer.Optimum;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;
import org.apache.commons.math3.fitting.leastsquares.MultivariateJacobianFunction;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.CMAESOptimizer;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.util.Pair;

import java.util.Arrays;

import static com.github.trinity.supermds.SuperMDSHelper.euclideanDistance;


/**
 * SuperMDSInverter provides a method to approximately invert a low-dimensional
 * MDS embedding back to its original high-dimensional space using algorithms such
 * as Multilateration and Pseudo Inverse by Landmark.
 *
 * <p>
 * The inversion is based on estimating the unknown high-dimensional position
 * of a point given its distances to a set of known "anchor" points (landmarks).
 * These distances are inferred from the low-dimensional space produced by
 * MDS or a similar dimensionality reduction technique.
 * </p>
 *
 * <p>
 * This approach is suitable for situations where an explicit inverse mapping is
 * not available, such as with Classical MDS or SMACOF, and provides a geometric
 * alternative to autoencoder-based inversion.
 * </p>
 *
 * @author Sean Phillips
 */
public class SuperMDSInverter {
    /**
     * Inverts low-dimensional embedded points using landmark-based pseudoinverse inversion.
     *
     * @param anchorsHD      Original high-D anchor points (k × D)
     * @param anchorsLD      Corresponding embedded low-D anchor points (k × d)
     * @param embeddedPoints Points to invert (n × d)
     * @param epsilon        Small constant to avoid division by zero
     * @return Approximated high-D points (n × D)
     */
    public static double[][] invertViaPseudoinverse(
        double[][] anchorsHD,
        double[][] anchorsLD,
        double[][] embeddedPoints,
        double epsilon
    ) {
        int k = anchorsHD.length;
        int D = anchorsHD[0].length;
        int d = anchorsLD[0].length;
        int n = embeddedPoints.length;

        // Step 1: Compute interpolation weights W ∈ ℝ^{n × k}
        double[][] W = new double[n][k];

        for (int i = 0; i < n; i++) {
            double[] xi = embeddedPoints[i];
            double weightSum = 0.0;

            for (int j = 0; j < k; j++) {
                double[] anchor = anchorsLD[j];
                double dist = euclideanDistance(xi, anchor);
                double w = 1.0 / (dist + epsilon);  // Shepard weights
                W[i][j] = w;
                weightSum += w;
            }

            // Normalize row to sum to 1
            for (int j = 0; j < k; j++) {
                W[i][j] /= weightSum;
            }
        }

        // Step 2: W (n × k), A_hd (k × D)
        RealMatrix Wmat = MatrixUtils.createRealMatrix(W);
        RealMatrix A_hd = MatrixUtils.createRealMatrix(anchorsHD);

        // Step 3: Y = W * A_hd
        RealMatrix Y = Wmat.multiply(A_hd);

        // Convert back to array
        return Y.getData();
    }

    /**
     * Approximates the high-dimensional coordinates of a point given its low-dimensional embedding
     * using non-linear multilateration based on a set of anchor points and their corresponding embeddings.
     * <p>
     * This method solves an inverse MDS problem by minimizing the discrepancy between:
     * <ul>
     *   <li>the distances from the low-dimensional point to embedded anchors, and</li>
     *   <li>the distances from the reconstructed high-dimensional point to the original high-dimensional anchors</li>
     * </ul>
     * using a configuration specified optimizer. Optional L2 regularization and other solver parameters
     * are provided via the {@link MultilaterationConfig} object.
     * </p>
     *
     * @param anchors       The known high-dimensional coordinates of landmark (anchor) points (shape: k × D)
     * @param embedded      The corresponding low-dimensional MDS embeddings of the same landmarks (shape: k × d)
     * @param pointToInvert The low-dimensional point to invert back into high-dimensional space (shape: d)
     * @param config        Configuration object specifying optimization behavior and regularization
     * @return The estimated high-dimensional coordinates of the input point (shape: D)
     */
    public static double[] invertViaMultilateration(
        double[][] anchors,
        double[][] embedded,
        double[] pointToInvert,
        MultilaterationConfig config) {
        int D = anchors[0].length;  // high-dimensional space dimension
        int k = anchors.length;     // number of landmark points

        //Compute Euclidean distances in low-dimensional space
        double[] delta = new double[k];
        for (int i = 0; i < k; i++) {
            delta[i] = SuperMDSHelper.euclideanDistance(pointToInvert, embedded[i]);
        }

        //Compute the mean of the high-d anchor points (used as prior and init guess)
        double[] mu = new double[D];
        for (double[] anchor : anchors) {
            for (int j = 0; j < D; j++) {
                mu[j] += anchor[j];
            }
        }
        for (int j = 0; j < D; j++) {
            mu[j] /= k;
        }
        double[] start = Arrays.copyOf(mu, D);  // use prior as starting point

        // Choose optimizer backend and optimize
        return optimize(start, anchors, delta, mu, config);
    }

    private static double[] optimize(double[] start, double[][] anchors,
                                     double[] delta, double[] mu, MultilaterationConfig config) {
        int D = anchors[0].length;  // high-dimensional space dimension
        int k = anchors.length;     // number of landmark points
        int dim = start.length;
        LeastSquaresOptimizer optimizer = null;
        switch (config.optimizer) {
            case CMAES:
                CMAESOptimizer cmaes = new CMAESOptimizer(
                    10000, 1e-12, true, 0, 0,
                    new MersenneTwister(),
                    false,
                    new CombinedConvergenceChecker(1e-6, 1e-8, 1e-6, 1e-8)
                );
                //Step 6: Optimize
                PointValuePair resultCMA = cmaes.optimize(
                    new MaxEval(config.maxEvaluations),
                    new ObjectiveFunction(buildScalarObjective(anchors, delta, mu, config.regularizationLambda)),
                    GoalType.MINIMIZE,
                    new InitialGuess(start),
                    SimpleBounds.unbounded(dim),
                    new CMAESOptimizer.Sigma(getSigma(dim, 0.3)),
                    new CMAESOptimizer.PopulationSize(4 * dim)
                );
                //Step 7: Return the optimized high-dimensional coordinates
                return resultCMA.getPoint();

            case BOBYQA:
                BOBYQAOptimizer bobyqa = new BOBYQAOptimizer(dim + 2);  // must be > dim + 1
                //Step 6: Optimize
                PointValuePair resultBOBY = bobyqa.optimize(
                    new MaxEval(config.maxEvaluations),
                    new ObjectiveFunction(buildScalarObjective(anchors, delta, mu, config.regularizationLambda)),
                    GoalType.MINIMIZE,
                    new InitialGuess(start),
                    SimpleBounds.unbounded(dim)
                );
                //Step 7: Return the optimized high-dimensional coordinates
                return resultBOBY.getPoint();

            case GAUSS_NEWTON:
                optimizer = new GaussNewtonOptimizer();
            default:
                optimizer = new LevenbergMarquardtOptimizer();
        }

        // ----- Step 5: Build the regularized least-squares optimization problem -----
        LeastSquaresProblem problem = new LeastSquaresBuilder()
            .start(start)
            .model(buildJacobianObjective(anchors, delta, mu, config.regularizationLambda))
            .target(new double[k + D])   // residual target: 0 for both distance and regularization terms
            .lazyEvaluation(false)
            .maxEvaluations(config.maxEvaluations)
            .maxIterations(config.maxIterations)
            .build();
        //Step 6: Optimize
        Optimum optimum = optimizer.optimize(problem);
        // ----- Step 7: Return the optimized high-dimensional coordinates -----
        return optimum.getPoint().toArray();
    }

    /**
     * Builds a scalar objective function for inverse MDS via multilateration.
     * This function measures how well a candidate high-dimensional point preserves
     * distances to known anchors given low-dimensional distances, with optional L2 regularization.
     *
     * @param anchors High-dimensional coordinates of the anchors (k × D)
     * @param delta   Low-dimensional distances from the point to embedded anchors (length k)
     * @param mu      Prior (mean) position of anchors used for regularization (length D)
     * @param lambda  L2 regularization strength
     * @return MultivariateFunction representing the scalar objective to minimize
     */
    public static MultivariateFunction buildScalarObjective(
        double[][] anchors,
        double[] delta,
        double[] mu,
        double lambda
    ) {
        int k = anchors.length;
        int D = anchors[0].length;

        return point -> {
            double error = 0.0;

            // Distance residuals
            for (int i = 0; i < k; i++) {
                double dist = SuperMDSHelper.euclideanDistance(point, anchors[i]);
                double diff = dist - delta[i];
                error += diff * diff;
            }

            // L2 regularization toward mean (optional)
            if (lambda > 0.0) {
                for (int j = 0; j < D; j++) {
                    double diff = point[j] - mu[j];
                    error += lambda * diff * diff;
                }
            }

            return error;
        };
    }

    /**
     * Builds a MultivariateJacobianFunction for inverse MDS multilateration using anchor distances
     * and optional L2 regularization toward the anchor mean.
     *
     * @param anchors High-dimensional coordinates of anchor points (shape: k × D)
     * @param delta   Low-dimensional distances from the point to embedded anchors (length k)
     * @param mu      Prior mean location in high-dimensional space (length D)
     * @param lambda  Regularization strength
     * @return MultivariateJacobianFunction to be passed to a least squares optimizer
     */
    public static MultivariateJacobianFunction buildJacobianObjective(
        double[][] anchors,
        double[] delta,
        double[] mu,
        double lambda
    ) {
        int k = anchors.length;
        int D = anchors[0].length;

        return (RealVector input) -> {
            double[] x = input.toArray();

            double[] residuals = new double[k + D];
            double[][] jacobian = new double[k + D][D];

            // --- Distance residuals (first k entries) ---
            for (int i = 0; i < k; i++) {
                double[] anchor = anchors[i];
                double dist = SuperMDSHelper.euclideanDistance(x, anchor);
                double epsilon = 1e-8;
                double safeDist = Math.max(dist, epsilon);

                residuals[i] = dist - delta[i];

                for (int j = 0; j < D; j++) {
                    jacobian[i][j] = (x[j] - anchor[j]) / safeDist;
                }
            }

            // --- Regularization residuals (last D entries) ---
            if (lambda > 0.0) {
                double scale = 1.0;  // You may adjust this based on k/D if needed
                for (int j = 0; j < D; j++) {
                    residuals[k + j] = lambda * scale * (x[j] - mu[j]);
                    jacobian[k + j][j] = lambda * scale;
                }
            }

            return new Pair<>(
                new ArrayRealVector(residuals),
                new Array2DRowRealMatrix(jacobian)
            );
        };
    }

    public static double[] computeInitialGuess(
        double[][] anchors,        // High-D anchor vectors (k × D)
        double[][] embedded,       // Low-D anchor embeddings (k × d)
        double[] pointToInvert     // Low-D point to invert (length d)
    ) {
        int k = anchors.length;
        int D = anchors[0].length;

        double[] weights = new double[k];
        double weightSum = 0.0;

        // Compute inverse-distance weights
        for (int i = 0; i < k; i++) {
            double dist = SuperMDSHelper.euclideanDistance(embedded[i], pointToInvert);
            weights[i] = 1.0 / (dist + 1e-6);  // Avoid div-by-zero
            weightSum += weights[i];
        }

        // Normalize weights
        for (int i = 0; i < k; i++) {
            weights[i] /= weightSum;
        }

        // Weighted sum of anchors
        double[] guess = new double[D];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < D; j++) {
                guess[j] += weights[i] * anchors[i][j];
            }
        }

        return guess;
    }

    private static double[] getSigma(int dim, double scale) {
        double[] sigma = new double[dim];
        Arrays.fill(sigma, scale);
        return sigma;
    }
}
