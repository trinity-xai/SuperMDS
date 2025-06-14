package SuperMDS;
import java.util.Arrays;
import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer.Optimum;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;
import org.apache.commons.math3.fitting.leastsquares.MultivariateJacobianFunction;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.Pair;

/**
 * SuperMDSInverter provides a method to approximately invert a low-dimensional
 * MDS embedding back to its original high-dimensional space using multilateration.
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
 *  @author Sean Phillips
 */
public class SuperMDSInverter {

    /**
     * Approximates the high-dimensional coordinates of a point given its low-dimensional embedding
     * using non-linear multilateration based on a set of anchor points and their corresponding embeddings.
     * <p>
     * This method solves an inverse MDS problem by minimizing the discrepancy between:
     * <ul>
     *   <li>the distances from the low-dimensional point to embedded anchors, and</li>
     *   <li>the distances from the reconstructed high-dimensional point to the original high-dimensional anchors</li>
     * </ul>
     * using a Levenberg–Marquardt optimizer. Optional L2 regularization and other solver parameters
     * are provided via the {@link MultilaterationConfig} object.
     * </p>
     *
     * @param anchors         The known high-dimensional coordinates of landmark (anchor) points (shape: k × D)
     * @param embedded        The corresponding low-dimensional MDS embeddings of the same landmarks (shape: k × d)
     * @param pointToInvert   The low-dimensional point to invert back into high-dimensional space (shape: d)
     * @param config          Configuration object specifying optimization behavior and regularization
     * @return                The estimated high-dimensional coordinates of the input point (shape: D)
     */
    public static double[] invertViaMultilateration(
        double[][] anchors,
        double[][] embedded,
        double[] pointToInvert,
        MultilaterationConfig config ) {
        int D = anchors[0].length;  // high-dimensional space dimension
        int k = anchors.length;     // number of landmark points

        // ----- Step 1: Compute Euclidean distances in low-dimensional space -----
        double[] delta = new double[k];
        for (int i = 0; i < k; i++) {
            delta[i] = euclideanDistance(pointToInvert, embedded[i]);
        }

        // ----- Step 2: Compute the mean of the high-d anchor points (used as prior and init guess) -----
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

        // ----- Step 3: Regularization strength (tune this parameter) -----
        //double lambda = 0.1;
        double lambda = config.regularizationLambda;

        // ----- Step 4: Define value + Jacobian in a unified model -----
        MultivariateJacobianFunction model = (RealVector point) -> {
            double[] x = point.toArray();
            double[] residuals = new double[k + D];      // k distance residuals + D regularization terms
            double[][] jacobian = new double[k + D][D];  // Jacobian of size (k+D) × D
            
            // --- Distance residuals: difference between estimated and target distances ---
            for (int i = 0; i < k; i++) {
                double dist = euclideanDistance(x, anchors[i]);
                residuals[i] = dist - delta[i];
                for (int j = 0; j < D; j++) {
                    jacobian[i][j] = (x[j] - anchors[i][j]) / (dist + 1e-8);  // numerical stability
                }
            }
            
            // --- Regularization residuals: λ * (x - μ) ---
            double distScale = 1.0;
            double regScale = Math.sqrt((double) k / D); // optional scaling            
            for (int j = 0; j < D; j++) {
                residuals[k + j] = lambda * (x[j] - mu[j]);
                for (int l = 0; l < D; l++) {
                    //jacobian[k + j][l] = (l == j) ? lambda : 0.0;
                    residuals[k + j] = lambda * regScale * (x[j] - mu[j]);
                    jacobian[k + j][l] = (l == j) ? lambda * regScale : 0.0;
                }
            }
            
            return new Pair<>(new ArrayRealVector(residuals), new Array2DRowRealMatrix(jacobian));
        };

        // ----- Step 5: Build the regularized least-squares optimization problem -----
        LeastSquaresProblem problem = new LeastSquaresBuilder()
                .start(start)
                .model(model)
                .target(new double[k + D])   // residual target: 0 for both distance and regularization terms
                .lazyEvaluation(false)
                .maxEvaluations(10000)
                .maxIterations(10000)
                .build();

        // ----- Step 6: Optimize using Levenberg-Marquardt -----
        LeastSquaresOptimizer optimizer = new LevenbergMarquardtOptimizer();
        Optimum optimum = optimizer.optimize(problem);

        // ----- Step 7: Return the optimized high-dimensional coordinates -----
        return optimum.getPoint().toArray();
    }

    
    private static double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }
}
