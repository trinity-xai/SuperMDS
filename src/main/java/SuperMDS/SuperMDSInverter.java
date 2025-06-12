package SuperMDS;
import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresBuilder;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresOptimizer.Optimum;
import org.apache.commons.math3.fitting.leastsquares.LeastSquaresProblem;
import org.apache.commons.math3.fitting.leastsquares.LevenbergMarquardtOptimizer;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

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
     * Approximates the high-dimensional coordinates of a point given its
     * low-dimensional embedding using multilateration with a set of landmark points.
     *
     * @param anchors    The known high-dimensional coordinates of landmark points (shape: k × D)
     * @param embedded   The corresponding low-dimensional coordinates of the same landmarks (shape: k × d)
     * @param z          The low-dimensional point to invert (shape: d)
     * @return           The estimated high-dimensional coordinates of the input point (shape: D)
     */
    public static double[] invertViaMultilateration(
            double[][] anchors,     // high-dimensional landmark positions (reference points)
            double[][] embedded,    // low-dimensional MDS embeddings of the landmarks
            double[] z              // low-dimensional target point to invert
    ) {
        int D = anchors[0].length;  // original high-dimensional space dimension
        int k = anchors.length;     // number of landmark points

        // Step 1: Compute Euclidean distances between the input point z and each low-d landmark
        double[] delta = new double[k];
        for (int i = 0; i < k; i++) {
            delta[i] = euclideanDistance(z, embedded[i]);
        }

        // Step 2: Initialize starting guess for optimization as the centroid of the high-d anchors
        double[] start = new double[D];
        for (double[] anchor : anchors) {
            for (int j = 0; j < D; j++) {
                start[j] += anchor[j] / k;
            }
        }

        // Step 3: Define the objective function: residuals between target distances and estimated distances
        MultivariateVectorFunction value = point -> {
            double[] result = new double[k];
            for (int i = 0; i < k; i++) {
                result[i] = euclideanDistance(point, anchors[i]) - delta[i];
            }
            return result;
        };

        // Step 4 (optional but recommended): Provide a Jacobian matrix for better convergence
        MultivariateMatrixFunction jacobian = point -> {
            double[][] J = new double[k][D];
            for (int i = 0; i < k; i++) {
                double dist = euclideanDistance(point, anchors[i]);
                for (int j = 0; j < D; j++) {
                    J[i][j] = (point[j] - anchors[i][j]) / (dist + 1e-6);  // avoid division by zero
                }
            }
            return new Array2DRowRealMatrix(J).getData();
        };

        // Step 5: Build the least-squares optimization problem
        LeastSquaresProblem problem = new LeastSquaresBuilder()
            .start(start)                      // initial guess
            .model(value, jacobian)            // objective and Jacobian
            .target(new double[k])             // target is zero (residuals)
            .lazyEvaluation(false)
            .maxEvaluations(10000)
            .maxIterations(10000)
            .build();

        // Step 6: Use Levenberg-Marquardt optimizer to solve the non-linear least squares
        LeastSquaresOptimizer optimizer = new LevenbergMarquardtOptimizer();
        Optimum optimum = optimizer.optimize(problem);

        // Step 7: Return the optimized high-dimensional coordinates
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
