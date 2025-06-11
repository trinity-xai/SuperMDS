package SuperMDS;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.IntStream;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

public class SuperMDS {

    public static double EPS_TOLERANCE =  1e-10;
    public enum Mode {
        CLASSICAL, //canonical implemenation of MDS using Eigenvalue Decomp
        METRIC, //SMACOF gradient descent stress minimization implementation
        PARALLEL, //Parallelized version of SMACOF with weights and stress sampling support
        NONMETRIC, //@TODO SMPuses default smacof as placeholder
        SUPERVISED, //Weighted version of basic SMACOF, serially processed
        LANDMARK, //Basic Landmark MDS
        APPROXIMATE //Approximated Landmark MDS via Nystrom's approximation
    }

    public static class Params {

        public Mode mode = Mode.CLASSICAL;
        public int outputDim = 2;
        public boolean useSMACOF = false;
        public boolean useSupervision = false;
        public boolean useParallel = false;
        public boolean useStressSampling = false;
        public int stressSampleCount = 1000;
        public boolean autoSymmetrize = false;
        public boolean useKMeansForLandmarks = false;
        public int numLandmarks = 50;
        public int maxIterations = 300;
        public int oseMaxIterations = 300;
        public double tolerance = 1e-6;
        public double oseTolerance = 1e-6;
        public int randomSeed = 42;
        public double[][] weights = null;
        public int[] classLabels = null;
        public double alpha = 1.0;
    }

    public static double[][] runMDS(double[][] data, Params params) {
        if (params.autoSymmetrize) {
            data = ensureSymmetricDistanceMatrix(data);
        }

        switch (params.mode) {
            case CLASSICAL:
                return classicalMDS(data, params.outputDim);
            case METRIC:
                return smacofMDS(data, params.outputDim, params.maxIterations, params.tolerance, params.weights,
                        true, params.useStressSampling, params.stressSampleCount);
            case NONMETRIC:
                return smacofMDS(data, params.outputDim, params.maxIterations, params.tolerance, params.weights,
                        true, params.useStressSampling, params.stressSampleCount);
            case SUPERVISED:
                return supervisedSMACOFMDS(data, params.outputDim, params.classLabels, params.alpha);
            case PARALLEL:
                return smacofMDSParallel(data, params.outputDim, params.maxIterations, params.tolerance,
                        true, params.useStressSampling, params.stressSampleCount);
            case LANDMARK:
                return landmarkMDS(data, params.outputDim, params.numLandmarks, params.useKMeansForLandmarks, params.randomSeed);
            case APPROXIMATE:
                return approximateMDSViaLandmarks(data, params.outputDim, params.numLandmarks, params.useKMeansForLandmarks, params.randomSeed);
            default:
                throw new IllegalArgumentException("Unsupported mode");
        }
    }

    /**
     * Classical Multidimensional Scaling (MDS).
     * Computes a low-dimensional embedding from a full squared distance matrix using eigendecomposition.
     *
     * @param D   A symmetric n × n matrix of squared Euclidean distances between points.
     * @param dim The target number of dimensions for the embedding.
     * @return A matrix of shape [n][dim] with the low-dimensional coordinates.
     */
    public static double[][] classicalMDS(double[][] D, int dim) {
       int n = D.length;
       double[][] B = new double[n][n];

       // Compute row and column means
       double[] rowMeans = new double[n];
       double[] colMeans = new double[n];
       double totalMean = 0;

       for (int i = 0; i < n; i++) {
           for (int j = 0; j < n; j++) {
               rowMeans[i] += D[i][j];
               colMeans[j] += D[i][j];
               totalMean += D[i][j];
           }
       }
       for (int i = 0; i < n; i++) {
           rowMeans[i] /= n;
           colMeans[i] /= n;
       }
       totalMean /= (n * n);

       // Compute B (double-centered matrix)
       for (int i = 0; i < n; i++) {
           for (int j = 0; j < n; j++) {
               B[i][j] = -0.5 * (D[i][j] - rowMeans[i] - colMeans[j] + totalMean);
           }
       }

       // Eigen-decomposition
       RealMatrix Bmat = MatrixUtils.createRealMatrix(B);
       EigenDecomposition eig = new EigenDecomposition(Bmat);
       double[][] X = new double[n][dim];
       for (int i = 0; i < dim; i++) {
           double eigVal = eig.getRealEigenvalue(i);
           if (eigVal < 0) continue;
           double scale = Math.sqrt(eigVal);
           RealVector eigVec = eig.getEigenvector(i);
           for (int j = 0; j < n; j++) {
               X[j][i] = eigVec.getEntry(j) * scale;
           }
       }
       return X;
   }

    public static double[][] smacofMDS(double[][] D, int dim, int maxIter, double tolerance,
            double[][] weights, boolean earlyExitOnStressIncrease, boolean useStressSampling, int stressSampleSize) {
        int n = D.length;
        Random rand = new Random(42);
        double[][] X = new double[n][dim];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                X[i][j] = rand.nextGaussian();
            }
        }

        double prevStress = Double.MAX_VALUE;
        for (int iter = 0; iter < maxIter; iter++) {
            double[][] dist = SuperMDSHelper.pairwiseDistances(X);
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < dim; k++) {
                    double num = 0, denom = 0;
                    for (int j = 0; j < n; j++) {
                        if (i == j) {
                            continue;
                        }
                        double w = weights != null ? weights[i][j] : 1.0;
                        double d = dist[i][j];
                        if (d > 0) {
                            double delta = D[i][j] / d;
                            num += w * (X[j][k] + delta * (X[i][k] - X[j][k]));
                            denom += w;
                        }
                    }
                    if (denom > 0) {
                        X[i][k] = num / denom;
                    }
                }
            }
            // Compute stress
            double stress = useStressSampling
                    ? computeSampledStress(D, X, stressSampleSize)
                    : computeFullStressParallelWeighted(D, X, weights);

            if (Math.abs(prevStress - stress) < tolerance) {
                break;
            }
            if (stress > prevStress && earlyExitOnStressIncrease) {
                break;
            }

            prevStress = stress;
        }
        return X;
    }

    private static double computeFullStress(double[][] D, double[][] X) {
        int n = D.length;
        double stress = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dij = SuperMDSHelper.euclideanDistance(X[i], X[j]);
                double diff = D[i][j] - dij;
                stress += diff * diff;
            }
        }
        return stress;
    }

    public static double computeFullStressParallelWeighted(double[][] D, double[][] X, double[][] W) {
        int n = D.length;
        return IntStream.range(0, n).parallel().mapToDouble(i -> {
            double localStress = 0.0;
            for (int j = i + 1; j < n; j++) {
                double dij = SuperMDSHelper.euclideanDistance(X[i], X[j]);
                double diff = D[i][j] - dij;
                double weight = W != null ? W[i][j] : 1.0;
                localStress += weight * diff * diff;
            }
            return localStress;
        }).sum();
    }

    public static double computeSampledStress(double[][] D, double[][] X, int sampleSize) {
        int n = D.length;
        Random rand = new Random(42);
        Set<Long> seen = new HashSet<>();
        List<int[]> pairs = new ArrayList<>();

        while (pairs.size() < sampleSize) {
            int i = rand.nextInt(n);
            int j = rand.nextInt(n);
            if (i == j) {
                continue;
            }
            long key = ((long) Math.min(i, j) << 32) | Math.max(i, j);
            if (seen.add(key)) {
                pairs.add(new int[]{i, j});
            }
        }

        return pairs.parallelStream().mapToDouble(pair -> {
            int i = pair[0], j = pair[1];
            double dij = SuperMDSHelper.euclideanDistance(X[i], X[j]);
            double diff = D[i][j] - dij;
            return diff * diff;
        }).sum();
    }

    /**
     * Performs parallelized SMACOF (Scaling by Majorizing a Complicated Function) MDS to reduce a distance matrix
     * to a low-dimensional Euclidean space.
     *
     * <p>This implementation supports optional early exit on stress increase, stress sampling to accelerate convergence,
     * and full parallelization of the majorization step.</p>
     *
     * @param D                         A symmetric n × n dissimilarity (distance) matrix.
     * @param dim                       Target embedding dimension (e.g., 2 or 3).
     * @param maxIter                   Maximum number of iterations to perform.
     * @param tol                       Convergence tolerance for stress delta.
     * @param earlyExitOnStressIncrease If true, stops if stress increases between iterations.
     * @param useStressSampling         If true, uses a subset of pairs for approximate stress evaluation.
     * @param stressSampleSize          Number of samples to use if stress sampling is enabled.
     * @return                          An n × dim matrix representing the embedding of the points.
     * @throws IllegalArgumentException If the input is not a square matrix.
     */
    public static double[][] smacofMDSParallel(
            double[][] D, int dim, int maxIter, double tol,
            boolean earlyExitOnStressIncrease, boolean useStressSampling, int stressSampleSize) {

        int n = D.length;

        // Initialize positions randomly from a normal distribution
        Random rand = new Random();
        double[][] X = new double[n][dim];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                X[i][j] = rand.nextGaussian();
            }
        }

        double prevStress = Double.MAX_VALUE;

        // Main SMACOF iteration loop
        for (int iter = 0; iter < maxIter; iter++) {

            // Compute pairwise distances between points in current configuration
            double[][] dist = SuperMDSHelper.pairwiseDistances(X);

            // Allocate array for updated configuration
            double[][] newX = new double[n][dim];

            // Make a deep copy of current configuration for use inside parallel lambda
            double[][] tempX = SuperMDSHelper.deepCopyParallel(X);

            // Perform the majorization update step in parallel
            IntStream.range(0, n).parallel().forEach(i -> {
                for (int k = 0; k < dim; k++) {
                    double num = 0, denom = 0;

                    // Update position of point i using all other points j
                    for (int j = 0; j < n; j++) {
                        if (i == j) continue;

                        double d = dist[i][j];
                        if (d > 0) {
                            // Majorization step: weighted average based on target vs current distances
                            double delta = D[i][j] / d;
                            num += tempX[j][k] + delta * (tempX[i][k] - tempX[j][k]);
                            denom++;
                        }
                    }

                    // Normalize by denominator to complete update
                    if (denom > 0) {
                        newX[i][k] = num / denom;
                    }
                }
            });

            // Compute the stress value after the update (optionally approximate)
            double stress = useStressSampling
                    ? computeSampledStress(D, newX, stressSampleSize)
                    : computeFullStressParallelWeighted(D, newX, null); // null weights = uniform

            // Check convergence condition
            if (Math.abs(prevStress - stress) < tol) {
                break;
            }

            // Optional early stopping if stress increases
            if (stress > prevStress && earlyExitOnStressIncrease) {
                break;
            }

            // Prepare for next iteration
            prevStress = stress;
            X = newX;
        }

        return X;
    }

    public static double[][] supervisedSMACOFMDS(double[][] D, int dim, int[] labels, double alpha) {
        int n = D.length;
        double[][] weights = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                weights[i][j] = (labels[i] == labels[j]) ? alpha : 1.0;
            }
        }

        return smacofMDS(D, dim, 300, 1e-6, weights, true, false, 1000);
    }

    /**
     * Performs Landmark Multidimensional Scaling (Landmark MDS) to embed high-dimensional data into a lower-dimensional space.
     * This method selects a subset of the data points (landmarks), computes their classical MDS embedding via eigendecomposition,
     * and then extends the embedding to all other points using the Nyström approximation.
     *
     * <p>This implementation assumes raw input data in Euclidean space, not a precomputed distance matrix. It supports
     * either uniform random landmark selection or KMeans++ centroids as landmarks.</p>
     *
     * @param data                 A 2D array of shape (n, d), where {@code n} is the number of data points and {@code d} is the input dimensionality.
     *                             The method expects raw input vectors, not a distance or similarity matrix.
     * @param dim                  The number of output dimensions for the embedding (e.g., 2 or 3).
     * @param numLandmarks         The number of landmark points to select for the Nyström approximation.
     * @param useKMeansPlusPlus    If {@code true}, landmark points are selected using KMeans++ clustering;
     *                             if {@code false}, landmarks are chosen randomly.
     * @param seed                 The random seed for reproducible landmark selection.
     *
     * @return A 2D array of shape (n, dim) containing the low-dimensional embedding for all input points.
     *
     * @throws IllegalArgumentException if the input is a square symmetric matrix, which suggests a distance matrix was passed instead of raw vectors.
     */    
    public static double[][] landmarkMDS(double[][] data, int dim, int numLandmarks, boolean useKMeansPlusPlus, int seed) {
        if (data.length == data[0].length && MatrixUtils.isSymmetric(
            MatrixUtils.createRealMatrix(data), EPS_TOLERANCE)) {
            throw new IllegalArgumentException("Expected raw high-dimensional input vectors, not a distance matrix.");
        }
        int n = data.length;
        int d = data[0].length;

        // Select landmark points
        double[][] landmarks;
        int[] landmarkIndices = new int[numLandmarks];
        if (useKMeansPlusPlus) {
            KMeansPlusPlus kpp = new KMeansPlusPlus(data, numLandmarks, seed);
            landmarks = kpp.getCentroids();
            // KMeans++ doesn't return indices; set to -1 as placeholder
            for (int i = 0; i < numLandmarks; i++) {
                landmarkIndices[i] = -1;
            }
        } else {
            java.util.Random rand = new java.util.Random(seed);
            java.util.Set<Integer> indices = new java.util.HashSet<>();
            while (indices.size() < numLandmarks) {
                indices.add(rand.nextInt(n));
            }
            landmarks = new double[numLandmarks][d];
            int idx = 0;
            for (int i : indices) {
                landmarks[idx] = data[i];
                landmarkIndices[idx] = i;
                idx++;
            }
            java.util.Arrays.sort(landmarkIndices); // Used for binarySearch below
        }

        // Compute squared distances between all landmark pairs
        double[][] squaredDistancesForLandmarks = new double[numLandmarks][numLandmarks];
        for (int i = 0; i < numLandmarks; i++) {
            for (int j = i; j < numLandmarks; j++) {
                squaredDistancesForLandmarks[i][j] = squaredDistancesForLandmarks[j][i] =
                    SuperMDSHelper.squaredEuclideanDistance(landmarks[i], landmarks[j]);
            }
        }

        // Double-center squaredDistancesForLandmarks to form the Gram matrix G
        double[] rowMeans = new double[numLandmarks];
        double[] colMeans = new double[numLandmarks];
        double totalMean = 0;
        for (int i = 0; i < numLandmarks; i++) {
            for (int j = 0; j < numLandmarks; j++) {
                rowMeans[i] += squaredDistancesForLandmarks[i][j];
                colMeans[j] += squaredDistancesForLandmarks[i][j];
            }
        }
        for (int i = 0; i < numLandmarks; i++) {
            rowMeans[i] /= numLandmarks;
            colMeans[i] /= numLandmarks;
            totalMean += rowMeans[i];
        }
        totalMean /= numLandmarks;
        double[][] gramMatrix = new double[numLandmarks][numLandmarks];
        for (int i = 0; i < numLandmarks; i++) {
            for (int j = 0; j < numLandmarks; j++) {
                gramMatrix[i][j] = -0.5 * (squaredDistancesForLandmarks[i][j] - rowMeans[i] - colMeans[j] + totalMean);
            }
        }

        // Eigen-decomposition of Gram matrix
        RealMatrix realGramMatrix = new Array2DRowRealMatrix(gramMatrix);
        EigenDecomposition eig = new EigenDecomposition(realGramMatrix);
        double[][] landmarkEmbedding = new double[numLandmarks][dim];
        for (int i = 0; i < dim; i++) {
            double eigValSqrt = FastMath.sqrt(FastMath.max(eig.getRealEigenvalue(i), 0));
            double[] eigVec = eig.getEigenvector(i).toArray();
            for (int j = 0; j < numLandmarks; j++) {
                landmarkEmbedding[j][i] = eigVec[j] * eigValSqrt;
            }
        }

        // Nystrom extension for all data points
        double[][] embedding = new double[n][dim];
        for (int i = 0; i < n; i++) {
            int landmarkIndex = (useKMeansPlusPlus) ? -1 : java.util.Arrays.binarySearch(landmarkIndices, i);
            if (landmarkIndex >= 0) {
                // Copy precomputed embedding
                System.arraycopy(landmarkEmbedding[landmarkIndex], 0, embedding[i], 0, dim);
            } else {
                // Interpolate using Nystrom approximation
                double[] dist = new double[numLandmarks];
                double meanDist = 0;
                for (int j = 0; j < numLandmarks; j++) {
                    dist[j] = SuperMDSHelper.squaredEuclideanDistance(data[i], landmarks[j]);
                    meanDist += dist[j];
                }
                meanDist /= numLandmarks;
                for (int k = 0; k < dim; k++) {
                    double eigVal = eig.getRealEigenvalue(k);
                    if (FastMath.abs(eigVal) < 1e-10) continue;
                    double coord = 0;
                    for (int j = 0; j < numLandmarks; j++) {
                        double centeredDist = -0.5 * (dist[j] - meanDist);
                        coord += centeredDist * (landmarkEmbedding[j][k] / eigVal);
                    }
                    embedding[i][k] = coord;
                }
            }
        }

        return embedding;
    }

    /**
     * Approximates Classical MDS using a Nyström extension based on a subset of landmark points.
     * This method embeds high-dimensional data into a lower-dimensional Euclidean space by:
     *  - Selecting landmark points (either randomly or via KMeans++)
     *  - Computing a Gram matrix from pairwise squared distances between landmarks
     *  - Performing eigendecomposition of the Gram matrix
     *  - Projecting all data points into the embedding space using the Nyström method
     *
     * @param data High-dimensional input data matrix (n × d)
     * @param dim Target output dimensionality (typically 2 or 3)
     * @param numLandmarks Number of landmark points to use (k << n)
     * @param useKMeansPlusPlus If true, selects landmarks using KMeans++, otherwise randomly
     * @param seed Random seed for reproducibility
     * @return Low-dimensional embedding of shape (n × dim)
     */
    public static double[][] approximateMDSViaLandmarks(double[][] data, int dim, int numLandmarks, boolean useKMeansPlusPlus, int seed) {
        int n = data.length;
        int d = data[0].length;
        double[][] landmarks;

        // 1. Landmark selection
        if (useKMeansPlusPlus) {
            KMeansPlusPlus kpp = new KMeansPlusPlus(data, numLandmarks, seed);
            landmarks = kpp.getCentroids();
        } else {
            Random rand = new Random(seed);
            Set<Integer> indices = new HashSet<>();
            while (indices.size() < numLandmarks) {
                indices.add(rand.nextInt(n));
            }
            landmarks = new double[numLandmarks][d];
            int idx = 0;
            for (int i : indices) {
                landmarks[idx++] = data[i];
            }
        }

        // 2. Compute squared distances: D_all (n x k) and D_landmarks (k x k)
        double[][] D_all = new double[n][numLandmarks];
        double[][] D_landmarks = new double[numLandmarks][numLandmarks];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < numLandmarks; j++) {
                D_all[i][j] = SuperMDSHelper.squaredEuclideanDistance(data[i], landmarks[j]);
            }
        }
        for (int i = 0; i < numLandmarks; i++) {
            for (int j = i; j < numLandmarks; j++) {
                double dist = SuperMDSHelper.squaredEuclideanDistance(landmarks[i], landmarks[j]);
                D_landmarks[i][j] = D_landmarks[j][i] = dist;
            }
        }

        // 3. Double center the landmark distance matrix to get the Gram matrix G_k
        double[][] G_k = new double[numLandmarks][numLandmarks];
        double[] rowMeans = new double[numLandmarks];
        double[] colMeans = new double[numLandmarks];
        double totalMean = 0;

        for (int i = 0; i < numLandmarks; i++) {
            for (int j = 0; j < numLandmarks; j++) {
                rowMeans[i] += D_landmarks[i][j];
                colMeans[j] += D_landmarks[i][j];
            }
        }
        for (int i = 0; i < numLandmarks; i++) {
            rowMeans[i] /= numLandmarks;
            colMeans[i] /= numLandmarks;
            totalMean += rowMeans[i];
        }
        totalMean /= numLandmarks;

        for (int i = 0; i < numLandmarks; i++) {
            for (int j = 0; j < numLandmarks; j++) {
                G_k[i][j] = -0.5 * (D_landmarks[i][j] - rowMeans[i] - colMeans[j] + totalMean);
            }
        }

        // 4. Eigendecomposition of G_k
        RealMatrix GkMat = new Array2DRowRealMatrix(G_k);
        EigenDecomposition eig = new EigenDecomposition(GkMat);

        double[][] V = new double[numLandmarks][dim];
        double[] sqrtEigenvalues = new double[dim];

        for (int i = 0; i < dim; i++) {
            sqrtEigenvalues[i] = Math.sqrt(Math.max(eig.getRealEigenvalue(i), 0));
            double[] eigVec = eig.getEigenvector(i).toArray();
            for (int j = 0; j < numLandmarks; j++) {
                V[j][i] = eigVec[j];
            }
        }

        // 5. Double-center D_all to get B (n x k)
        double[][] B = new double[n][numLandmarks];
        double[] rowMeansAll = Arrays.stream(D_all).mapToDouble(r -> Arrays.stream(r).average().orElse(0)).toArray();
        double[] colMeansAll = new double[numLandmarks];
        for (int j = 0; j < numLandmarks; j++) {
            for (int i = 0; i < n; i++) {
                colMeansAll[j] += D_all[i][j];
            }
            colMeansAll[j] /= n;
        }
        double totalMeanAll = Arrays.stream(rowMeansAll).average().orElse(0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < numLandmarks; j++) {
                B[i][j] = -0.5 * (D_all[i][j] - rowMeansAll[i] - colMeansAll[j] + totalMeanAll);
            }
        }

        // 6. Nyström projection: X = B * V * Λ^(-1/2)
        double[][] embedding = new double[n][dim];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < dim; k++) {
                double coord = 0;
                double sqrtEigVal = sqrtEigenvalues[k];
                if (sqrtEigVal < 1e-10) continue;
                for (int j = 0; j < numLandmarks; j++) {
                    coord += B[i][j] * V[j][k] / sqrtEigVal;
                }
                embedding[i][k] = coord;
            }
        }

        return embedding;
    }


    /**
     * Embeds a new data point into an existing MDS embedding using the
     * Out-of-Sample Extension (OSE) method with parallelized coordinate
     * updates. This method iteratively adjusts the position of the new point to
     * minimize stress with respect to the existing embedded training points.
     *
     * @param embeddings The coordinates of the n already embedded training
     * points (size n x dim).
     * @param distancesToNew The dissimilarities (distances) from the new point
     * to each of the n training points (size n).
     * @param weightsToNewPoint Optional weights (importance) for each
     * dissimilarity (size n). Can be null for uniform weighting.
     * @param params SMACOF and OSE configuration parameters including max
     * iterations and convergence tolerance.
     * @return The coordinates of the embedded new point in the target
     * low-dimensional space (size dim).
     */
    public static double[] embedPointOSEParallel(
            double[][] embeddings,
            double[] distancesToNew,
            double[] weightsToNewPoint,
            Params params
    ) {
        int n = embeddings.length;          // Number of training points
        int dim = embeddings[0].length;     // Target embedding dimension
        int maxIter = params.maxIterations; // Max iterations for OSE

        // Initialize new point randomly in the embedding space
        double[] originalCoordinates = new double[dim];
        Random rand = new Random();
        for (int i = 0; i < dim; i++) {
            originalCoordinates[i] = rand.nextGaussian();
        }

        // Iteratively update the position of the new point
        for (int iter = 0; iter < maxIter; iter++) {
            final double[] ySnapshot = originalCoordinates.clone(); // Freeze current coords for consistent reads

            // Update each dimension of the new point in parallel
            double[] newCoordinates = IntStream.range(0, dim).parallel().mapToDouble(k -> {
                double num = 0.0;
                double denom = 0.0;

                // Accumulate contributions from all training points
                for (int i = 0; i < n; i++) {
                    double d = SuperMDSHelper.euclideanDistance(ySnapshot, embeddings[i]);
                    if (d > 1e-9) { // Avoid division by zero
                        double delta = distancesToNew[i] / d;
                        double weight = (weightsToNewPoint != null) ? weightsToNewPoint[i] : 1.0;

                        // SMACOF update term
                        num += weight * (embeddings[i][k] + delta * (ySnapshot[k] - embeddings[i][k]));
                        denom += weight;
                    }
                }

                // Compute new coordinate if weights are non-zero
                return (denom > 0) ? num / denom : ySnapshot[k];
            }).toArray();

            // Check for convergence in stress
            double stressChange = stressConvergenceParallel(embeddings, originalCoordinates, newCoordinates, distancesToNew);
            if (stressChange < params.oseTolerance) {
                break; // Converged
            }

            // Update for next iteration
            originalCoordinates = newCoordinates;
        }

        return originalCoordinates; // Final embedded coordinates
    }

    /**
     * Computes the change in stress between two embeddings for a single point using parallelized evaluation.
     * <p>
     * This method is used in the out-of-sample embedding (OSE) context, where a new point is inserted into
     * an existing MDS embedding. It calculates how much the embedding stress changes when the point's coordinates
     * are updated from {@code embedded} to {@code newCoords}, given the distances from the new point to all existing points.
     * <p>
     * The stress is defined as the sum of squared differences between the input distances and the corresponding
     * Euclidean distances in the low-dimensional embedding. This function performs the comparison in parallel
     * for performance across all existing points.
     *
     * @param X The existing embedded points (n × dim)
     * @param embedded The current coordinates of the new point (length = dim)
     * @param newCoords The proposed updated coordinates of the new point (length = dim)
     * @param distsToExisting The original distances from the new point to each existing point (length = n)
     * @return The absolute change in stress between the old and new coordinates
     */
    public static double stressConvergenceParallel(double[][] X,
            double[] embedded, double[] newCoords, double[] distsToExisting) {
        int n = X.length;         // Number of existing points
        int dim = X[0].length;    // Dimensionality of the embedding space

        // Thread-safe accumulators for new and old stress values
        DoubleAdder stress = new DoubleAdder();
        DoubleAdder oldStress = new DoubleAdder();

        // Clone coordinate arrays to safely use inside parallel lambda
        double[] finalNewCoords = newCoords.clone();     // Proposed updated coordinates for the new point
        double[] finalOldCoords = embedded.clone();      // Current coordinates of the new point

        // Parallel loop over each existing point in the dataset
        IntStream.range(0, n).parallel().forEach(i -> {
            double d1 = 0, d2 = 0;

            // Compute squared Euclidean distance from point i to newCoords and embedded
            for (int d = 0; d < dim; d++) {
                double diff1 = finalNewCoords[d] - X[i][d];   // Difference along dimension d (new position)
                double diff2 = finalOldCoords[d] - X[i][d];   // Difference along dimension d (old position)
                d1 += diff1 * diff1;                          // Accumulate squared distance (new position)
                d2 += diff2 * diff2;                          // Accumulate squared distance (old position)
            }

            // Compute Euclidean distances from new and old embeddings
            double dist1 = Math.sqrt(d1);  // Distance from newCoords to point i
            double dist2 = Math.sqrt(d2);  // Distance from embedded to point i

            // True dissimilarity from the original high-dimensional space
            double delta = distsToExisting[i];

            // Accumulate squared stress error for the new and old coordinates
            // stress = Σ (delta - distance)^2 over all i
            stress.add((delta - dist1) * (delta - dist1));
            oldStress.add((delta - dist2) * (delta - dist2));
        });

        // Return the absolute change in total stress
        return Math.abs(oldStress.sum() - stress.sum());
        
    }

    /**
     * Ensures that a given matrix is a valid symmetric distance matrix.
     * <p>
     * This method handles two cases:
     * <ul>
     *   <li><b>Raw Data Matrix (non-square):</b> If the input matrix has dimensions n × d (i.e., not square),
     *   it is treated as a dataset of n d-dimensional points. The method computes the full symmetric
     *   pairwise Euclidean distance matrix of size n × n.</li>
     *   <li><b>Square Matrix:</b> If the input matrix is already square (n × n), it is assumed to be a
     *   distance matrix. This case symmetrizes the matrix by averaging each pair (i,j) and (j,i)
     *   to enforce symmetry.</li>
     * </ul>
     * The computation is parallelized for efficiency.
     *
     * @param input A 2D array representing either raw data (n × d) or a distance matrix (n × n).
     * @return A symmetric n × n distance matrix.
     * @throws IllegalArgumentException if the input is null or empty.
     */    
    public static double[][] ensureSymmetricDistanceMatrix(double[][] input) {
        int rows = input.length;
        int cols = input[0].length;

        // Case 1: Raw data matrix (non-square)
        if (rows != cols) {
            double[][] dist = new double[rows][rows];

            // Parallelize the upper triangle distance computation
            IntStream.range(0, rows).parallel().forEach(i -> {
                for (int j = i; j < rows; j++) {
                    double sum = 0;
                    for (int k = 0; k < cols; k++) {
                        double diff = input[i][k] - input[j][k];
                        sum += diff * diff;
                    }
                    double d = Math.sqrt(sum);

                    dist[i][j] = d;
                    dist[j][i] = d;
                }
            });

            return dist;
        }

        // Case 2: Square matrix – assume it's a distance matrix and symmetrize
        double[][] sym = new double[rows][rows];

        // Parallelize the symmetrization step
        IntStream.range(0, rows).parallel().forEach(i -> {
            for (int j = i; j < rows; j++) {
                double avg = (input[i][j] + input[j][i]) / 2.0;
                sym[i][j] = avg;
                sym[j][i] = avg;
            }
        });

        return sym;
    }
}
