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
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.util.FastMath;

public class SuperMDS {

    public enum Mode {
        CLASSICAL,
        METRIC,
        NONMETRIC,
        SUPERVISED,
        PARALLEL,
        LANDMARK
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
            double[][] dist = pairwiseDistances(X);
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

    private static List<int[]> sampleIndexPairs(int n, int sampleSize) {
        Random rand = new Random(42); // or pass via Params
        Set<Long> seen = new HashSet<>();
        List<int[]> pairs = new ArrayList<>(sampleSize);
        int attempts = 0;

        while (pairs.size() < sampleSize && attempts < sampleSize * 10) {
            int i = rand.nextInt(n);
            int j = rand.nextInt(n);
            if (i == j) {
                continue;
            }

            long key = ((long) Math.min(i, j) << 32) | Math.max(i, j);
            if (seen.add(key)) {
                pairs.add(new int[]{i, j});
            }
            attempts++;
        }

        return pairs;
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

    public static double[][] smacofMDSParallel(
            double[][] D, int dim, int maxIter, double tol,
            boolean earlyExitOnStressIncrease, boolean useStressSampling, int stressSampleSize) {

        int n = D.length;
        Random rand = new Random();
        double[][] X = new double[n][dim];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                X[i][j] = rand.nextGaussian();
            }
        }

        double prevStress = Double.MAX_VALUE;
        for (int iter = 0; iter < maxIter; iter++) {
            double[][] dist = pairwiseDistances(X);
            double[][] newX = new double[n][dim];
            double[][] tempX = SuperMDSHelper.deepCopyParallel(X);

            IntStream.range(0, n).parallel().forEach(i -> {
                for (int k = 0; k < dim; k++) {
                    double num = 0, denom = 0;
                    for (int j = 0; j < n; j++) {
                        if (i == j) {
                            continue;
                        }
                        double d = dist[i][j];
                        if (d > 0) {
                            double delta = D[i][j] / d;
                            num += tempX[j][k] + delta * (tempX[i][k] - tempX[j][k]);
                            denom++;
                        }
                    }
                    if (denom > 0) {
                        newX[i][k] = num / denom;
                    }
                }
            });

            // Compute stress
            double stress = useStressSampling
                    ? computeSampledStress(D, newX, stressSampleSize)
                    : computeFullStressParallelWeighted(D, newX, null); //null weights means ignore

            if (Math.abs(prevStress - stress) < tol) {
                break;
            }
            if (stress > prevStress && earlyExitOnStressIncrease) {
                break;
            }

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

    public static double[][] computeLandmarkMDS(double[][] data, int dim, int numLandmarks, boolean useKMeansPlusPlus, int seed) {
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
     * Performs Canonical Landmark Multidimensional Scaling (MDS) using a subset of landmark points to reduce 
     * computational complexity for large datasets. It uses eigen-decomposition of the Gram matrix computed from
     * pairwise distances between landmarks and applies a Nystrom approximation to embed the remaining points.
     *
     * @param data               The full input dataset as an array of N data points, each of dimensionality D.
     * @param dim                The number of dimensions for the target low-dimensional embedding (e.g., 2 or 3).
     * @param numLandmarks       The number of landmark points to use for the MDS approximation.
     * @param useKMeansPlusPlus  If {@code true}, selects landmark points using the k-means++ algorithm for better coverage.
     *                           If {@code false}, selects landmarks uniformly at random.
     * @param seed               A random seed for reproducibility when selecting landmarks (used in both k-means++ and random modes).
     *
     * @return A 2D array of shape [N x dim] containing the embedded coordinates of all input points in the reduced-dimensional space.
     *         Landmark points are embedded directly from the eigen-decomposition; all other points are approximated using Nystrom extension.
     */    
    public static double[][] canonicalLandmarkMDS_OLD(double[][] data, int dim, int numLandmarks, boolean useKMeansPlusPlus, int seed) {
        int n = data.length;
        int d = data[0].length;

        // === Step 1: Select landmark points (either randomly or via k-means++) ===
        double[][] landmarks;
        int[] landmarkIndices = new int[numLandmarks];

        if (useKMeansPlusPlus) {
            KMeansPlusPlus kpp = new KMeansPlusPlus(data, numLandmarks, seed);
            landmarks = kpp.getCentroids();
            // KMeans++ doesn't return indices; set to -1 as placeholder
            Arrays.fill(landmarkIndices, -1); 
        } else {
            Random rand = new Random(seed);
            Set<Integer> indices = new HashSet<>();
            while (indices.size() < numLandmarks)
                indices.add(rand.nextInt(n));

            landmarks = new double[numLandmarks][d];
            int idx = 0;
            for (int i : indices) {
                landmarks[idx] = data[i];
                landmarkIndices[idx] = i;
                idx++;
            }
            Arrays.sort(landmarkIndices); // Used for binarySearch below
        }

        // === Step 2: Compute squared distances between all landmark pairs ===
        double[][] squaredDistancesForLandmarks = new double[numLandmarks][numLandmarks];
        for (int i = 0; i < numLandmarks; i++) {
            for (int j = i; j < numLandmarks; j++) {
                squaredDistancesForLandmarks[i][j] = squaredDistancesForLandmarks[j][i] = 
                    SuperMDSHelper.squaredEuclideanDistance(landmarks[i], landmarks[j]);
            }
        }        
        // === Step 3: Double-center squaredDistancesForLandmarks to form the Gram matrix G ===
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
                //gramMatrix[i][j] = -0.5 * (squaredDistancesForLandmarks[i][j] - rowMeans[i] - rowMeans[j] + totalMean);
                gramMatrix[i][j] = -0.5 * (squaredDistancesForLandmarks[i][j] - rowMeans[i] - colMeans[j] + totalMean);
            }
        }

        // === Step 4: Eigen-decomposition of Gram matrix  ===
        RealMatrix realGramMatrix = MatrixUtils.createRealMatrix(gramMatrix);
        EigenDecomposition eig = new EigenDecomposition(realGramMatrix);
        double[][] landmarkEmbedding = new double[numLandmarks][dim];

        for (int i = 0; i < dim; i++) {
            double eigValSqrt = Math.sqrt(Math.max(eig.getRealEigenvalue(i), 0));
            RealVector eigVec = eig.getEigenvector(i);
            for (int j = 0; j < numLandmarks; j++) {
                landmarkEmbedding[j][i] = eigVec.getEntry(j) * eigValSqrt;
            }
        }

        // === Step 5: Embed all data points using Nystrom extension ===
        double[][] embedding = new double[n][dim];
        for (int i = 0; i < n; i++) {
            // Check if this point is one of the landmarks (only works for random selection)
            int landmarkIndex = (useKMeansPlusPlus) ? -1 : Arrays.binarySearch(landmarkIndices, i);
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
                    if (eigVal <= 1e-10) continue;
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
    //A Nyström approximation to Classical MDS using n × k matrix of centered squared distances to landmarks.
    public static double[][] approximateMDSViaLandmarks(double[][] data, int dim, int numLandmarks, boolean useKMeansPlusPlus, int seed) {
        int n = data.length;
        int d = data[0].length;
        double[][] landmarks;

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

        double[][] D = new double[n][numLandmarks];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < numLandmarks; j++) {
                D[i][j] = SuperMDSHelper.squaredEuclideanDistance(data[i], landmarks[j]);
            }
        }

        double[][] B = new double[n][numLandmarks];
        double[] rowMeans = Arrays.stream(D).mapToDouble(row -> Arrays.stream(row).average().orElse(0)).toArray();
        double[] colMeans = new double[numLandmarks];
        for (int j = 0; j < numLandmarks; j++) {
            for (int i = 0; i < n; i++) {
                colMeans[j] += D[i][j];
            }
            colMeans[j] /= n;
        }

        double totalMean = Arrays.stream(rowMeans).average().orElse(0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < numLandmarks; j++) {
                B[i][j] = -0.5 * (D[i][j] - rowMeans[i] - colMeans[j] + totalMean);
            }
        }

        RealMatrix Bmat = MatrixUtils.createRealMatrix(B);
        SingularValueDecomposition svd = new SingularValueDecomposition(Bmat);
        RealMatrix U = svd.getU();
        double[] S = svd.getSingularValues();
        double[][] result = new double[n][dim];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                result[i][j] = U.getEntry(i, j) * Math.sqrt(S[j]);
            }
        }

        return result;
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

    //Parallelized Stress convergence check
    public static double stressConvergenceParallel(double[][] X,
            double[] embedded, double[] newCoords, double[] distsToExisting) {
        int n = X.length;
        int dim = X[0].length;
        DoubleAdder stress = new DoubleAdder();
        DoubleAdder oldStress = new DoubleAdder();
        double[] finalNewCoords = newCoords.clone();
        double[] finalOldCoords = embedded.clone();

        IntStream.range(0, n).parallel().forEach(i -> {
            double d1 = 0, d2 = 0;
            for (int d = 0; d < dim; d++) {
                double diff1 = finalNewCoords[d] - X[i][d];
                double diff2 = finalOldCoords[d] - X[i][d];
                d1 += diff1 * diff1;
                d2 += diff2 * diff2;
            }

            double dist1 = Math.sqrt(d1);
            double dist2 = Math.sqrt(d2);
            double delta = distsToExisting[i];

            stress.add((delta - dist1) * (delta - dist1));
            oldStress.add((delta - dist2) * (delta - dist2));
        });
        return Math.abs(oldStress.sum() - stress.sum());
    }

    public static double[][] pairwiseDistances(double[][] X) {
        int n = X.length;
        double[][] D = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dist = SuperMDSHelper.squaredEuclideanDistance(X[i], X[j]);
                D[i][j] = D[j][i] = Math.sqrt(dist);
            }
        }
        return D;
    }

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
