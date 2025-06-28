package com.github.trinity.supermds;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * @author Sean Phillips
 */
public class SuperMDSHelper {

    public static List<int[]> sampleIndexPairs(int n, int sampleSize) {
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

    public static double[][] computeSquaredEuclideanDistanceMatrix(double[][] data) {
        int n = data.length;
        double[][] D = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double dist = 0.0;
                for (int k = 0; k < data[i].length; k++) {
                    double diff = data[i][k] - data[j][k];
                    dist += diff * diff;
                }
                D[i][j] = dist;
                D[j][i] = dist;
            }
        }
        return D;
    }

    public static double[][] pairwiseDistances(double[][] X) {
        int n = X.length;
        double[][] D = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dist = squaredEuclideanDistance(X[i], X[j]);
                D[i][j] = D[j][i] = Math.sqrt(dist);
            }
        }
        return D;
    }

    public static double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    public static double squaredEuclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    public static double[][] computeReconstructedDistances(double[][] embedding) {
        int n = embedding.length;
        double[][] distances = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dist = 0.0;
                for (int d = 0; d < embedding[i].length; d++) {
                    double diff = embedding[i][d] - embedding[j][d];
                    dist += diff * diff;
                }
                dist = Math.sqrt(dist);
                distances[i][j] = dist;
                distances[j][i] = dist; // ensure symmetry
            }
        }

        return distances;
    }

    public static double[][] deepCopyParallel(double[][] original) {
        int rows = original.length;
        double[][] copy = new double[rows][];

        ForkJoinPool forkJoinPool = ForkJoinPool.commonPool();

        try {
            forkJoinPool.submit(() ->
                IntStream.range(0, rows).parallel().forEach(i -> {
                    int cols = original[i].length;
                    copy[i] = new double[cols];
                    System.arraycopy(original[i], 0, copy[i], 0, cols);
                })
            ).get(); // Wait for completion
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Parallel array copy failed", e);
        }

        return copy;
    }

    /**
     * Normalizes the input data using Z-score normalization.
     * <p>
     * For each feature (column), this method subtracts the mean and divides by the standard deviation,
     * resulting in each column having zero mean and unit variance. This normalization is commonly used
     * in distance-based algorithms like MDS, PCA, and clustering to ensure all features contribute equally.
     *
     * @param data A 2D array of shape [n][d] where n is the number of data points and d is the dimensionality.
     * @return A new 2D array of the same shape with Z-score normalized values.
     */
    public static double[][] zScoreNormalize(double[][] data) {
        int n = data.length;          // Number of data points
        int d = data[0].length;       // Number of dimensions (features)
        double[][] normalized = new double[n][d]; // Output array for normalized data

        // Iterate over each dimension (column/feature)
        for (int j = 0; j < d; j++) {
            double sum = 0;
            double sumSq = 0;

            // Compute the sum and sum of squares for this feature
            for (int i = 0; i < n; i++) {
                sum += data[i][j];
                sumSq += data[i][j] * data[i][j];
            }

            // Calculate the mean and standard deviation
            double mean = sum / n;
            double variance = sumSq / n - mean * mean;
            double stdDev = Math.sqrt(variance);

            // Normalize each data point in this dimension
            for (int i = 0; i < n; i++) {
                // If stdDev is 0 (constant column), avoid division by zero
                normalized[i][j] = (data[i][j] - mean) / (stdDev == 0 ? 1 : stdDev);
            }
        }

        return normalized;
    }

    /**
     * Normalizes the input data so that each feature has zero mean and unit variance.
     *
     * @param data The input data matrix of shape [n][d], where n is the number of samples and d is the number of features.
     *             This array is modified in-place.
     * @return The normalized data matrix (same reference as input).
     */
    public static double[][] normalizeDataZeroMean(double[][] data) {
        int n = data.length;
        if (n == 0) return data;

        int d = data[0].length;
        double[] means = new double[d];
        double[] stdDevs = new double[d];

        // Compute mean of each feature
        for (int j = 0; j < d; j++) {
            for (int i = 0; i < n; i++) {
                means[j] += data[i][j];
            }
            means[j] /= n;
        }

        // Compute standard deviation of each feature
        for (int j = 0; j < d; j++) {
            for (int i = 0; i < n; i++) {
                stdDevs[j] += Math.pow(data[i][j] - means[j], 2);
            }
            stdDevs[j] = Math.sqrt(stdDevs[j] / n);
        }

        // Normalize each feature
        for (int j = 0; j < d; j++) {
            double std = stdDevs[j] == 0 ? 1e-8 : stdDevs[j]; // avoid divide-by-zero
            for (int i = 0; i < n; i++) {
                data[i][j] = (data[i][j] - means[j]) / std;
            }
        }

        return data;
    }

    /**
     * Normalizes the input data so that each feature (column) is scaled to the [0, 1] range.
     * This is done independently for each column using the formula:
     * x_norm = (x - min) / (max - min)
     *
     * @param data A 2D array of input vectors where data[i][j] is the j-th feature of the i-th sample.
     * @return A new 2D array of the same shape where each column is normalized to [0, 1].
     */
    public static double[][] maxNormalize(double[][] data) {
        int n = data.length;
        if (n == 0) return new double[0][];
        int d = data[0].length;

        double[] min = new double[d];
        double[] max = new double[d];
        Arrays.fill(min, Double.POSITIVE_INFINITY);
        Arrays.fill(max, Double.NEGATIVE_INFINITY);

        // Find min and max for each column
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                double val = data[i][j];
                if (val < min[j]) min[j] = val;
                if (val > max[j]) max[j] = val;
            }
        }

        // Create normalized output
        double[][] normalized = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                double range = max[j] - min[j];
                if (range == 0) {
                    normalized[i][j] = 0.0; // Avoid division by zero, treat constant column as 0
                } else {
                    normalized[i][j] = (data[i][j] - min[j]) / range;
                }
            }
        }

        return normalized;
    }

    public static double[][] normalizeDistancesParallel(double[][] D) {
        int n = D.length;
        double[][] normalized = new double[n][n];

        // Step 1: Find maximum distance (serial — fast and avoids race conditions)
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (D[i][j] > max) {
                    max = D[i][j];
                }
            }
        }
        double finalMax = max;
        // Step 2: Normalize distances in parallel
        IntStream.range(0, n).parallel().forEach(i -> {
            for (int j = i; j < n; j++) {
                double val = D[i][j] / finalMax;
                normalized[i][j] = val;
                normalized[j][i] = val; // enforce symmetry
            }
        });

        return normalized;
    }

    public static double[] normalizeDistances(double[] dists) {
        double max = Arrays.stream(dists).max().orElse(1.0);
        if (max == 0.0) return dists.clone(); // all zero
        return Arrays.stream(dists).map(d -> d / max).toArray();
    }


    /**
     * Computes Euclidean distances from a new point to each row in the training dataset.
     *
     * @param newPoint     The new point as a 1D array (length = dim).
     * @param trainingData The training dataset as a 2D array (shape: n x dim).
     * @return A 1D array of distances from the new point to each training point.
     */
    public static double[] distancesToNewPoint(double[] newPoint, double[][] trainingData) {
        int n = trainingData.length;
        double[] distances = new double[n];

        for (int i = 0; i < n; i++) {
            distances[i] = euclideanDistance(newPoint, trainingData[i]);
        }

        return distances;
    }
}
