package SuperMDS;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 *
 * @author Sean Phillips
 */
public class SuperMDSHelper {
    
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
    
    // Generate random N x D data
    public static double[][] generateSyntheticData(int n, int dim) {
        double[][] data = new double[n][dim];
        Random rand = new Random(42); // deterministic seed

        for (int i = 0; i < n; i++)
            for (int j = 0; j < dim; j++)
                data[i][j] = rand.nextGaussian(); // normal distribution

        return data;
    }

    // Generate synthetic integer class labels (0 to numClasses-1)
    public static int[] generateSyntheticLabels(int n, int numClasses) {
        int[] labels = new int[n];
        Random rand = new Random(42);
        for (int i = 0; i < n; i++)
            labels[i] = rand.nextInt(numClasses);
        return labels;
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
