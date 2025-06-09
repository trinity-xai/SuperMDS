package SuperMDS;

import java.util.Arrays;
import java.util.Random;

/**
 *
 * @author Sean Phillips
 */
public class KMeansPlusPlus {

    private double[][] centroids;

    public KMeansPlusPlus(double[][] data, int k, int seed) {
        Random rand = new Random(seed);
        int n = data.length, d = data[0].length;
        centroids = new double[k][d];
        centroids[0] = Arrays.copyOf(data[rand.nextInt(n)], d);

        for (int i = 1; i < k; i++) {
            double[] distances = new double[n];
            for (int j = 0; j < n; j++) {
                double minDist = Double.MAX_VALUE;
                for (int c = 0; c < i; c++)
                    minDist = Math.min(minDist, squaredEuclideanDistance(data[j], centroids[c]));
                distances[j] = minDist;
            }

            double sum = Arrays.stream(distances).sum();
            double r = rand.nextDouble() * sum;
            double cumulative = 0;
            for (int j = 0; j < n; j++) {
                cumulative += distances[j];
                if (cumulative >= r) {
                    centroids[i] = Arrays.copyOf(data[j], d);
                    break;
                }
            }
        }
    }

    public double[][] getCentroids() {
        return centroids;
    }
    private static double euclideanDistance(double[] a, double[] b) {
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
    
}