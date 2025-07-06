package com.github.trinity.supermds;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Simple implementation of the KMeans++ initialization algorithm.
 * This class selects k diverse initial centroids from the dataset
 * and keeps track of the indices of the selected data points.
 *
 * @author Sean Phillips
 */
public class KMeansPlusPlus {

    private final double[][] centroids;          // Selected centroids
    private final List<Integer> centroidIndices; // Indices of selected centroids in original dataset

    /**
     * Constructs the KMeans++ initializer.
     *
     * @param data Input dataset (n x d)
     * @param k    Number of centroids to select
     * @param seed Random seed for reproducibility
     */
    public KMeansPlusPlus(double[][] data, int k, int seed) {
        Random rand = new Random(seed);
        int n = data.length;
        int d = data[0].length;

        centroids = new double[k][d];
        centroidIndices = new ArrayList<>();

        // Step 1: Choose first centroid randomly
        int firstIndex = rand.nextInt(n);
        centroids[0] = Arrays.copyOf(data[firstIndex], d);
        centroidIndices.add(firstIndex);

        // Step 2: Choose remaining centroids with weighted probability
        for (int i = 1; i < k; i++) {
            double[] distances = new double[n];

            // Compute minimum squared distance to any existing centroid
            for (int j = 0; j < n; j++) {
                double minDist = Double.MAX_VALUE;
                for (int c = 0; c < i; c++) {
                    minDist = Math.min(minDist, squaredEuclideanDistance(data[j], centroids[c]));
                }
                distances[j] = minDist;
            }

            // Weighted random selection
            double sum = Arrays.stream(distances).sum();
            double r = rand.nextDouble() * sum;
            double cumulative = 0;

            for (int j = 0; j < n; j++) {
                cumulative += distances[j];
                if (cumulative >= r) {
                    centroids[i] = Arrays.copyOf(data[j], d);
                    centroidIndices.add(j);
                    break;
                }
            }
        }
    }

    /**
     * Returns the selected centroids.
     *
     * @return Array of k centroids (k x d)
     */
    public double[][] getCentroids() {
        return centroids;
    }

    /**
     * Returns the indices of the selected centroids in the original dataset.
     *
     * @return List of selected indices (size k)
     */
    public List<Integer> getCentroidIndices() {
        return centroidIndices;
    }

    /**
     * Helper method to compute squared Euclidean distance.
     */
    private static double squaredEuclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
}
