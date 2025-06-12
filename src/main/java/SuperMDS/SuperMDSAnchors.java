package SuperMDS;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.correlation.Covariance;

/**
 * Utility class for selecting anchor (or landmark) points from a high-dimensional dataset.
 * 
 * These anchors are often used in dimensionality reduction, inversion, or approximation algorithms
 * like Landmark MDS, Nystrom approximation, or for multilateration-based inversion methods.
 *
 * Supports multiple strategies for anchor selection:
 * - RANDOM: Randomly sampled data points.
 * - KMEANS_PLUS_PLUS: Centroids from k-means++ initialization.
 * - PCA: Points spread along principal component directions.
 * - UNIFORM: Uniformly spaced indices across the dataset.
 */
public class SuperMDSAnchors {
    /** Available strategies for selecting anchor points. */
    public enum Strategy {
        RANDOM,
        KMEANS_PLUS_PLUS,
        PCA,
        UNIFORM
    }
    public record AnchorSetRecord (double[][] anchors, List<Integer> indices){
    
    };
    /**
     * Select anchor points from a dataset using the given strategy.
     *
     * @param data        Input dataset (n x d), where n is the number of points and d the dimensionality.
     * @param numAnchors  Number of anchors to select.
     * @param strategy    Strategy to use for selection.
     * @param seed        Random seed (used by stochastic methods like RANDOM and KMEANS_PLUS_PLUS).
     * @return            A AnchorSetRecord with (numAnchors x d) matrix of selected anchor points.
     */
    public static AnchorSetRecord selectAnchors(double[][] data, int numAnchors, Strategy strategy, int seed) {
        switch (strategy) {
            case RANDOM:
                return selectRandomAnchors(data, numAnchors, seed);
            case KMEANS_PLUS_PLUS:
                return selectKMeansPlusPlusAnchors(data, numAnchors, seed);
            case PCA:
                return selectPCAAnchors(data, numAnchors);
            case UNIFORM:
                return selectUniformAnchors(data, numAnchors);
            default:
                throw new IllegalArgumentException("Unknown anchor selection strategy.");
        }
    }

    /**
     * Helper method that extracts a subset of rows from the input 2D array based on a list of row indices.
     * The order of the extracted rows will match the order of indices in the indicesList.
     *
     * @param data        The original 2D data array (n x d)
     * @param indicesList A list of row indices to extract
     * @return A new 2D array containing only the selected rows in the specified order
     */
    public static double[][] extractByIndices(double[][] data, List<Integer> indicesList) {
        int numRows = indicesList.size();
        int numCols = data[0].length;
        double[][] extracted = new double[numRows][numCols];

        for (int i = 0; i < numRows; i++) {
            int index = indicesList.get(i);
            extracted[i] = java.util.Arrays.copyOf(data[index], numCols);
        }

        return extracted;
    }
    

    /**
     * Randomly selects a subset of data points as anchors.
     * Ensures no duplicates.
     */
    private static AnchorSetRecord selectRandomAnchors(double[][] data, int numAnchors, int seed) {
        Random rand = new Random(seed);
        Set<Integer> indices = new HashSet<>();

        // Randomly sample unique indices
        while (indices.size() < numAnchors) {
            indices.add(rand.nextInt(data.length));
        }

        // Map selected indices to data points
        return new AnchorSetRecord(
            indices.stream()
                .map(i -> data[i])
                .toArray(double[][]::new),
            indices.stream().toList());
        
    }

    /**
     * Selects anchors by evenly sampling data across its index range.
     * Useful when data is sorted or structured.
     */
    private static AnchorSetRecord selectUniformAnchors(double[][] data, int numAnchors) {
        double step = (double) data.length / numAnchors;
        double[][] anchors = new double[numAnchors][];

        List<Integer> indicesList = new ArrayList<>(numAnchors);
        // Select every step-th point
        for (int i = 0; i < numAnchors; i++) {
            int index = Math.min((int) (i * step), data.length - 1);
            anchors[i] = data[index];
            indicesList.add(index);
        }
        return new AnchorSetRecord(anchors, indicesList);
    }

    /**
     * Selects centroids using a k-means++ initialization algorithm.
     * Provides diversity in selected anchor locations.
     */
    private static AnchorSetRecord selectKMeansPlusPlusAnchors(double[][] data, int numAnchors, int seed) {
        // Assumes you have a KMeansPlusPlus class implemented already.
        KMeansPlusPlus kpp = new KMeansPlusPlus(data, numAnchors, seed);
        
        return new AnchorSetRecord(kpp.getCentroids(), kpp.getCentroidIndices());        
    }

    /**
     * Selects anchors by projecting data onto the first principal component,
     * then choosing samples spaced along that direction.
     * This spreads anchors along the main axis of variance in the dataset.
     */
    private static AnchorSetRecord selectPCAAnchors(double[][] data, int numAnchors) {
        // Convert data to Apache Commons Math RealMatrix
        RealMatrix X = MatrixUtils.createRealMatrix(data);

        // Compute covariance matrix and eigen-decomposition
        Covariance covariance = new Covariance(X);
        RealMatrix covMatrix = covariance.getCovarianceMatrix();
        EigenDecomposition eig = new EigenDecomposition(covMatrix);

        // Get the first principal component (top eigenvector)
        RealVector pc1 = eig.getV().getColumnVector(0);

        // Project each data point onto the first principal component
        double[] projections = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            projections[i] = pc1.dotProduct(new ArrayRealVector(data[i]));
        }

        // Sort points based on their projection values (ascending)
        int[] sortedIndices = IntStream.range(0, projections.length)
                .boxed()
                .sorted(Comparator.comparingDouble(i -> projections[i]))
                .mapToInt(i -> i)
                .toArray();

        // Evenly sample along the sorted projection space
        int step = Math.max(1, projections.length / numAnchors);
        double[][] anchors = new double[numAnchors][];
        for (int i = 0; i < numAnchors; i++) {
            anchors[i] = data[sortedIndices[i * step]];
        }
        List<Integer> indicesList = Arrays.stream(sortedIndices)
            .limit(numAnchors)
            .boxed()
            .collect(Collectors.toList());
        
        return new AnchorSetRecord(anchors, indicesList);
    }
}
