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
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
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
 * - FARTHEST_POINT: Emphasize farthest distance points to minimize extrapolation
 * - BOUNDARY_SENSITIVE: iterative boundary finding... can't remember but its good
 */
public class SuperMDSAnchors {
    /** Available strategies for selecting anchor points. */
    public enum Strategy {
        RANDOM,
        KMEANS_PLUS_PLUS,
        PCA,
        UNIFORM,
        FARTHEST_POINT,
        BOUNDARY_SENSITIVE        
    }
    //Just a quick little record to associate selected indices with their vectors
    public record AnchorSetRecord (double[][] anchors, List<Integer> indices){};
    
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
            case FARTHEST_POINT:
                return farthestPointSampling(data, numAnchors);
            case BOUNDARY_SENSITIVE:
                return boundarySensitiveSampling(data, numAnchors);                
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
     * Selects a subset of anchor points from the given dataset using the Farthest Point Sampling (FPS) strategy.
     * <p>
     * This method begins by randomly selecting an initial point, and iteratively adds the point that is farthest
     * from the current set of selected anchors. Distance is measured in the input space using Euclidean distance.
     * The process continues until the desired number of anchor points is selected.
     * </p>
     *
     * @param data        The input data matrix (shape: N × D), where N is the number of samples and D is the feature dimension.
     * @param numAnchors  The number of anchor points to select.
     * @return            An {@link AnchorSetRecord} containing the selected anchor vectors and their corresponding indices in the original dataset.
     */    
    public static AnchorSetRecord farthestPointSampling(double[][] data, int numAnchors) {
        Random rand = new Random(42);
        List<Integer> selected = new ArrayList<>();
        boolean[] chosen = new boolean[data.length];

        // Start with a random point
        int first = rand.nextInt(data.length);
        selected.add(first);
        chosen[first] = true;

        double[] minDists = new double[data.length];
        Arrays.fill(minDists, Double.POSITIVE_INFINITY);

        for (int i = 1; i < numAnchors; i++) {
            int farthest = -1;
            double maxDist = -1;

            for (int j = 0; j < data.length; j++) {
                if (chosen[j]) continue;
                double dist = SuperMDSHelper.euclideanDistance(data[j], data[selected.get(i - 1)]);
                minDists[j] = Math.min(minDists[j], dist);
                if (minDists[j] > maxDist) {
                    maxDist = minDists[j];
                    farthest = j;
                }
            }

            if (farthest != -1) {
                selected.add(farthest);
                chosen[farthest] = true;
            }
        }
        // Map selected indices to data points
        return new AnchorSetRecord(
            selected.stream().map(i -> data[i]).toArray(double[][]::new), 
            selected);
    }
    /**
     * Selects a subset of anchor points that are sensitive to the boundary of the data distribution
     * by leveraging projections along the first principal component.
     * <p>
     * This method computes the first principal axis of the data (via PCA) and ranks all data points
     * by the absolute magnitude of their projection onto this axis. The top-scoring points—those farthest
     * from the center along the primary direction of variance—are selected as anchors. This tends to favor
     * boundary or extreme points that capture the spread of the dataset.
     * </p>
     *
     * @param data        The input data matrix (shape: N × D), where N is the number of samples and D is the feature dimension.
     * @param numAnchors  The number of anchor points to select.
     * @return            An {@link AnchorSetRecord} containing the selected anchor vectors and their corresponding indices in the original dataset.
     */
    public static AnchorSetRecord boundarySensitiveSampling(double[][] data, int numAnchors) {
        RealMatrix X = new Array2DRowRealMatrix(data);
        Covariance cov = new Covariance(X);
        RealMatrix covMatrix = cov.getCovarianceMatrix();
        EigenDecomposition eig = new EigenDecomposition(covMatrix);
        RealVector principal = eig.getEigenvector(0);  // first principal axis

        // Score each point by its projection on principal axis
        List<Integer> indices = IntStream.range(0, data.length).boxed().collect(Collectors.toList());
        indices.sort(Comparator.comparingDouble(i -> {
            return -Math.abs(new ArrayRealVector(data[i]).dotProduct(principal));
        }));

        // Map selected indices to data points
        List<Integer> selected = indices.subList(0, Math.min(numAnchors, indices.size()));
        return new AnchorSetRecord(
            selected.stream().map(i -> data[i]).toArray(double[][]::new), 
            selected);        
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
