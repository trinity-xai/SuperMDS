package com.github.trinity.supermds;

/**
 * Utility class for normalizing and unnormalizing data using either Min-Max or Z-Score normalization.
 *
 * <p>This class supports two normalization strategies:</p>
 * <ul>
 *   <li><b>MIN_MAX:</b> Scales features to the range [0, 1] using min and max per dimension.</li>
 *   <li><b>Z_SCORE:</b> Standardizes features to have zero mean and unit variance.</li>
 * </ul>
 *
 * <p>Usage example:</p>
 * <pre>
 *     Normalizer normalizer = new Normalizer(data, Normalizer.Type.Z_SCORE);
 *     double[][] normalized = normalizer.normalizeAll(data);
 *     ...
 *     double[][] originalScale = normalizer.unnormalizeAll(normalized);
 * </pre>
 *
 * @author Sean Phillips
 */
public class Normalizer {

    public enum Type {
        MIN_MAX, Z_SCORE
    }

    private final int dim;
    private final Type type;

    private double[] min;
    private double[] max;

    private double[] mean;
    private double[] std;

    /**
     * Constructs a Normalizer with the given data and normalization type.
     *
     * @param data Dataset as a 2D array (points × dimensions).
     * @param type Normalization strategy (MIN_MAX or Z_SCORE).
     */
    public Normalizer(double[][] data, Type type) {
        this.dim = data[0].length;
        this.type = type;

        switch (type) {
            case MIN_MAX -> computeMinMax(data);
            case Z_SCORE -> computeZScoreStats(data);
            default -> throw new IllegalArgumentException("Unsupported normalization type: " + type);
        }
    }

    // ----- Strategy Computation -----

    private void computeMinMax(double[][] data) {
        min = new double[dim];
        max = new double[dim];

        for (int d = 0; d < dim; d++) {
            min[d] = Double.POSITIVE_INFINITY;
            max[d] = Double.NEGATIVE_INFINITY;
        }

        for (double[] vec : data) {
            for (int d = 0; d < dim; d++) {
                min[d] = Math.min(min[d], vec[d]);
                max[d] = Math.max(max[d], vec[d]);
            }
        }
    }

    private void computeZScoreStats(double[][] data) {
        mean = new double[dim];
        std = new double[dim];

        // Compute mean
        for (double[] vec : data) {
            for (int d = 0; d < dim; d++) {
                mean[d] += vec[d];
            }
        }
        for (int d = 0; d < dim; d++) {
            mean[d] /= data.length;
        }

        // Compute standard deviation
        for (double[] vec : data) {
            for (int d = 0; d < dim; d++) {
                std[d] += Math.pow(vec[d] - mean[d], 2);
            }
        }
        for (int d = 0; d < dim; d++) {
            std[d] = Math.sqrt(std[d] / data.length);
            if (std[d] == 0.0) std[d] = 1.0; // prevent divide-by-zero
        }
    }

    // ----- Normalization / Unnormalization -----

    /**
     * Normalizes a single vector using the selected strategy.
     *
     * @param vec The input vector.
     * @return A normalized vector.
     */
    public double[] normalize(double[] vec) {
        double[] result = new double[dim];
        switch (type) {
            case MIN_MAX -> {
                for (int d = 0; d < dim; d++) {
                    result[d] = (max[d] == min[d]) ? 0.0 : (vec[d] - min[d]) / (max[d] - min[d]);
                }
            }
            case Z_SCORE -> {
                for (int d = 0; d < dim; d++) {
                    result[d] = (vec[d] - mean[d]) / std[d];
                }
            }
        }
        return result;
    }

    /**
     * Unnormalizes a normalized vector back to original scale.
     *
     * @param vec Normalized input vector.
     * @return The original-scale vector.
     */
    public double[] unnormalize(double[] vec) {
        double[] result = new double[dim];
        switch (type) {
            case MIN_MAX -> {
                for (int d = 0; d < dim; d++) {
                    result[d] = vec[d] * (max[d] - min[d]) + min[d];
                }
            }
            case Z_SCORE -> {
                for (int d = 0; d < dim; d++) {
                    result[d] = vec[d] * std[d] + mean[d];
                }
            }
        }
        return result;
    }

    /**
     * Normalizes all vectors in the dataset.
     *
     * @param data 2D array of vectors to normalize.
     * @return Normalized 2D array.
     */
    public double[][] normalizeAll(double[][] data) {
        double[][] result = new double[data.length][dim];
        for (int i = 0; i < data.length; i++) {
            result[i] = normalize(data[i]);
        }
        return result;
    }

    /**
     * Unnormalizes all normalized vectors to their original scale.
     *
     * @param data 2D array of normalized vectors.
     * @return 2D array of unnormalized (original scale) vectors.
     */
    public double[][] unnormalizeAll(double[][] data) {
        double[][] result = new double[data.length][dim];
        for (int i = 0; i < data.length; i++) {
            result[i] = unnormalize(data[i]);
        }
        return result;
    }

    // ----- Accessors -----

    public double[] getMin() {
        return min;
    }

    public double[] getMax() {
        return max;
    }

    public double[] getMean() {
        return mean;
    }

    public double[] getStd() {
        return std;
    }

    public Type getType() {
        return type;
    }
}
