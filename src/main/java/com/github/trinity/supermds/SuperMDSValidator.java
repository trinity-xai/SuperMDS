package com.github.trinity.supermds;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static com.github.trinity.supermds.SuperMDSHelper.euclideanDistance;

/**
 * @author Sean Phillips
 */
public class SuperMDSValidator {
    private static final Logger LOG = LoggerFactory.getLogger(SuperMDSValidator.class);

    public static class StressMetrics {
        public double stress1;
        public double stress2;
        public double goodnessOfFit;

        @Override
        public String toString() {
            return String.format(
                "Stress-1 (Kruskal): %.6f\n" +
                    "Stress-2 (Normalized raw stress): %.6f\n" +
                    "Goodness-of-Fit: %.6f <--------------------",
                stress1, stress2, goodnessOfFit
            );
        }
    }

    public static StressMetrics computeStressMetrics(double[][] originalDistances, double[][] embeddedDistances) {
        int n = originalDistances.length;
        double sumSquaredDiff = 0.0;
        double sumSquaredOriginal = 0.0;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dij = originalDistances[i][j];
                double dij_hat = embeddedDistances[i][j];
                double diff = dij - dij_hat;

                sumSquaredDiff += diff * diff;
                sumSquaredOriginal += dij * dij;
            }
        }

        StressMetrics result = new StressMetrics();

        result.stress2 = sumSquaredDiff / sumSquaredOriginal;
        result.stress1 = Math.sqrt(result.stress2);
        result.goodnessOfFit = 1.0 - result.stress2;

        return result;
    }

    public static void computeStressMetricsClassic(double[][] original, double[][] embedded) {
        int n = original.length;
        double stressNumerator = 0;
        double stressDenominator = 0;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dOrig = 0, dEmb = 0;
                for (int k = 0; k < original[0].length; k++) {
                    double diff = original[i][k] - original[j][k];
                    dOrig += diff * diff;
                }
                for (int k = 0; k < embedded[0].length; k++) {
                    double diff = embedded[i][k] - embedded[j][k];
                    dEmb += diff * diff;
                }
                dOrig = Math.sqrt(dOrig);
                dEmb = Math.sqrt(dEmb);

                double diff = dOrig - dEmb;
                stressNumerator += diff * diff;
                stressDenominator += dOrig * dOrig;
            }
        }

        double stress1 = Math.sqrt(stressNumerator / stressDenominator);
        double stress2 = stressNumerator / stressDenominator;
        double gof = 1.0 - stress2;

        LOG.info("Stress-1 (Kruskal): {}", String.format("%.6f", stress1));
        LOG.info("Stress-2 (Normalized raw stress): {}", String.format("%.6f", stress2));
        LOG.info("Goodness-of-Fit: {}  <--------------------", String.format("%.6f", gof));
    }

    public static double computeOSEGoodnessOfFit(double[][] embeddings, double[] newPointCoords, double[] distancesToNew) {
        double stress = 0.0;
        double totalVar = 0.0;

        for (int i = 0; i < embeddings.length; i++) {
            double dHat = SuperMDSHelper.euclideanDistance(newPointCoords, embeddings[i]);
            double diff = distancesToNew[i] - dHat;
            stress += diff * diff;
            totalVar += distancesToNew[i] * distancesToNew[i];
        }

        if (totalVar == 0.0) return 1.0; // perfect fit (edge case)
        return 1.0 - (stress / totalVar); // normalized goodness-of-fit
    }

    /**
     * stress = Σᵢ (dᵢ - d̂ᵢ)²
     * Where:
     * dᵢ = actual (original) distance from the new point to training point i
     * d̂ᵢ = reconstructed distance between the new point and point i in the MDS embedding
     * The stress is simply the sum of squared errors between original and reconstructed distances
     *
     * @param embeddings
     * @param newPointCoords
     * @param distancesToNew
     * @return computed stress value
     */
    public static double computeOSEStress(double[][] embeddings, double[] newPointCoords, double[] distancesToNew) {
        double stress = 0.0;
        for (int i = 0; i < embeddings.length; i++) {
            double dHat = SuperMDSHelper.euclideanDistance(newPointCoords, embeddings[i]);
            double diff = distancesToNew[i] - dHat;
            stress += diff * diff;
        }
        return stress;
    }

    /**
     * Computes the mean reconstruction error (mean squared error) between the original
     * high-dimensional vectors and their reconstructed versions after dimensionality
     * reduction and inverse transformation.
     * <p>
     * This metric evaluates how well the inverse mapping process is able to recover
     * the original high-dimensional structure from a low-dimensional embedding.
     * It computes the average squared Euclidean distance between corresponding points
     * in the original and reconstructed spaces:
     * <pre>
     * error = (1 / n) * Σ_{i=1 to n} ||original[i] - reconstructed[i]||^2
     * </pre>
     * where n is the number of points, and each ||·|| is the Euclidean norm.
     * <p>
     * A lower value indicates better reconstruction fidelity, while a higher value
     * implies a less accurate inversion.
     *
     * @param original      The original high-dimensional data points (n × d).
     * @param reconstructed The reconstructed high-dimensional points (n × d),
     *                      typically obtained by applying an inverse transform to
     *                      their low-dimensional embeddings.
     * @return The mean squared reconstruction error across all points.
     */
    public static double meanReconstructionError(double[][] original, double[][] reconstructed) {
        int n = original.length;
        int d = original[0].length;
        double totalError = 0;

        for (int i = 0; i < n; i++) {
            double error = 0;
            for (int j = 0; j < d; j++) {
                double diff = original[i][j] - reconstructed[i][j];
                error += diff * diff;
            }
            totalError += error;
        }

        return totalError / n;
    }

    /**
     * Computes the round-trip consistency error between original and reconstructed anchor points
     * after dimensionality reduction and inversion.
     * <p>
     * This method measures how well the pairwise distances between anchor points are preserved
     * after projecting them to a low-dimensional space and then reconstructing them back to the
     * high-dimensional space. It calculates the sum of squared differences between the original
     * and reconstructed pairwise distances:
     * <pre>
     * error = Σ_{i < j} (||originalHigh[i] - originalHigh[j]|| - ||anchors[i] - anchors[j]||)^2
     * </pre>
     * This metric is useful for assessing the geometric fidelity of MDS inversion with respect to
     * a selected set of anchor points.
     *
     * @param originalHigh The original high-dimensional coordinates of the anchor points (k × d).
     * @param embeddedLow  The low-dimensional projections of the anchor points (k × dim).
     * @param anchors      The reconstructed high-dimensional coordinates (after inverse mapping).
     * @return The total round-trip consistency error between original and reconstructed anchors.
     */
    public static double roundTripAnchorConsistency(
        double[][] originalHigh, double[][] embeddedLow, double[][] anchors) {
        int n = originalHigh.length;
        int k = anchors.length;
        double totalError = 0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                double origDist = SuperMDSHelper.euclideanDistance(originalHigh[i], anchors[j]);
                double reconDist = SuperMDSHelper.euclideanDistance(embeddedLow[i], anchors[j]);
                double diff = origDist - reconDist;
                totalError += diff * diff;
            }
        }

        return totalError / (n * k);
    }

    /**
     * Computes the inverse stress between two high-dimensional datasets.
     * <p>
     * This metric measures how well pairwise distances are preserved between the original
     * and reconstructed (inverse-mapped) data. It is analogous to classical MDS stress,
     * but applied in reverse to evaluate the distortion introduced by the inverse mapping.
     * <p>
     * Formally, for each pair of points {@code (i, j)}, the method computes the squared difference
     * between the Euclidean distance in the original space and the reconstructed space:
     * <pre>
     * stress = Σ_{i < j} (||original_i - original_j|| - ||reconstructed_i - reconstructed_j||)^2
     * </pre>
     *
     * @param original      The original high-dimensional data array (n × d).
     * @param reconstructed The reconstructed data array after inverse mapping (n × d).
     * @return The total inverse stress value as a double.
     */
    public static double inverseStress(double[][] original, double[][] reconstructed) {
        int n = original.length;
        double totalStress = 0;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double origDist = SuperMDSHelper.euclideanDistance(original[i], original[j]);
                double reconDist = SuperMDSHelper.euclideanDistance(reconstructed[i], reconstructed[j]);
                double diff = origDist - reconDist;
                totalStress += diff * diff;
            }
        }

        return totalStress / (n * (n - 1) / 2.0);
    }

    /**
     * Evaluates the quality of an inverse MDS approximation by computing a set of validation metrics:
     * <ul>
     *     <li><b>Mean Reconstruction Error</b>: Measures the average squared distance between the original
     *         high-dimensional vectors and the reconstructed ones.</li>
     *     <li><b>Round-Trip Anchor Consistency</b>: Compares distances from each point to anchor points
     *         before and after the inverse mapping, measuring distortion introduced by the round trip.</li>
     *     <li><b>Inverse Stress</b>: A stress-like metric computed over all pairwise distances between points
     *         in the original and reconstructed high-dimensional spaces.</li>
     * </ul>
     *
     * @param original      The original high-dimensional data (n × d).
     * @param reconstructed The inverse-mapped approximation of the original data (n × d).
     * @param anchors       The subset of reference points (k × d) used for round-trip consistency evaluation.
     * @return A {@link ValidationResults} object containing all three validation metrics.
     */
    public static ValidationResults validateInversion(
        double[][] original, double[][] reconstructed, double[][] anchors) {

        ValidationResults results = new ValidationResults();
        results.meanReconstructionError = meanReconstructionError(original, reconstructed);
        results.roundTripAnchorConsistency = roundTripAnchorConsistency(original, reconstructed, anchors);
        results.inverseStress = inverseStress(original, reconstructed);

        return results;
    }

    /**
     * Helper class to store validation results.
     */
    public static class ValidationResults {
        public double meanReconstructionError;
        public double roundTripAnchorConsistency;
        public double inverseStress;

        @Override
        public String toString() {
            return String.format(
                "Reconstruction Error: %.6f\nRound-Trip Anchor Consistency: %.6f\nInverse Stress: %.6f",
                meanReconstructionError, roundTripAnchorConsistency, inverseStress
            );
        }
    }

    public static void runPseudoinverseInversionSanityCheck(
        double[][] highDimPoints,        // Original high-D data (N x D)
        double[][] lowDimEmbeddings,     // Embedded points from SMACOF (N x d)
        int[] anchorIndices              // Indices used as anchors
    ) {
        // Step 1: Extract anchors
        double[][] anchorsHD = Arrays.stream(anchorIndices).mapToObj(i -> highDimPoints[i]).toArray(double[][]::new);
        double[][] anchorsLD = Arrays.stream(anchorIndices).mapToObj(i -> lowDimEmbeddings[i]).toArray(double[][]::new);

        // Step 2: Invert all points using pseudoinverse
        double[][] recovered = SuperMDSInverter.invertViaPseudoinverse(
            anchorsHD, anchorsLD, lowDimEmbeddings, 1e-8
        );

        // Step 3: Evaluate reconstruction error
        double totalError = 0.0;
        double maxError = 0.0;
        for (int i = 0; i < highDimPoints.length; i++) {
            double[] x_orig = highDimPoints[i];
            double[] x_rec = recovered[i];
            double error = euclideanDistance(x_orig, x_rec);
            totalError += error * error;
            maxError = Math.max(maxError, error);
        }

        double mse = totalError / highDimPoints.length;
        double rmse = Math.sqrt(mse);

        LOG.info("=== Landmark-Based Inversion Sanity Check ===");
        LOG.info("RMSE: {}", String.format("%.6f", rmse));
        LOG.info("Max Error: {}", String.format("%.6f", maxError));
    }

    /**
     * Runs a multilateration sanity check by attempting to invert the embedding
     * of one or more known anchor points and comparing the result to the original.
     *
     * @param anchors       High-dimensional anchor points (shape: k × D)
     * @param embedded      Corresponding low-dimensional embeddings (shape: k × d)
     * @param config        Inversion configuration
     * @param indicesToTest List of anchor indices to test (e.g., List.of(0, 1, 2))
     */
    public static void runMultilaterationSanityCheck(
        double[][] anchors,
        double[][] embedded,
        MultilaterationConfig config,
        List<Integer> indicesToTest
    ) {
        LOG.info("=== Sanity Check: Multilateration Inversion ===");
        for (int idx : indicesToTest) {
            LOG.info("--- Anchor Index: {} ---", idx);

            double[] x_orig = anchors[idx];
            double[] x_emb = embedded[idx];

            LOG.info("Original anchor (x_orig): {}", Arrays.toString(x_orig));
            LOG.info("Embedded anchor (x_emb): {}", Arrays.toString(x_emb));

            double[] x_recovered = SuperMDSInverter.invertViaMultilateration(
                anchors, embedded, x_emb, config
            );

            LOG.info("Recovered (x_rec):       {}", Arrays.toString(x_recovered));
            double error = SuperMDSHelper.euclideanDistance(x_orig, x_recovered);
            LOG.info("Euclidean Error:         {}", String.format("%.6f", error));
        }
    }

    /**
     * Convenience overload to test just a single index.
     */
    public static void runMultilaterationSanityCheck(
        double[][] anchors,
        double[][] embedded,
        MultilaterationConfig config,
        int index
    ) {
        runMultilaterationSanityCheck(anchors, embedded, config, List.of(index));
    }

    public static double maxDistanceError(double[][] D, double[][] reconstructed) {
        int n = D.length;
        double maxError = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double error = Math.abs(D[i][j] - reconstructed[i][j]);
                maxError = Math.max(maxError, error);
            }
        }
        return maxError;
    }

    public static double meanSquaredError(double[][] D, double[][] reconstructed) {
        int n = D.length;
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double diff = D[i][j] - reconstructed[i][j];
                sum += diff * diff;
                count++;
            }
        }
        return sum / count;
    }

    public static double rawStress(double[][] D, double[][] reconstructed, double[][] weights) {
        int n = D.length;
        double stress = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double diff = reconstructed[i][j] - D[i][j];
                double w = (weights != null) ? weights[i][j] : 1.0;
                stress += w * diff * diff;
            }
        }
        return stress;
    }

    public static double[][] generateSphereData(int numPoints, int embedDim, int seed) {
        double[][] data = new double[numPoints][embedDim];
        Random rand = new Random(seed);
        for (int i = 0; i < numPoints; i++) {
            double theta = 2 * Math.PI * i / numPoints;
            double phi = Math.acos(2 * rand.nextDouble() - 1);

            double x = Math.sin(phi) * Math.cos(theta);
            double y = Math.sin(phi) * Math.sin(theta);
            double z = Math.cos(phi);

            data[i][0] = x;
            data[i][1] = y;
            data[i][2] = z;

            // Add small noise in extra dimensions
            for (int j = 3; j < embedDim; j++) {
                data[i][j] = 0.01 * rand.nextGaussian();
            }
        }
        return data;
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
}
