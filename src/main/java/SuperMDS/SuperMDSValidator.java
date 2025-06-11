package SuperMDS;

import java.util.Random;

/**
 *
 * @author Sean Phillips
 */
public class SuperMDSValidator {

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

    System.out.printf("Stress-1 (Kruskal): %.6f\n", stress1);
    System.out.printf("Stress-2 (Normalized raw stress): %.6f\n", stress2);
    System.out.printf("Goodness-of-Fit: %.6f  <--------------------  \n", gof);
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
