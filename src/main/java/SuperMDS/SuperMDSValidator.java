package SuperMDS;

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
                "Goodness-of-Fit: %.6f",
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
}
