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
}
