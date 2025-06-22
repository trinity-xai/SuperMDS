package SuperMDS;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;


/**
 * Provides variations and combinations of hread-safe, pre-buffered SIMD 
 * operations that mimic the traditional SIMD steps for a CVAE.
 * @author Sean Phillips
 */
public class FusedSIMDMath {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    /**
     * Fused dot(x, W) + b + ReLU with layout: W[input][output]
     */
    public static void fusedDotAddReluSIMD(double[] x, double[][] W, double[] b, double[] out) {
        int inputDim = x.length;
        int outputDim = b.length;

        for (int j = 0; j < outputDim; j++) {
            double sum = 0.0;
            int i = 0;
            int limit = SPECIES.loopBound(inputDim);
            for (; i < limit; i += SPECIES.length()) {
                var xVec = DoubleVector.fromArray(SPECIES, x, i);
                var wVec = DoubleVector.fromArray(SPECIES, W[i], j);
                sum += xVec.mul(wVec).reduceLanes(VectorOperators.ADD);
            }
            for (; i < inputDim; i++) {
                sum += x[i] * W[i][j];
            }

            out[j] = Math.max(0.0, sum + b[j]);
        }
    }

    /**
     * Fused affine transformation + ReLU with layout: W[output][input]
     */
    public static void fusedAffineSIMD(double[] input, double[][] weights, double[] bias, double[] output) {
        int rows = weights.length;
        int cols = input.length;

        for (int row = 0; row < rows; row++) {
            double[] wRow = weights[row];
            double sum = 0.0;
            int i = 0;
            int limit = SPECIES.loopBound(cols);
            for (; i < limit; i += SPECIES.length()) {
                var xVec = DoubleVector.fromArray(SPECIES, input, i);
                var wVec = DoubleVector.fromArray(SPECIES, wRow, i);
                sum += xVec.mul(wVec).reduceLanes(VectorOperators.ADD);
            }
            for (; i < cols; i++) {
                sum += input[i] * wRow[i];
            }
            output[row] = Math.max(0.0, sum + bias[row]);
        }
    }

    /**
     * Fused affine transformation + dropout + ReLU
     */
    public static void fusedAffineDropoutSIMD(double[] input, double[][] weights, double[] bias,
                                              double[] output, SIMDRandomBuffer randBuffer,
                                              double dropoutRate) {
        int rows = weights.length;
        int cols = input.length;
        double scale = 1.0 / (1.0 - dropoutRate);
        double[] rand = new double[rows];
        randBuffer.nextUniformBatch(rand);

        for (int row = 0; row < rows; row++) {
            double[] wRow = weights[row];
            double sum = 0.0;
            int i = 0;
            int limit = SPECIES.loopBound(cols);
            for (; i < limit; i += SPECIES.length()) {
                var xVec = DoubleVector.fromArray(SPECIES, input, i);
                var wVec = DoubleVector.fromArray(SPECIES, wRow, i);
                sum += xVec.mul(wVec).reduceLanes(VectorOperators.ADD);
            }
            for (; i < cols; i++) {
                sum += input[i] * wRow[i];
            }

            double preAct = sum + bias[row];
            boolean keep = rand[row] >= dropoutRate;
            output[row] = keep ? Math.max(0.0, preAct * scale) : 0.0;
        }
    }

    // Add additional fused ops as needed (e.g., fusedAffineSoftmax, fusedAffineTanh, etc.)
}
