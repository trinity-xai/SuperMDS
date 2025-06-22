package SuperMDS;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 *
 * @author Sean Phillips
 */
public class SIMDMath {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    public static void reluInPlaceSIMD(double[] x) {
        int i = 0;
        int length = x.length;

        while (i < SPECIES.loopBound(length)) {
            var v = DoubleVector.fromArray(SPECIES, x, i);
            var relu = v.max(0.0); // ReLU = max(x, 0)
            relu.intoArray(x, i);
            i += SPECIES.length();
        }

        for (; i < length; i++) {
            x[i] = Math.max(0.0, x[i]);
        }
    }
    /**
     * Applies elementwise gradient of ReLU: grad[i] = (output[i] > 0) ?
     * upstream[i] : 0 Stores the result in the output array 'grad' (can be same
     * as upstream).
     *
     * @param output The original output from ReLU (used as a mask)
     * @param upstream The upstream gradient
     * @param grad The output gradient buffer (overwritten)
     */
    public static void reluGradInPlaceSIMD(double[] output, double[] upstream, double[] grad) {
        int len = output.length;
        int i = 0;
        int limit = SPECIES.loopBound(len);

        for (; i < limit; i += SPECIES.length()) {
            var outVec = DoubleVector.fromArray(SPECIES, output, i);
            var upVec = DoubleVector.fromArray(SPECIES, upstream, i);
            var mask = outVec.compare(VectorOperators.GT, 0.0);
            upVec.intoArray(grad, i, mask);
            DoubleVector.zero(SPECIES).intoArray(grad, i, mask.not());
        }

        // Tail loop
        for (; i < len; i++) {
            grad[i] = output[i] > 0.0 ? upstream[i] : 0.0;
        }
    }
    public static void sampleLatentInPlaceSIMD(double[] mu, double[] logvar, double[] z, SIMDRandomBuffer rng) {
        int len = mu.length;
        double[] eps = new double[len];
        rng.nextGaussianBatch(eps); // Fill with N(0,1) samples

        int i = 0;
        int limit = SPECIES.loopBound(len);
        for (; i < limit; i += SPECIES.length()) {
            var muVec = DoubleVector.fromArray(SPECIES, mu, i);
            var logvarVec = DoubleVector.fromArray(SPECIES, logvar, i);
            var epsVec = DoubleVector.fromArray(SPECIES, eps, i);

            var stdVec = logvarVec.mul(0.5).lanewise(VectorOperators.EXP);
            var zVec = muVec.add(stdVec.mul(epsVec));

            zVec.intoArray(z, i);
        }

        // Tail loop
        for (; i < len; i++) {
            double std = Math.exp(0.5 * logvar[i]);
            z[i] = mu[i] + std * eps[i];
        }
    }
    public static void applyDropoutInPlaceSIMD(double[] input, double[] output, double rate, SIMDRandomBuffer rng) {
        int len = input.length;
        double invKeepProb = 1.0 / (1.0 - rate);

        double[] mask = new double[len];
        rng.nextUniformBatch(mask); // Fill with [0,1) values

        int i = 0;
        int limit = SPECIES.loopBound(len);
        for (; i < limit; i += SPECIES.length()) {
            var inputVec = DoubleVector.fromArray(SPECIES, input, i);
            var randVec = DoubleVector.fromArray(SPECIES, mask, i);
            var scaledVec = inputVec.mul(invKeepProb);

            var keepMask = randVec.compare(VectorOperators.GE, rate); // keep if >= rate
            scaledVec.intoArray(output, i, keepMask);
            DoubleVector.zero(SPECIES).intoArray(output, i, keepMask.not());
        }

        // Tail loop
        for (; i < len; i++) {
            output[i] = (mask[i] >= rate) ? input[i] * invKeepProb : 0.0;
        }
    }
    /**
     * Multiply two matrices: out = A * B
     *
     * @param A Matrix A of size M x N
     * @param B Matrix B of size N x K
     * @param out Output matrix of size M x K (preallocated)
     */
    public static void matMulSIMD(double[][] A, double[][] B, double[][] out) {
        int M = A.length;
        int N = A[0].length;
        int K = B[0].length;

        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                double sum = 0.0;
                int j = 0;

                // SIMD part
                for (; j < SPECIES.loopBound(N); j += SPECIES.length()) {
                    var aVec = DoubleVector.fromArray(SPECIES, A[i], j);
                    var bVec = DoubleVector.fromArray(SPECIES, getColumn(B, k), j);
                    sum += aVec.mul(bVec).reduceLanes(VectorOperators.ADD);
                }

                // Tail
                for (; j < N; j++) {
                    sum += A[i][j] * B[j][k];
                }

                out[i][k] = sum;
            }
        }
    }
    public static void dotT_SIMD(double[] dy, double[][] W, double[] out) {
        int N = W.length;
        int M = W[0].length;

        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            int j = 0;

            while (j < SPECIES.loopBound(M)) {
                var vDy = DoubleVector.fromArray(SPECIES, dy, j);

                double[] wSlice = new double[SPECIES.length()];
                for (int k = 0; k < SPECIES.length(); k++) {
                    wSlice[k] = W[i][j + k];
                }

                var vW = DoubleVector.fromArray(SPECIES, wSlice, 0);
                sum += vDy.mul(vW).reduceLanes(VectorOperators.ADD);

                j += SPECIES.length();
            }

            for (; j < M; j++) {
                sum += dy[j] * W[i][j];
            }

            out[i] = sum;
        }
    }
    public static void mseGradient_SIMD(double[] predicted, double[] target, double[] out) {
        int len = predicted.length;
        double scale = 2.0 / len;

        int i = 0;
        while (i < SPECIES.loopBound(len)) {
            var vPred = DoubleVector.fromArray(SPECIES, predicted, i);
            var vTarg = DoubleVector.fromArray(SPECIES, target, i);
            var vGrad = vPred.sub(vTarg).mul(scale);
            vGrad.intoArray(out, i);
            i += SPECIES.length();
        }

        for (; i < len; i++) {
            out[i] = 2.0 * (predicted[i] - target[i]) / len;
        }
    }
    public static void addInPlaceSIMD(double[] a, double[] b, double[] out) {
        int i = 0;
        int length = a.length;

        while (i < SPECIES.loopBound(length)) {
            var va  = DoubleVector.fromArray(SPECIES, a, i);
            var vb = DoubleVector.fromArray(SPECIES, b, i);
            var vsum = va.add(vb);
            vsum.intoArray(out, i);
            i += SPECIES.length();
        }

        for (; i < length; i++) {
            out[i] = a[i] + b[i];
        }
    }
    public static void dotSIMD(double[] x, double[][] W, double[] out) {
        int rows = W.length;
        int cols = W[0].length;

        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            int i = 0;

            while (i < SPECIES.loopBound(rows)) {
                var vx = DoubleVector.fromArray(SPECIES, x, i);

                double[] wColSlice = new double[SPECIES.length()];
                for (int k = 0; k < SPECIES.length(); k++) {
                    wColSlice[k] = W[i + k][j];
                }

                var vw = DoubleVector.fromArray(SPECIES, wColSlice, 0);
                sum += vx.mul(vw).reduceLanes(VectorOperators.ADD);
                i += SPECIES.length();
            }

            for (; i < rows; i++) {
                sum += x[i] * W[i][j];
            }

            out[j] = sum;
        }
    }
    /**
     * Computes the dot product of dy with the transpose of W and stores the
     * result in 'out'. Equivalent to: out = W^T * dy
     *
     * @param dy Vector of length M
     * @param W Weight matrix of size N x M
     * @param out Output vector of length N (overwritten)
     */
    public static void dotTInPlaceSIMD(double[] dy, double[][] W, double[] out) {
        int N = W.length;
        int M = dy.length;

        for (int i = 0; i < N; i++) {
            double[] row = W[i];
            double sum = 0.0;

            int j = 0;
            int limit = SPECIES.loopBound(M);
            for (; j < limit; j += SPECIES.length()) {
                var wVec = DoubleVector.fromArray(SPECIES, row, j);
                var dyVec = DoubleVector.fromArray(SPECIES, dy, j);
                sum += wVec.mul(dyVec).reduceLanes(VectorOperators.ADD);
            }

            // Tail loop
            for (; j < M; j++) {
                sum += row[j] * dy[j];
            }

            out[i] = sum;
        }
    }
    public static void clipGradient_SIMD(double[] grad, double clipVal) {
        if (clipVal <= 0.0) {
            return;
        }

        int len = grad.length;
        double normSquared = 0.0;

        int i = 0;
        while (i < SPECIES.loopBound(len)) {
            var v = DoubleVector.fromArray(SPECIES, grad, i);
            normSquared += v.mul(v).reduceLanes(VectorOperators.ADD);
            i += SPECIES.length();
        }

        for (; i < len; i++) {
            normSquared += grad[i] * grad[i];
        }

        double norm = Math.sqrt(normSquared);
        if (norm <= clipVal) {
            return;
        }

        double scale = clipVal / norm;

        i = 0;
        while (i < SPECIES.loopBound(len)) {
            var v = DoubleVector.fromArray(SPECIES, grad, i);
            v.mul(scale).intoArray(grad, i);
            i += SPECIES.length();
        }

        for (; i < len; i++) {
            grad[i] *= scale;
        }
    }
    public static void updateVector_SIMD(double[] b, double[] grad, double lr) {
        int len = b.length;

        int i = 0;
        while (i < SPECIES.loopBound(len)) {
            var vB = DoubleVector.fromArray(SPECIES, b, i);
            var vGrad = DoubleVector.fromArray(SPECIES, grad, i);
            vB.sub(vGrad.mul(lr)).intoArray(b, i);
            i += SPECIES.length();
        }

        for (; i < len; i++) {
            b[i] -= lr * grad[i];
        }
    }
    public static void updateMatrix_SIMD(double[][] W, double[] input, double[] grad, double lr) {
        int rows = W.length;
        int cols = W[0].length;

        int jLimit = SPECIES.loopBound(cols);

        for (int i = 0; i < rows; i++) {
            double scale = lr * input[i];
            int j = 0;

            while (j < jLimit) {
                var vW = DoubleVector.fromArray(SPECIES, W[i], j);
                var vGrad = DoubleVector.fromArray(SPECIES, grad, j);
                vW.sub(vGrad.mul(scale)).intoArray(W[i], j);
                j += SPECIES.length();
            }

            for (; j < cols; j++) {
                W[i][j] -= scale * grad[j];
            }
        }
    }
    /**
     * Extract a column vector from a matrix
     */
    private static double[] getColumn(double[][] matrix, int colIndex) {
        int rows = matrix.length;
        double[] col = new double[rows];
        for (int i = 0; i < rows; i++) {
            col[i] = matrix[i][colIndex];
        }
        return col;
    }
}
