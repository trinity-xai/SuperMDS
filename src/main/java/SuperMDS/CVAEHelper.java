package SuperMDS;

import java.util.Arrays;
import java.util.Random;

/**
 * Captures various vector and matrix math operations that are static.
 *
 * @author Sean Phillips
 */
public class CVAEHelper {

    /**
     * Compute the current KL annealing weight using a sigmoid ramp-up schedule.
     * This method gradually increases the weight of the KL divergence term
     * during training to help the model focus on reconstruction early on and
     * avoid posterior collapse.
     *
     * The sigmoid curve is scaled and shifted to flatten the early growth and
     * reach a configurable maximum weight (e.g. 0.75) near the end of the
     * ramp-up period.
     *
     * @param epoch Current training epoch
     * @param klWarmupEpochs Number of epochs over which to ramp up the KL
     * weight
     * @param maxKLWeight The maximum KL weight (recommended: 0.5–1.0)
     * @param sharpness Controls how steep the sigmoid ramp is (recommended:
     * 6–10)
     * @return The KL weight (between 0.0 and maxKLWeight)
     */
    public static double getKLWeight(int epoch, int klWarmupEpochs, double maxKLWeight, double sharpness) {
        // Ensure progress is in [0.0, 1.0]
        double progress = Math.min(1.0, Math.max(0.0, (double) epoch / klWarmupEpochs));

        // Center the sigmoid at halfway point of ramp-up
        double x = sharpness * (progress - 0.5);

        // Standard sigmoid function
        double sigmoid = 1.0 / (1.0 + Math.exp(-x));

        // Scale sigmoid output to the desired max KL weight
        return maxKLWeight * sigmoid;
    }

    /**
     * Computes a cyclical KL divergence weight using a sigmoid ramp within each
     * cycle. This method allows the KL weight to rise smoothly from 0 to
     * maxKLWeight during each cycle, encouraging the model to periodically use
     * the latent space more fully (helps avoid posterior collapse).
     *
     * @param epoch The current training epoch (non-negative integer).
     * @param cycleLength The number of epochs in one annealing cycle (e.g.,
     * 200).
     * @param maxKLWeight The maximum KL weight to reach at the peak of each
     * cycle (e.g., 1.0).
     * @param sharpness Controls the steepness of the sigmoid curve (typical
     * range: 5 to 20).
     * @return The KL weight for the current epoch, smoothly ranging from 0 to
     * maxKLWeight.
     */
    public static double getCyclicalKLWeightSigmoid(int epoch, int cycleLength, double maxKLWeight, double sharpness) {
        // Compute how far we are through the current cycle: a value in [0.0, 1.0]
        double cycleProgress = (epoch % cycleLength) / (double) cycleLength;

        // Center the sigmoid curve at the midpoint of the cycle (i.e., 0.5)
        // Sharpness controls how quickly the weight increases near the midpoint
        double x = sharpness * (cycleProgress - 0.5);

        // Standard sigmoid function maps x into [0.0, 1.0]
        double sigmoid = 1.0 / (1.0 + Math.exp(-x));

        // Scale by the max desired KL weight
        return maxKLWeight * sigmoid;
    }

    /**
     * Initialize vector with zeros
     */
    public static double[] initVector(int len) {
        return new double[len]; // Java initializes to 0.0
    }

    /**
     * Initializes a weight matrix for a neural network layer using either
     * Xavier Glorot initialization (for tanh/linear activations) or He
     * initialization (for ReLU/LeakyReLU activations).
     *
     * @param in The number of input neurons (fan-in).
     * @param out The number of output neurons (fan-out).
     * @param forReLU If true, uses He initialization (recommended for ReLU
     * activations). If false, uses Xavier Glorot initialization (recommended
     * for tanh or linear activations).
     * @param rand
     * @return A 2D array representing the initialized weight matrix.
     */
    public static double[][] initMatrix(int in, int out, boolean forReLU, Random rand) {
        double[][] mat = new double[in][out];

        if (forReLU) {
            // He initialization uses a normal distribution with standard deviation sqrt(2 / fan_in)
            double std = Math.sqrt(2.0 / in);
            for (int i = 0; i < in; i++) {
                for (int j = 0; j < out; j++) {
                    // Draw from Gaussian distribution centered at 0
                    mat[i][j] = rand.nextGaussian() * std;
                }
            }
        } else {
            // Xavier Glorot initialization uses a uniform distribution in [-limit, limit]
            double limit = Math.sqrt(6.0 / (in + out));
            for (int i = 0; i < in; i++) {
                for (int j = 0; j < out; j++) {
                    // Draw from uniform distribution between -limit and +limit
                    mat[i][j] = rand.nextDouble() * 2 * limit - limit;
                }
            }
        }

        return mat;
    }

    /**
     * Gradient descent update for weight matrix: W[i][j] -= lr * grad[i] *
     * input[j]
     *
     * W: [outputDim][inputDim] grad: gradient w.r.t. output layer activations
     * (length = outputDim) input: input vector to the layer (length = inputDim)
     */
    public static void updateMatrix(double[][] W, double[] input, double[] grad, double lr) {
        int outputDim = W.length;
        int inputDim = W[0].length;

        if (grad.length != outputDim) {
            throw new IllegalArgumentException("grad.length = " + grad.length + " but W.rows = " + outputDim);
        }
        if (input.length != inputDim) {
            throw new IllegalArgumentException("input.length = " + input.length + " but W.cols = " + inputDim);
        }

        double gi = 0;
        double scale = 0;
        for (int i = 0; i < outputDim; i++) {
            gi = grad[i];
            scale = lr * gi;
            for (int j = 0; j < inputDim; j++) {
                W[i][j] -= scale * input[j];
            }
        }
    }

    // Gradient descent update: b += -lr * grad
    public static void updateVector(double[] b, double[] grad, double lr) {
        for (int i = 0; i < b.length; i++) {
            b[i] -= lr * grad[i];
        }
    }

    public static double[] applyDropout(double[] input, double dropoutRate, Random rng) {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            //output[i] = (rng.nextDouble() < dropoutRate) ? 0.0 : input[i];
            output[i] = (rng.nextDouble() < dropoutRate) ? 0.0 : input[i] / (1.0 - dropoutRate);
        }
        return output;
    }

    public static int[] shuffledIndices(int n, Random rand) {
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        return indices;
    }

    /**
     * Utility function to detect NaNs or Infs in a vector.
     */
    public static boolean hasNaNsOrInfs(double[] vec) {
        for (double v : vec) {
            if (Double.isNaN(v) || Double.isInfinite(v)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Sample latent vector z from mu and log variance using reparameterization
     * trick.
     *
     * @param mu Mean vector of latent distribution
     * @param logvar Log variance vector of latent distribution
     * @param rand
     * @return Sampled latent vector z
     */
    public static double[] sampleLatent(double[] mu, double[] logvar, Random rand) {
        double[] z = new double[mu.length];
        for (int i = 0; i < z.length; i++) {
            double eps = rand.nextGaussian(); // Standard normal noise
            z[i] = mu[i] + Math.exp(0.5 * logvar[i]) * eps;
        }
        return z;
    }
// In place operations variants    

    public static void reluInPlace(double[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = Math.max(0.0, Math.min(50.0, x[i]));
        }
    }

    public static void addMatrixInPlace(double[][] target, double[][] src) {
        for (int i = 0; i < target.length; i++) {
            for (int j = 0; j < target[i].length; j++) {
                target[i][j] += src[i][j];
            }
        }
    }

    public static void addVectorInPlace(double[] target, double[] src) {
        for (int i = 0; i < target.length; i++) {
            target[i] += src[i];
        }
    }

    public static void addInPlace(double[] a, double[] b) {
        for (int i = 0; i < a.length; i++) {
            a[i] += b[i];
        }
    }

    public static void addInPlace(double[] a, double[] b, double[] out) {
        if (a.length != b.length || a.length != out.length) {
            throw new IllegalArgumentException("Dimension mismatch in addInPlace");
        }

        for (int i = 0; i < a.length; i++) {
            out[i] = a[i] + b[i];
            if (Double.isNaN(out[i]) || Double.isInfinite(out[i])) {
                throw new RuntimeException("addInPlace: NaN or Inf at index " + i);
            }
        }
    }

    /**
     * Computes out = A · x where A is [outDim][inDim] and x is [inDim]. Result
     * is written into out[] of length outDim.
     */
    public static void dot(double[] x, double[][] A, double[] out) {
        int outDim = A.length;
        int inDim = x.length;
        for (int i = 0; i < outDim; i++) {
            double sum = 0.0;
            for (int j = 0; j < inDim; j++) {
                sum += A[i][j] * x[j];
            }
            out[i] = sum;
        }
    }

    /**
     * Computes out = Aᵗ · x where A is [outDim][inDim] and x is [outDim].
     * Result is written into out[] of length inDim.
     */
    public static void dotT(double[] x, double[][] A, double[] out) {
        int outDim = A.length;       // rows of A
        int inDim = A[0].length;     // columns of A

        Arrays.fill(out, 0.0);       // important: zero out accumulator

        for (int i = 0; i < outDim; i++) {
            double xi = x[i];
            for (int j = 0; j < inDim; j++) {
                out[j] += A[i][j] * xi;
            }
        }
    }

    public static void reluGradInPlace(double[] output, double[] upstream, double[] out) {
        for (int i = 0; i < output.length; i++) {
            out[i] = output[i] > 0.0 ? upstream[i] : 0.0;
        }
    }

    /**
     * Computes dL/dPred = (2 / n) * (pred - target), stored in out[]. Assumes
     * pred.length == target.length == out.length.
     */
    public static void mseGradient(double[] pred, double[] target, double[] out) {
        int n = pred.length;
        double scale = 2.0 / n;
        for (int i = 0; i < n; i++) {
            out[i] = scale * (pred[i] - target[i]);
        }
    }

    public static void concatInPlace(double[] a, double[] b, double[] out) {
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
    }

    public static void clipGradientInPlace(double[] grad, double clipVal) {
        if (clipVal <= 0.0) {
            return;
        }
        double norm = 0.0;
        for (double v : grad) {
            norm += v * v;
        }
        norm = Math.sqrt(norm);
        if (norm > clipVal) {
            double scale = clipVal / norm;
            for (int i = 0; i < grad.length; i++) {
                grad[i] *= scale;
            }
        }
    }

    public static void sampleLatentInPlace(double[] mu, double[] logvar, double[] out, Random rng) {
        if (mu.length != logvar.length || mu.length != out.length) {
            throw new IllegalArgumentException("Dimension mismatch in sampleLatentInPlace");
        }

        for (int i = 0; i < mu.length; i++) {
            double std = Math.exp(0.5 * logvar[i]);
            double eps = rng.nextGaussian(); // standard normal sample
            out[i] = mu[i] + std * eps;
        }
    }

    public static void applyDropoutInPlace(double[] x, double rate, Random rng) {
        if (rate <= 0.0) {
            return; // No-op if dropout is disabled
        }
        double scale = 1.0 / (1.0 - rate);
        for (int i = 0; i < x.length; i++) {
            x[i] = rng.nextDouble() < rate ? 0.0 : x[i] * scale;
        }
    }
// Accumulate gradient of outer product into gradMatrix

    public static void accumulateOuterProduct(double[][] gradMatrix, double[] input, double[] gradOut) {
        for (int i = 0; i < input.length; i++) {
            double inVal = input[i];
            for (int j = 0; j < gradOut.length; j++) {
                gradMatrix[i][j] += inVal * gradOut[j];
            }
        }
    }

    public static void accumulateMatrix(double[][] target, double[][] source) {
        for (int i = 0; i < target.length; i++) {
            for (int j = 0; j < target[i].length; j++) {
                target[i][j] += source[i][j];
            }
        }
    }

    public static void accumulateVector(double[] target, double[] source) {
        for (int i = 0; i < target.length; i++) {
            target[i] += source[i];
        }
    }

// Apply the accumulated gradients once
    public static void applyUpdate(double[][] W, double[][] gradW, double lr) {
        for (int i = 0; i < W.length; i++) {
            for (int j = 0; j < W[i].length; j++) {
                W[i][j] -= lr * gradW[i][j];
            }
        }
    }

    public static void applyUpdate(double[] b, double[] gradB, double lr) {
        for (int i = 0; i < b.length; i++) {
            b[i] -= lr * gradB[i];
        }
    }

    public static void accumulateBias(double[] gradB, double[] gradOut) {
        for (int i = 0; i < gradB.length; i++) {
            gradB[i] += gradOut[i];
        }
    }

    public static void scaleMatrixInPlace(double[][] mat, double scale) {
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[i].length; j++) {
                mat[i][j] *= scale;
            }
        }
    }

    public static void scaleVectorInPlace(double[] vec, double scale) {
        for (int i = 0; i < vec.length; i++) {
            vec[i] *= scale;
        }
    }

    /**
     * Compute the gradient of the ReLU activation function using the output as
     * a mask. This is the elementwise product of the upstream gradient and the
     * ReLU derivative.
     *
     * @param output The output of the ReLU function (i.e.,
     * ReLU(pre-activation))
     * @param upstream The gradient flowing from the next layer
     * @return Gradient to backpropagate to the previous layer
     */
    public static double[] reluGrad(double[] output, double[] upstream) {
        double[] grad = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            grad[i] = output[i] > 0.0 ? upstream[i] : 0.0;
        }
        return grad;
    }

    public static double[] relu(double[] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            if (Double.isNaN(x[i]) || Double.isInfinite(x[i])) {
                throw new RuntimeException("ReLU input contains NaN/Inf at index " + i + ": " + x[i]);
            }
            out[i] = Math.max(0.0, Math.min(50.0, x[i]));  // Clamp to [0, 50]
        }
        return out;
    }

    /**
     * Element-wise vector addition
     */
    public static double[] add(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("add(): Dimension mismatch " + a.length + " vs " + b.length);
        }
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i] + b[i];
            if (Double.isNaN(out[i]) || Double.isInfinite(out[i])) {
                throw new RuntimeException("add(): NaN or Inf at index " + i);
            }
        }
        return out;
    }

    /**
     * Compute matrix product y = W · x, where W is [out][in], x is [in] Returns
     * y: [out]
     */
    public static double[] dot(double[] x, double[][] W) {
        if (x.length != W[0].length) {
            throw new IllegalArgumentException("dot(): x.length=" + x.length + " but W.cols=" + W[0].length);
        }
        int outDim = W.length;
        int inDim = x.length;
        double[] out = new double[outDim];
        for (int i = 0; i < outDim; i++) {
            double sum = 0.0;
            for (int j = 0; j < inDim; j++) {
                sum += W[i][j] * x[j];
            }
            out[i] = sum;
        }
        return out;
    }

    /**
     * Compute dot product of gradient vector dy with transpose of weight matrix
     * W. Used for backpropagation.
     *
     * @param dy Gradient vector of length m
     * @param W Weight matrix n x m
     * @return Gradient vector length n
     */
    public static double[] dotT(double[] dy, double[][] W) {
        if (dy.length != W[0].length) {
            throw new IllegalArgumentException("dotT(): Dimension mismatch: dy.length=" + dy.length + ", W[0].length=" + W[0].length);
        }
        int inDim = W.length;
        double[] out = new double[inDim];
        for (int i = 0; i < inDim; i++) {
            for (int j = 0; j < dy.length; j++) {
                out[i] += dy[j] * W[i][j];
            }
            if (Double.isNaN(out[i]) || Double.isInfinite(out[i])) {
                throw new RuntimeException("dotT(): NaN or Inf in output at index " + i);
            }
        }
        return out;
    }

    /**
     * Concatenate two vectors
     */
    public static double[] concat(double[] a, double[] b) {
        double[] r = new double[a.length + b.length];
        System.arraycopy(a, 0, r, 0, a.length);
        System.arraycopy(b, 0, r, a.length, b.length);
        return r;
    }

    /**
     * Slice subvector from array starting at 'start' of length 'len'
     */
    public static double[] slice(double[] arr, int start, int len) {
        if (start < 0 || start + len > arr.length) {
            throw new IllegalArgumentException("slice(): Invalid range: start=" + start + ", len=" + len + ", arr.length=" + arr.length);
        }
        double[] r = new double[len];
        System.arraycopy(arr, start, r, 0, len);
        return r;
    }

    /**
     * Clip gradient vector to have at most L2 norm = clipVal
     */
    public static void clipGradient(double[] grad, double clipVal) {
        if (clipVal <= 0.0) {
            return;
        }

        double norm = 0.0;
        for (double g : grad) {
            if (Double.isNaN(g) || Double.isInfinite(g)) {
                throw new RuntimeException("clipGradient(): NaN/Inf in gradient: " + g);
            }
            norm += g * g;
        }

        norm = Math.sqrt(norm);
        if (norm > clipVal) {
            double scale = clipVal / norm;
            for (int i = 0; i < grad.length; i++) {
                grad[i] *= scale;
            }
        }
    }

    public static void assertShape(double[][] matrix, int expectedRows, int expectedCols, String name) {
        if (matrix.length != expectedRows || matrix[0].length != expectedCols) {
            throw new IllegalArgumentException(
                    String.format("Shape mismatch in %s: expected [%d][%d], found [%d][%d]",
                            name, expectedRows, expectedCols, matrix.length, matrix[0].length));
        }
    }

    public static void assertLength(double[] vector, int expectedLength, String name) {
        if (vector.length != expectedLength) {
            throw new IllegalArgumentException(
                    String.format("Length mismatch in %s: expected %d, found %d",
                            name, expectedLength, vector.length));
        }
    }

    /**
     * Compute mean squared error loss between vectors a and b
     */
    public static double mseLoss(double[] target, double[] predicted) {
        if (target.length != predicted.length) {
            throw new IllegalArgumentException("mseLoss(): Dimension mismatch: target.length=" + target.length + ", predicted.length=" + predicted.length);
        }

        double sum = 0.0;
        for (int i = 0; i < target.length; i++) {
            double diff = predicted[i] - target[i];
            if (Double.isNaN(diff) || Double.isInfinite(diff)) {
                throw new RuntimeException("mseLoss(): NaN/Inf at index " + i + ": predicted=" + predicted[i] + ", target=" + target[i]);
            }
            sum += diff * diff;
        }
        return sum / target.length;
    }

    /**
     * Compute gradient of MSE loss w.r.t predicted output
     */
    public static double[] mseGradient(double[] predicted, double[] target) {
        if (predicted.length != target.length) {
            throw new IllegalArgumentException("mseGradient(): Dimension mismatch: predicted.length=" + predicted.length + ", target.length=" + target.length);
        }

        double[] grad = new double[predicted.length];
        for (int i = 0; i < predicted.length; i++) {
            grad[i] = 2.0 * (predicted[i] - target[i]) / predicted.length;
            if (Double.isNaN(grad[i]) || Double.isInfinite(grad[i])) {
                throw new RuntimeException("mseGradient(): NaN/Inf at index " + i + ": grad=" + grad[i]);
            }
        }
        return grad;
    }

    /**
     * Utility: Generate random Gaussian data
     */
    public static double[][] generateRandomData(int rows, int cols) {
        double[][] data = new double[rows][cols];
        java.util.Random rand = new java.util.Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = rand.nextGaussian(); // standard normal distribution
            }
        }
        return data;
    }
}
