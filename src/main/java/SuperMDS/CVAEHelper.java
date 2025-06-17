package SuperMDS;

import java.util.Arrays;
import java.util.Random;

/**
 * Captures various vector and matrix math operations that are static.
 * @author Sean Phillips
 */
public class CVAEHelper {
    private static Random rand = new Random();
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
     * Initialize vector with zeros
     */
    public static double[] initVector(int len) {
        return new double[len]; // Java initializes to 0.0
    }

    /**
     * Initializes a weight matrix for a neural network layer using either 
     * Xavier Glorot initialization (for tanh/linear activations) or 
     * He initialization (for ReLU/LeakyReLU activations).
     * 
     * @param in       The number of input neurons (fan-in).
     * @param out      The number of output neurons (fan-out).
     * @param forReLU  If true, uses He initialization (recommended for ReLU activations).
     *                 If false, uses Xavier Glorot initialization (recommended for tanh or linear activations).
     * @return A 2D array representing the initialized weight matrix.
     */
    public static double[][] initMatrix(int in, int out, boolean forReLU) {
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
  
    
// Gradient descent update: W += -lr * outer(input, grad)
    public static void updateMatrix(double[][] W, double[] input, double[] grad, double lr) {
        if (W.length != input.length) {
            throw new IllegalArgumentException("W.rows = " + W.length + " but input.length = " + input.length);
        }
        if (W[0].length != grad.length) {
            throw new IllegalArgumentException("W.cols = " + W[0].length + " but grad.length = " + grad.length);
        }

        for (int i = 0; i < W.length; i++) {
            for (int j = 0; j < W[i].length; j++) {
                W[i][j] -= lr * input[i] * grad[j];
            }
        }
    }

// Gradient descent update: b += -lr * grad
    public static void updateVector(double[] b, double[] grad, double lr) {
        for (int i = 0; i < b.length; i++) {
            b[i] -= lr * grad[i];
        }
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
     * Compute matrix product x * W
     *
     * @param x Vector of length n
     * @param W Matrix n x m
     * @return Result vector length m
     */
    public static double[] dot(double[] x, double[][] W) {
        if (x.length != W.length) {
            throw new IllegalArgumentException("dot(): Dimension mismatch: x.length=" + x.length + ", W.length=" + W.length);
        }
        int outDim = W[0].length;
        double[] out = new double[outDim];
        for (int j = 0; j < outDim; j++) {
            for (int i = 0; i < x.length; i++) {
                out[j] += x[i] * W[i][j];
            }
            if (Double.isNaN(out[j]) || Double.isInfinite(out[j])) {
                System.err.println("Bad dot(): x=" + Arrays.toString(x));
                System.err.println("W[0]=" + Arrays.toString(W[0]));
                throw new RuntimeException("dot(): NaN or Inf in output at index " + j);
            }
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
        if (clipVal <= 0.0) return;

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
    
//    /** Utility: Mean squared error between two vectors */
//    private static double mseLoss(double[] a, double[] b) {
//        double sum = 0;
//        for (int i = 0; i < a.length; i++) {
//            double d = a[i] - b[i];
//            sum += d * d;
//        }
//        return sum / a.length;
//    }    
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
    /** Utility: Generate random Gaussian data */
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
