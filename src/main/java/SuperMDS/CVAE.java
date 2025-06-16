package SuperMDS;

import java.util.Arrays;
import java.util.Random;

/**
 * Conditional Variational Autoencoder (CVAE) implementation in pure Java.
 * Supports arbitrary input, condition, latent, and hidden dimensions.
 * Training uses gradient descent with backpropagation through encoder, latent sampling, and decoder.
 *
 * Activation functions:
 * - Encoder/hidden layers: tanh
 * - Output layer: linear
 *
 * Loss:
 * - Reconstruction loss: mean squared error (MSE)
 * - KL divergence loss for latent regularization
 *
 * Usage:
 * - Construct with input dimension, condition dimension, latent dimension, and hidden dimension.
 * - Call train() per sample to perform one training step.
 * 
 * @author Sean Phillips
 */
public class CVAE {
    private int inputDim;      // Dimensionality of input vector
    private int conditionDim;  // Dimensionality of condition vector
    private int latentDim;     // Dimensionality of latent space
    private int hiddenDim;     // Number of hidden units in encoder

    private double learningRate = 0.01;  // Gradient descent learning rate
    private Random rand = new Random();

    // Encoder weight matrices and biases
    private double[][] W_enc;    // Weights: (inputDim + conditionDim) x hiddenDim
    private double[] b_enc;      // Biases: hiddenDim
    private double[][] W_mu;     // Weights: hiddenDim x latentDim for mean output
    private double[] b_mu;       // Biases: latentDim
    private double[][] W_logvar; // Weights: hiddenDim x latentDim for log variance output
    private double[] b_logvar;   // Biases: latentDim

    // Decoder weight matrix and bias
    private double[][] W_dec;    // Weights: (latentDim + conditionDim) x inputDim
    private double[] b_dec;      // Biases: inputDim

    /**
     * Initialize a new CVAE instance.
     * 
     * | Parameter      | Description                                                                |
     * | -------------- | -------------------------------------------------------------------------- |
     * | `inputDim`     | The dimensionality of the **original high-dimensional data** (e.g. 128D).  |
     * | `conditionDim` | The dimensionality of the **embedding space** (e.g. 2D or 3D from SMACOF). |
     * | `latentDim`    | The size of the internal latent space for the VAE (e.g. 8–32).             |
     * | `hiddenDim`    | The number of neurons in the encoder's hidden layer (e.g. 64–256).         |
     * 
     * @param inputDim Dimensionality of input vector
     * @param conditionDim Dimensionality of conditioning vector
     * @param latentDim Dimensionality of latent space vector
     * @param hiddenDim Number of hidden units in encoder hidden layer
     */
    public CVAE(int inputDim, int conditionDim, int latentDim, int hiddenDim) {
        this.inputDim = inputDim;
        this.conditionDim = conditionDim;
        this.latentDim = latentDim;
        this.hiddenDim = hiddenDim;

        int encInputDim = inputDim + conditionDim; // Encoder input concatenated size
        int decInputDim = latentDim + conditionDim; // Decoder input concatenated size

        // Initialize encoder weights and biases with small random values
        W_enc = initMatrix(encInputDim, hiddenDim);
        b_enc = initVector(hiddenDim);

        W_mu = initMatrix(hiddenDim, latentDim);
        b_mu = initVector(latentDim);
        W_logvar = initMatrix(hiddenDim, latentDim);
        b_logvar = initVector(latentDim);

        // Initialize decoder weights and biases
        W_dec = initMatrix(decInputDim, inputDim);
        b_dec = initVector(inputDim);
    }

    /**
     * Encoder forward pass: input + condition → hidden representation
     *
     * @param x Input vector
     * @param c Condition vector
     * @return Hidden representation after tanh activation
     */
    public double[] encode(double[] x, double[] c) {
        double[] xc = concat(x, c);                 // Concatenate input and condition
        double[] h = tanh(add(dot(xc, W_enc), b_enc)); // Linear transform + bias + tanh
        return h;
    }

    /**
     * Sample latent vector z from mu and log variance using reparameterization trick.
     *
     * @param mu Mean vector of latent distribution
     * @param logvar Log variance vector of latent distribution
     * @return Sampled latent vector z
     */
    public double[] sampleLatent(double[] mu, double[] logvar) {
        double[] z = new double[mu.length];
        for (int i = 0; i < z.length; i++) {
            double eps = rand.nextGaussian(); // Standard normal noise
            z[i] = mu[i] + Math.exp(0.5 * logvar[i]) * eps;
        }
        return z;
    }

    /**
     * Decoder forward pass: latent vector + condition → reconstruction
     *
     * @param z Latent vector
     * @param c Condition vector
     * @return Reconstruction vector (linear output)
     */
    public double[] decode(double[] z, double[] c) {
        double[] zc = concat(z, c);
        return add(dot(zc, W_dec), b_dec); // Linear output layer
    }

/**
 * Perform one training step on a single (input, condition) pair.
 * Includes full forward + backward pass, with gradient clipping and numerical stability controls.
 *
 * @param x Input vector
 * @param c Condition vector
 * @return Total loss (reconstruction + KL divergence)
 */
public double train(double[] x, double[] c) {
    if (hasNaNsOrInfs(x) || hasNaNsOrInfs(c)) {
        throw new IllegalArgumentException("Input or condition vector contains NaNs or Infs.");
    }

    double[] xc = concat(x, c);

    // ----- Forward Pass -----
    double[] h = tanh(add(dot(xc, W_enc), b_enc));
    double[] mu = add(dot(h, W_mu), b_mu);
    double[] logvar = add(dot(h, W_logvar), b_logvar);

    // Clamp logvar between [-10, 10] to avoid exp() overflow
    double[] safeLogvar = new double[latentDim];
    for (int i = 0; i < latentDim; i++) {
        safeLogvar[i] = Math.max(Math.min(logvar[i], 10.0), -10.0);
    }

    double[] z = sampleLatent(mu, safeLogvar);
    double[] xRecon = decode(z, c);

    // Optional: clamp xRecon to avoid instability
    for (int i = 0; i < xRecon.length; i++) {
        xRecon[i] = Math.max(Math.min(xRecon[i], 1e6), -1e6);
    }

    // ----- Loss -----
    double reconLoss = mseLoss(x, xRecon);

    double klLoss = 0.0;
    for (int i = 0; i < latentDim; i++) {
        double var = Math.exp(safeLogvar[i]);
        klLoss += -0.5 * (1 + safeLogvar[i] - mu[i] * mu[i] - var);
    }

    double loss = reconLoss + klLoss;

//    // --- Logging ---
//    System.out.printf("ReconLoss: %.6e, KL: %.6e, Total: %.6e\n", reconLoss, klLoss, loss);
//    System.out.println("mu:      " + Arrays.toString(mu));
//    System.out.println("logvar:  " + Arrays.toString(logvar));
//    System.out.println("z:       " + Arrays.toString(z));
//    System.out.println("xRecon:  " + Arrays.toString(xRecon));

    if (Double.isNaN(loss) || Double.isInfinite(loss)) {
        throw new RuntimeException("Training loss became NaN or Infinite — check input data or model stability.");
    }

    // ----- Backward Pass -----
    double[] dL_dxRecon = mseGradient(xRecon, x);
    double[] dL_dzc = dotT(dL_dxRecon, W_dec);
    double[] dz = slice(dL_dzc, 0, latentDim);

    double[] dL_dmu = new double[latentDim];
    double[] dL_dlogvar = new double[latentDim];
    for (int i = 0; i < latentDim; i++) {
        double sigma = Math.exp(0.5 * safeLogvar[i]);
        double eps = (z[i] - mu[i]) / sigma;
        dL_dmu[i] = dz[i] - mu[i];
        dL_dlogvar[i] = 0.5 * dz[i] * eps - 0.5 * (1 - Math.exp(safeLogvar[i]));
    }

    double[] dmu_dh = dotT(dL_dmu, W_mu);
    double[] dlogvar_dh = dotT(dL_dlogvar, W_logvar);
    double[] dL_dh = new double[hiddenDim];
    for (int i = 0; i < hiddenDim; i++) {
        double dtanh = 1 - h[i] * h[i];
        dL_dh[i] = (dmu_dh[i] + dlogvar_dh[i]) * dtanh;
    }

    // ----- Gradient Clipping -----
    clipGradient(dL_dh, 5.0);
    clipGradient(dL_dmu, 5.0);
    clipGradient(dL_dlogvar, 5.0);
    clipGradient(dL_dxRecon, 5.0);

    // ----- Parameter Update -----
    updateParameters(xc, h, dL_dh, mu, dL_dmu, dL_dlogvar, z, c, dL_dxRecon);

    return loss;
}

/**
 * Clip gradients elementwise to avoid exploding updates.
 */
private void clipGradient(double[] grad, double maxNorm) {
    for (int i = 0; i < grad.length; i++) {
        if (grad[i] > maxNorm) grad[i] = maxNorm;
        else if (grad[i] < -maxNorm) grad[i] = -maxNorm;
    }
}

/**
 * Utility function to detect NaNs or Infs in a vector.
 */
private static boolean hasNaNsOrInfs(double[] vec) {
    for (double v : vec) {
        if (Double.isNaN(v) || Double.isInfinite(v)) return true;
    }
    return false;
}

    /**
     * Apply gradient descent updates to all weights and biases.
     *
     * @param xc Concatenated input + condition vector (encoder input)
     * @param h Encoder hidden activations
     * @param dL_dh Gradient of loss w.r.t encoder hidden layer
     * @param mu Encoder mean output
     * @param dL_dmu Gradient of loss w.r.t mu output
     * @param dL_dlogvar Gradient of loss w.r.t log variance output
     * @param z Sampled latent vector
     * @param c Condition vector
     * @param dL_dxRecon Gradient of loss w.r.t decoder output (reconstruction)
     */
    private void updateParameters(double[] xc, double[] h, double[] dL_dh,
                                  double[] mu, double[] dL_dmu, double[] dL_dlogvar,
                                  double[] z, double[] c, double[] dL_dxRecon) {

        // Update encoder weights and biases
        updateWeights(W_enc, b_enc, xc, dL_dh);
        updateWeights(W_mu, b_mu, h, dL_dmu);
        updateWeights(W_logvar, b_logvar, h, dL_dlogvar);

        // Update decoder weights and biases
        double[] zc = concat(z, c);
        updateWeights(W_dec, b_dec, zc, dL_dxRecon);
    }

    /**
     * Reconstructs an input vector from a given condition vector (e.g., MDS embedding).
     * Uses a standard Gaussian latent vector (z ~ N(0, I)) as the stochastic source.
     *
     * @param condition The condition vector (e.g., a 2D or 3D embedding)
     * @return Reconstructed input vector from decoder
     */
    public double[] inverseTransform(double[] condition) {
        if (condition.length != conditionDim)
            throw new IllegalArgumentException("Condition vector must have dimension " + conditionDim);

        // Use a standard normal latent vector (z ~ N(0, I))
        double[] z = new double[latentDim];
        for (int i = 0; i < latentDim; i++) {
            z[i] = rand.nextGaussian(); // sample from N(0,1)
        }

        return decode(z, condition);
    }
    
    /**
     * Deterministic inverse transform using zero latent vector.
     * Useful for evaluating the mean reconstruction.
     */
    public double[] inverseTransformZeroLatent(double[] condition) {
        double[] z = new double[latentDim]; // all zeros
        return decode(z, condition);
    }    
    // --- Utility methods for matrix/vector operations and initialization ---

    /** Initialize vector with zeros */
    private static double[] initVector(int len) {
        return new double[len]; // Zero initialization (bias)
    }

    /** Initialize matrix with Xavier  initialization */
    private static double[][] initMatrix(int in, int out) {
        double[][] m = new double[in][out];
        Random rand = new Random();
        double scale = Math.sqrt(2.0 / (in + out)); // Xavier Glorot
        for (int i = 0; i < in; i++) {
            for (int j = 0; j < out; j++) {
                m[i][j] = rand.nextGaussian() * scale;
            }
        }
        return m;
    }    

    /** Apply element-wise tanh activation */
    private static double[] tanh(double[] x) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) y[i] = Math.tanh(x[i]);
        return y;
    }

    /** Element-wise vector addition */
    private static double[] add(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] + b[i];
        return r;
    }

    /**
     * Compute matrix product x * W
     * @param x Vector of length n
     * @param W Matrix n x m
     * @return Result vector length m
     */
    private static double[] dot(double[] x, double[][] W) {
        double[] result = new double[W[0].length];
        for (int j = 0; j < W[0].length; j++) {
            for (int i = 0; i < x.length; i++) {
                result[j] += x[i] * W[i][j];
            }
        }
        return result;
    }

    /**
     * Compute dot product of gradient vector dy with transpose of weight matrix W.
     * Used for backpropagation.
     *
     * @param dy Gradient vector of length m
     * @param W Weight matrix n x m
     * @return Gradient vector length n
     */
    private static double[] dotT(double[] dy, double[][] W) {
        double[] result = new double[W.length];
        for (int i = 0; i < W.length; i++) {
            for (int j = 0; j < W[0].length; j++) {
                result[i] += dy[j] * W[i][j];
            }
        }
        return result;
    }

    /**
     * Update weight matrix and bias vector using gradient descent.
     * W := W - learningRate * grad_outer_product
     *
     * @param W Weight matrix (inputSize x outputSize)
     * @param b Bias vector (outputSize)
     * @param input Input vector (inputSize)
     * @param gradOutput Gradient vector w.r.t layer output (outputSize)
     */
    private void updateWeights(double[][] W, double[] b, double[] input, double[] gradOutput) {
        for (int i = 0; i < W.length; i++) {
            for (int j = 0; j < W[0].length; j++) {
                W[i][j] -= learningRate * gradOutput[j] * input[i];
            }
        }
        for (int j = 0; j < b.length; j++) {
            b[j] -= learningRate * gradOutput[j];
        }
    }

    /** Concatenate two vectors */
    private static double[] concat(double[] a, double[] b) {
        double[] r = new double[a.length + b.length];
        System.arraycopy(a, 0, r, 0, a.length);
        System.arraycopy(b, 0, r, a.length, b.length);
        return r;
    }

    /** Slice subvector from array starting at 'start' of length 'len' */
    private static double[] slice(double[] arr, int start, int len) {
        double[] r = new double[len];
        System.arraycopy(arr, start, r, 0, len);
        return r;
    }

    /**
     * Compute mean squared error loss between vectors a and b
     */
    private static double mseLoss(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return sum / a.length;
    }

    /**
     * Compute gradient of MSE loss w.r.t predicted output
     */
    private static double[] mseGradient(double[] pred, double[] target) {
        double[] grad = new double[pred.length];
        for (int i = 0; i < pred.length; i++) {
            grad[i] = 2.0 * (pred[i] - target[i]) / pred.length;
        }
        return grad;
    }
   
}
