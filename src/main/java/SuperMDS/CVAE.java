package SuperMDS;

import static SuperMDS.CVAEHelper.add;
import static SuperMDS.CVAEHelper.clipGradient;
import static SuperMDS.CVAEHelper.concat;
import static SuperMDS.CVAEHelper.dot;
import static SuperMDS.CVAEHelper.dotT;
import static SuperMDS.CVAEHelper.getKLWeight;
import static SuperMDS.CVAEHelper.hasNaNsOrInfs;
import static SuperMDS.CVAEHelper.initMatrix;
import static SuperMDS.CVAEHelper.initVector;
import static SuperMDS.CVAEHelper.mseGradient;
import static SuperMDS.CVAEHelper.mseLoss;
import static SuperMDS.CVAEHelper.relu;
import static SuperMDS.CVAEHelper.slice;
import static SuperMDS.CVAEHelper.updateMatrix;
import static SuperMDS.CVAEHelper.updateVector;
import java.util.Arrays;
import java.util.Random;

/**
 * Conditional Variational Autoencoder (CVAE) implementation in pure Java.
 * Supports arbitrary input, condition, latent, and hidden dimensions. Training
 * uses gradient descent with backpropagation through encoder, latent sampling,
 * and decoder.
 *
 * Activation functions: - Encoder/hidden layers: 2 ReLU - Output layer: linear
 *
 * Loss: - Reconstruction loss: mean squared error (MSE) - KL divergence loss
 * for latent regularization
 *
 * Usage: - Construct with input dimension, condition dimension, latent
 * dimension, and hidden dimension. - Call train() per sample to perform one
 * training step.
 *
 * @author Sean Phillips
 */
public class CVAE {

    private int inputDim;      // Dimensionality of input vector
    private int conditionDim;  // Dimensionality of condition vector
    private int latentDim;     // Dimensionality of latent space
    private int hiddenDim;     // Number of hidden units in each hidden layer

    private Random rand;

    // === Encoder weights ===
    private double[][] W_enc1, W_enc2; // Weights for encoder layers
    private double[] b_enc1, b_enc2;   // Biases for encoder layers

    private double[][] W_mu;       // Weights: hiddenDim x latentDim
    private double[] b_mu;         // Biases: latentDim
    private double[][] W_logvar;   // Weights: hiddenDim x latentDim
    private double[] b_logvar;     // Biases: latentDim

    // === Decoder weights ===
    // Decoder weight matrices for 2 hidden layers + output layer
    private double[][] W_dec1, W_dec2, W_decOut; // Weights for decoder layers
    private double[] b_dec1, b_dec2, b_decOut; // Biases for decoder layers
    // === Annealing settings ===
    private int currentEpoch = 0;
    private int klWarmupEpochs = 1000;
    private double maxKLWeight = 0.75;
    private double klSharpness = 8.0;
    private double learningRate = 0.0001;
    

    /**
     * Initialize a new CVAE instance with 3 hidden layers in encoder and
     * decoder.
     *
     * @param inputDim Dimensionality of input vector (e.g. 128)
     * @param conditionDim Dimensionality of conditioning vector (e.g. 3 for
     * MDS)
     * @param latentDim Dimensionality of latent space (e.g. 8)
     * @param hiddenDim Number of units in each hidden layer (e.g. 128)
     */
    public CVAE(int inputDim, int conditionDim, int latentDim, int hiddenDim) {
        this.inputDim = inputDim;
        this.conditionDim = conditionDim;
        this.latentDim = latentDim;
        this.hiddenDim = hiddenDim;
        rand = new Random();
        int encInputDim = inputDim + conditionDim;    // Encoder input: [x | c]
        int decInputDim = latentDim + conditionDim;   // Decoder input: [z | c]

        // === Encoder ===
        // Encoder weights (ReLU activations → He initialization)
        W_enc1 = initMatrix(encInputDim, hiddenDim, true); //Relu us HE init
        b_enc1 = initVector(hiddenDim);

        W_enc2 = initMatrix(hiddenDim, hiddenDim, true); //Relu us HE init
        b_enc2 = initVector(hiddenDim);

//        W_enc3 = initMatrix(hiddenDim, hiddenDim, true); //Relu us HE init
//        b_enc3 = initVector(hiddenDim);
        // Output layers of encoder: linear → Xavier
        W_mu = initMatrix(hiddenDim, latentDim, false);
        b_mu = initVector(latentDim);

        W_logvar = initMatrix(hiddenDim, latentDim, false);
        b_logvar = initVector(latentDim);

        // === Decoder === (2 hidden layers + output)
        // Decoder weights (ReLU activations → He initialization)
        W_dec1 = initMatrix(decInputDim, hiddenDim, true); //Relu us HE init
        b_dec1 = initVector(hiddenDim);

        W_dec2 = initMatrix(hiddenDim, hiddenDim, true); //Relu us HE init
        b_dec2 = initVector(hiddenDim);
        // Final layer of decoder: assume linear output → Xavier
        W_decOut = initMatrix(hiddenDim, inputDim, false);   // Output layer, Xavier initialization
        b_decOut = initVector(inputDim);
    }

    /**
     * Encoder forward pass: input + condition → deep hidden representation
     * using 3 hidden layers with ReLU activation.
     *
     * @param x Input vector (e.g. original high-dimensional point)
     * @param c Condition vector (e.g. 2D or 3D embedding from MDS)
     * @return Final hidden layer activation vector (depth = hiddenDim)
     */
    public double[] encode(double[] x, double[] c) {
        // Step 1: Concatenate input and condition vectors
        double[] xc = concat(x, c); // [inputDim + conditionDim]

        // Step 2: Hidden layer 1
        double[] h1 = relu(add(dot(xc, W_enc1), b_enc1)); // [hiddenDim]

        // Step 3: Hidden layer 2
        double[] h2 = relu(add(dot(h1, W_enc2), b_enc2)); // [hiddenDim]

        return h2;
        
//        // Step 4: Hidden layer 3
//        double[] h3 = relu(add(dot(h2, W_enc3), b_enc3)); // [hiddenDim]
//
//        return h3;
    }

    /**
     * Decoder forward pass: latent vector + condition → reconstruction. Uses 3
     * hidden layers with ReLU activations followed by a linear output layer.
     *
     * @param z Latent vector (sampled or reparameterized vector from encoder)
     * @param c Condition vector (e.g., embedding from SMACOF MDS)
     * @return Reconstructed input vector (linear output, same shape as original
     * input)
     */
    public double[] decode(double[] z, double[] c) {
        // Step 1: Concatenate latent vector and condition vector
        double[] zc = concat(z, c); // [latentDim + conditionDim]

        // Step 2: Hidden layer 1
        double[] h1 = relu(add(dot(zc, W_dec1), b_dec1)); // [hiddenDim]

        // Step 3: Hidden layer 2
        double[] h2 = relu(add(dot(h1, W_dec2), b_dec2)); // [hiddenDim]

        // Step 4: Final linear output (no activation)
        double[] out = add(dot(h2, W_decOut), b_decOut); // [inputDim]

        return out;
    }

    /**
     * Sample latent vector z from mu and log variance using reparameterization
     * trick.
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
     * Compute the gradient of the ReLU activation function using the output as
     * a mask. This is the elementwise product of the upstream gradient and the
     * ReLU derivative.
     *
     * @param output The output of the ReLU function (i.e.,
     * ReLU(pre-activation))
     * @param upstream The gradient flowing from the next layer
     * @return Gradient to backpropagate to the previous layer
     */
    public double[] reluGrad(double[] output, double[] upstream) {
        double[] grad = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            grad[i] = output[i] > 0.0 ? upstream[i] : 0.0;
        }
        return grad;
    }

    /**
     * Perform one training step on a single (input, condition) pair. Includes
     * full forward + backward pass, with gradient clipping and numerical
     * stability controls.
     *
     * @param x Input vector
     * @param c Condition vector
     * @return Total loss (reconstruction + weighted KL divergence)
     */
    public double train(double[] x, double[] c) {
        if (hasNaNsOrInfs(x) || hasNaNsOrInfs(c)) {
            throw new IllegalArgumentException("Input or condition vector contains NaNs or Infs.");
        }

        double[] xc = concat(x, c);

        // ===== Forward Pass =====
        // Encoder
        double[] h1 = relu(add(dot(xc, W_enc1), b_enc1));
        double[] h2 = relu(add(dot(h1, W_enc2), b_enc2));
//        double[] h3 = relu(add(dot(h2, W_enc3), b_enc3));
//
//        double[] mu = add(dot(h3, W_mu), b_mu);
//        double[] logvar = add(dot(h3, W_logvar), b_logvar);

        double[] mu = add(dot(h2, W_mu), b_mu);
        double[] logvar = add(dot(h2, W_logvar), b_logvar);
        
        double[] safeLogvar = new double[latentDim];
        for (int i = 0; i < latentDim; i++) {
            safeLogvar[i] = Math.max(Math.min(logvar[i], 10.0), -10.0);
        }

        double[] z = sampleLatent(mu, safeLogvar);
        double[] zc = concat(z, c);

        // Decoder
        double[] d1 = relu(add(dot(zc, W_dec1), b_dec1));
        double[] d2 = relu(add(dot(d1, W_dec2), b_dec2));
        double[] xRecon = add(dot(d2, W_decOut), b_decOut);  // Linear output layer

        for (int i = 0; i < xRecon.length; i++) {
            xRecon[i] = Math.max(Math.min(xRecon[i], 1e6), -1e6);  // Clamp output
        }

        // ===== Loss =====
        double reconLoss = mseLoss(x, xRecon);
        double klLoss = 0.0;
        for (int i = 0; i < latentDim; i++) {
            double var = Math.exp(safeLogvar[i]);
            klLoss += -0.5 * (1 + safeLogvar[i] - mu[i] * mu[i] - var);
        }

        double klWeight = getKLWeight(currentEpoch, klWarmupEpochs, maxKLWeight, klSharpness);
        double loss = reconLoss + klWeight * klLoss;

        if (Double.isNaN(loss) || Double.isInfinite(loss)) {
            System.out.println("x = " + Arrays.toString(x));
            System.out.println("c = " + Arrays.toString(c));
            System.out.println("xRecon = " + Arrays.toString(xRecon));
            System.out.println("mu = " + Arrays.toString(mu));
            System.out.println("logvar = " + Arrays.toString(logvar));
            System.out.println("safeLogvar = " + Arrays.toString(safeLogvar));
            System.out.println("z = " + Arrays.toString(z));
            throw new RuntimeException("Training loss became NaN or Infinite — check input data or model stability.");
        }

        // ===== Backward Pass =====
        double[] dL_dxRecon = mseGradient(xRecon, x);

        // Decoder
        double[] dL_dDecOut = dotT(dL_dxRecon, W_decOut);
        double[] dL_dDec2 = reluGrad(d2, dL_dDecOut);
        double[] dL_dDec1 = reluGrad(d1, dotT(dL_dDec2, W_dec2));
        double[] dL_dZC = dotT(dL_dDec1, W_dec1);

        double[] dz = slice(dL_dZC, 0, latentDim);

        // VAE reparam gradients
        double[] dL_dmu = new double[latentDim];
        double[] dL_dlogvar = new double[latentDim];
        for (int i = 0; i < latentDim; i++) {
            double sigma = Math.exp(0.5 * safeLogvar[i]);
            double eps = (z[i] - mu[i]) / sigma;
            dL_dmu[i] = dz[i] - klWeight * mu[i];
            dL_dlogvar[i] = klWeight * (0.5 * dz[i] * eps - 0.5 * (1 - Math.exp(safeLogvar[i])));
        }

        // Encoder gradients
        double[] dmu_dh2 = dotT(dL_dmu, W_mu);
        double[] dlogvar_dh2 = dotT(dL_dlogvar, W_logvar);
//        double[] dL_dh3 = reluGrad(h3, add(dmu_dh3, dlogvar_dh3));
//        double[] dL_dh2 = reluGrad(h2, dotT(dL_dh3, W_enc3));

        double[] dL_dh2 = reluGrad(h2, add(dmu_dh2, dlogvar_dh2));

        double[] dL_dh1 = reluGrad(h1, dotT(dL_dh2, W_enc2));

        // ===== Gradient Clipping =====
        clipGradient(dL_dxRecon, 5.0);
        clipGradient(dL_dmu, 5.0);
        clipGradient(dL_dlogvar, 5.0);
//        clipGradient(dL_dh3, 5.0);
        clipGradient(dL_dh2, 5.0);
        clipGradient(dL_dh1, 5.0);

        // ===== Parameter Updates =====
        updateParametersDeep(
                xc, h1, h2, 
                dL_dh1, dL_dh2, 
                dL_dmu, dL_dlogvar,
                z, c, d1, d2, 
                dL_dDec1, dL_dDec2, dL_dxRecon 
        );

        currentEpoch++;
        return loss;
    }

    /**
     * Perform gradient descent updates for all weights and biases in the
     * 3-layer encoder and 2-layer decoder CVAE.
     */
    private void updateParametersDeep(
            double[] xc, double[] h1, double[] h2, 
            double[] dh1, double[] dh2, 
            double[] dmu, double[] dlogvar,
            double[] z, double[] c,
            double[] d1, double[] d2, 
            double[] dL_dDec1, double[] dL_dDec2, double[] dL_dxRecon
    ) {
        // ----- Encoder -----
        updateMatrix(W_enc1, xc, dh1, learningRate);
        updateVector(b_enc1, dh1, learningRate);

        updateMatrix(W_enc2, h1, dh2, learningRate);
        updateVector(b_enc2, dh2, learningRate);

//        updateMatrix(W_enc3, h2, dh3, learningRate);
//        updateVector(b_enc3, dh3, learningRate);

//        updateMatrix(W_mu, h3, dmu, learningRate);
        updateMatrix(W_mu, h2, dmu, learningRate);
        updateVector(b_mu, dmu, learningRate);

//        updateMatrix(W_logvar, h3, dlogvar, learningRate);
        updateMatrix(W_logvar, h2, dlogvar, learningRate);
        updateVector(b_logvar, dlogvar, learningRate);

        // ----- Decoder -----
        double[] zc = concat(z, c);

        // Decoder Layer 1
        updateMatrix(W_dec1, zc, dL_dDec1, learningRate);
        updateVector(b_dec1, dL_dDec1, learningRate);

        // Decoder Layer 2
        updateMatrix(W_dec2, d1, dL_dDec2, learningRate);
        updateVector(b_dec2, dL_dDec2, learningRate);

        // Output Layer
        updateMatrix(W_decOut, d2, dL_dxRecon, learningRate);
        updateVector(b_decOut, dL_dxRecon, learningRate);
    }

    /**
     * Reconstructs an input vector from a given condition vector (e.g., MDS
     * embedding). Uses a standard Gaussian latent vector (z ~ N(0, I)) as the
     * stochastic source.
     *
     * @param condition The condition vector (e.g., a 2D or 3D embedding)
     * @return Reconstructed input vector from decoder
     */
    public double[] inverseTransform(double[] condition) {
        if (condition.length != conditionDim) {
            throw new IllegalArgumentException("Condition vector must have dimension " + conditionDim);
        }

        // Use a standard normal latent vector (z ~ N(0, I))
        double[] z = new double[latentDim];
        for (int i = 0; i < latentDim; i++) {
            z[i] = rand.nextGaussian(); // sample from N(0,1)
        }

        return decode(z, condition);
    }

    /**
     * Deterministic inverse transform using zero latent vector. Useful for
     * evaluating the mean reconstruction.
     */
    public double[] inverseTransformZeroLatent(double[] condition) {
        double[] z = new double[latentDim]; // all zeros
        return decode(z, condition);
    }

    /**
     * Set the current training epoch, used for KL annealing. Should be called
     * once per epoch from the training loop.
     *
     * @param epoch Current epoch number (0-based)
     */
    public void setCurrentEpoch(int epoch) {
        this.currentEpoch = epoch;
    }

    /**
     * Sets the number of epochs used to warm up the KL divergence term.
     *
     * @param epochs Number of warm-up epochs
     */
    public void setKlWarmupEpochs(int epochs) {
        this.klWarmupEpochs = epochs;
    }

    /**
     * Resets the internal training epoch counter (optional if needed).
     */
    public void resetEpochCounter() {
        this.currentEpoch = 0;
    }
    /**
     * @return the maxKLWeight
     */
    public double getMaxKLWeight() {
        return maxKLWeight;
    }

    /**
     * @param maxKLWeight the maxKLWeight to set
     */
    public void setMaxKLWeight(double maxKLWeight) {
        this.maxKLWeight = maxKLWeight;
    }
}
