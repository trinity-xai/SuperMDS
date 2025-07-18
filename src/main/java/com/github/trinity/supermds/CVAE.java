package com.github.trinity.supermds;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static com.github.trinity.supermds.CVAEHelper.*;

/**
 * Conditional Variational Autoencoder (CVAE) implementation in pure Java.
 * Supports arbitrary input, condition, latent, and hidden dimensions. Training
 * uses gradient descent with backpropagation through encoder, latent sampling,
 * and decoder.
 * <p>
 * Activation functions: - Encoder/hidden layers: 2 ReLU - Output layer: linear
 * <p>
 * Loss: - Reconstruction loss: mean squared error (MSE) - KL divergence loss
 * for latent regularization
 * <p>
 * Usage: - Construct with input dimension, condition dimension, latent
 * dimension, and hidden dimension. - Call train() per sample to perform one
 * training step.
 *
 * @author Sean Phillips
 */
public class CVAE {

    private static final Logger LOG = LoggerFactory.getLogger(CVAE.class);
    private int inputDim;      // Dimensionality of input vector
    private int conditionDim;  // Dimensionality of condition vector
    private int latentDim;     // Dimensionality of latent space
    private int hiddenDim;     // Number of hidden units in each hidden layer
    // === Encoder weights ===
    private final double[][] W_enc1;
    private final double[][] W_enc2; // Weights for encoder layers
    private final double[] b_enc1;
    private final double[] b_enc2;   // Biases for encoder layers
    private final double[][] W_mu;       // Weights: hiddenDim x latentDim
    private final double[] b_mu;         // Biases: latentDim
    private final double[][] W_logvar;   // Weights: hiddenDim x latentDim
    private final double[] b_logvar;     // Biases: latentDim
    // Decoder weight matrices for 2 hidden layers + output layer
    private final double[][] W_dec1;
    private final double[][] W_dec2;
    private final double[][] W_decOut; // Weights for decoder layers
    private final double[] b_dec1;
    private final double[] b_dec2;
    private final double[] b_decOut; // Biases for decoder layers
    // === Annealing settings ===
    private final AtomicInteger currentEpoch = new AtomicInteger(0);
    private int klWarmupEpochs = 100;
    private double maxKLWeight = 0.5;
    private double klSharpness = 10.0;
    private double learningRate = 0.0001;
    private boolean useCyclicalAnneal = false;
    private int klAnnealCycleLength = 100;
    private double dropoutRate = 0.01; // 20% dropout is typical
    private boolean useDropout = true;
    private boolean isTraining = false;
    private final long seed = 42L;
    ThreadLocal<Random> threadLocalRandom;
    private final ThreadLocal<CVAEBufferSet> threadLocalBuffer
        = ThreadLocal.withInitial(() -> new CVAEBufferSet(inputDim, conditionDim, latentDim, hiddenDim));

    private int debugEpochCount = 10000;
    private boolean debug = false;

    /**
     * Initialize a new CVAE instance with 3 hidden layers in encoder and
     * decoder.
     *
     * @param inputDim     Dimensionality of input vector (e.g. 128)
     * @param conditionDim Dimensionality of conditioning vector (e.g. 3 for
     *                     MDS)
     * @param latentDim    Dimensionality of latent space (e.g. 8)
     * @param hiddenDim    Number of units in each hidden layer (e.g. 128)
     */
    public CVAE(int inputDim, int conditionDim, int latentDim, int hiddenDim) {
        this.inputDim = inputDim;
        this.conditionDim = conditionDim;
        this.latentDim = latentDim;
        this.hiddenDim = hiddenDim;
        threadLocalRandom = ThreadLocal.withInitial(()
            -> new Random(seed));
        int encInputDim = inputDim + conditionDim;    // Encoder input: [x | c]
        int decInputDim = latentDim + conditionDim;   // Decoder input: [z | c]
        Random rand = threadLocalRandom.get();

        // === Encoder ===
        W_enc1 = initMatrix(hiddenDim, encInputDim, true, rand);   // h1 = W_enc1 * [x | c]
        b_enc1 = initVector(hiddenDim);

        W_enc2 = initMatrix(hiddenDim, hiddenDim, true, rand);     // h2 = W_enc2 * h1
        b_enc2 = initVector(hiddenDim);

        W_mu = initMatrix(latentDim, hiddenDim, false, rand);    // mu = W_mu * h2
        b_mu = initVector(latentDim);

        W_logvar = initMatrix(latentDim, hiddenDim, false, rand);    // logvar = W_logvar * h2
        b_logvar = initVector(latentDim);

        // === Decoder ===
        W_dec1 = initMatrix(hiddenDim, decInputDim, true, rand);   // d1 = W_dec1 * [z | c]
        b_dec1 = initVector(hiddenDim);

        W_dec2 = initMatrix(hiddenDim, hiddenDim, true, rand);     // d2 = W_dec2 * d1
        b_dec2 = initVector(hiddenDim);

        W_decOut = initMatrix(inputDim, hiddenDim, false, rand);     // xRecon = W_decOut * d2
        b_decOut = initVector(inputDim);
    }

    public double trainBatch(double[][] xBatch, double[][] cBatch) {
        int batchSize = xBatch.length;
        double totalLoss = IntStream.range(0, batchSize).parallel()
            .mapToDouble(i -> train(xBatch[i], cBatch[i]))
            .sum();
        return totalLoss / batchSize;
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

        CVAEBufferSet buf = threadLocalBuffer.get();  // Reuse buffers
        Random rand = threadLocalRandom.get();

        // ===== Forward Pass =====
        // Encoder input = concat(x, c)
        System.arraycopy(x, 0, buf.xc, 0, inputDim);
        System.arraycopy(c, 0, buf.xc, inputDim, conditionDim);

        // Encoder layer 1: h1 = ReLU(W1 * xc + b1)
        dot(buf.xc, W_enc1, buf.h1);
        addInPlace(buf.h1, b_enc1);
        reluInPlace(buf.h1);

        // Encoder layer 2: h2 = ReLU(W2 * h1 + b2)
        dot(buf.h1, W_enc2, buf.h2);
        addInPlace(buf.h2, b_enc2);
        reluInPlace(buf.h2);

        // μ and log(σ²) outputs
        dot(buf.h2, W_mu, buf.mu);
        addInPlace(buf.mu, b_mu);

        dot(buf.h2, W_logvar, buf.logvar);
        addInPlace(buf.logvar, b_logvar);

        // Clip logvar for stability
        for (int i = 0; i < latentDim; i++) {
            buf.safeLogvar[i] = Math.max(Math.min(buf.logvar[i], 10.0), -10.0);
        }

        // Sample z ~ N(mu, sigma^2)
        sampleLatentInPlace(buf.mu, buf.safeLogvar, buf.z, rand);

        // Clamp extreme values
        for (int i = 0; i < latentDim; i++) {
            if (Math.abs(buf.z[i]) > 10.0) {
                buf.z[i] = Math.signum(buf.z[i]) * 10.0;
            }
        }

        // Decoder input = concat(z, c)
        System.arraycopy(buf.z, 0, buf.zc, 0, latentDim);
        System.arraycopy(c, 0, buf.zc, latentDim, conditionDim);

        // Decoder layer 1: d1 = ReLU(W1 * zc + b1)
        dot(buf.zc, W_dec1, buf.d1);
        addInPlace(buf.d1, b_dec1);
        reluInPlace(buf.d1);

        // Decoder layer 2: d2 = ReLU(W2 * d1 + b2)
        dot(buf.d1, W_dec2, buf.d2);
        addInPlace(buf.d2, b_dec2);
        reluInPlace(buf.d2);

        // Output layer: xRecon = W_out * d2 + b_out
        dot(buf.d2, W_decOut, buf.xRecon);
        addInPlace(buf.xRecon, b_decOut);

        for (int i = 0; i < inputDim; i++) {
            buf.xRecon[i] = Math.max(Math.min(buf.xRecon[i], 1e6), -1e6);
        }

        // ===== Loss =====
        double reconLoss = mseLoss(x, buf.xRecon);

        double klLoss = 0.0;
        for (int i = 0; i < latentDim; i++) {
            double var = Math.exp(buf.safeLogvar[i]);
            klLoss += -0.5 * (1 + buf.safeLogvar[i] - buf.mu[i] * buf.mu[i] - var);
        }

        double klWeight = isUseCyclicalAnneal()
            ? getCyclicalKLWeightSigmoid(currentEpoch.get(), getKlAnnealCycleLength(), maxKLWeight, getKlSharpness())
            : getKLWeight(currentEpoch.get(), klWarmupEpochs, maxKLWeight, getKlSharpness());

        double loss = reconLoss + klWeight * klLoss;

        if (Double.isNaN(loss) || Double.isInfinite(loss)) {
            if (debug) {
                LOG.info("x = {}", Arrays.toString(x));
                LOG.info("c = {}", Arrays.toString(c));
                LOG.info("xRecon = {}", Arrays.toString(buf.xRecon));
                LOG.info("mu = {}", Arrays.toString(buf.mu));
                LOG.info("logvar = {}", Arrays.toString(buf.logvar));
                LOG.info("safeLogvar = {}", Arrays.toString(buf.safeLogvar));
                LOG.info("z = {}", Arrays.toString(buf.z));
            }
            throw new RuntimeException("Training loss became NaN or Infinite — check input data or model stability.");
        }

        if (debug && currentEpoch.get() % getDebugEpochCount() == 0) {
            LOG.info("Epoch {} — Recon: {}, KL: {} (weight {}), Total: {}",
                currentEpoch.get(),
                String.format("%.6f", reconLoss),
                String.format("%.6f", klLoss),
                String.format("%.3f", klWeight),
                String.format("%.6f", loss));
        }

        // ===== Backward Pass =====
        // Grad from MSE loss
        mseGradient(buf.xRecon, x, buf.dL_dxRecon);

        // Decoder
        dotT(buf.dL_dxRecon, W_decOut, buf.dL_dDecOut);
        reluGradInPlace(buf.d2, buf.dL_dDecOut, buf.dL_dDec2);
        dotT(buf.dL_dDec2, W_dec2, buf.dL_dDec1);
        reluGradInPlace(buf.d1, buf.dL_dDec1, buf.dL_dDec1);
        dotT(buf.dL_dDec1, W_dec1, buf.dL_dZC);

        // Extract grad wrt z
        System.arraycopy(buf.dL_dZC, 0, buf.dz, 0, latentDim);

        // KL backprop
        for (int i = 0; i < latentDim; i++) {
            double sigma = Math.exp(0.5 * buf.safeLogvar[i]);
            double eps = (buf.z[i] - buf.mu[i]) / sigma;

            buf.dL_dmu[i] = buf.dz[i] + klWeight * buf.mu[i];
            buf.dL_dlogvar[i] = 0.5 * buf.dz[i] * eps + klWeight * 0.5 * (Math.exp(buf.safeLogvar[i]) - 1);
        }

        // Encoder backprop
        dotT(buf.dL_dmu, W_mu, buf.dmu_dh2);
        dotT(buf.dL_dlogvar, W_logvar, buf.dlogvar_dh2);
        for (int i = 0; i < hiddenDim; i++) {
            buf.dL_dh2[i] = buf.dmu_dh2[i] + buf.dlogvar_dh2[i];
        }
        reluGradInPlace(buf.h2, buf.dL_dh2, buf.dL_dh2);

        dotT(buf.dL_dh2, W_enc2, buf.dL_dh1);
        reluGradInPlace(buf.h1, buf.dL_dh1, buf.dL_dh1);

        // Clip gradients
        clipGradient(buf.dL_dxRecon, 5.0);
        clipGradient(buf.dL_dmu, 5.0);
        clipGradient(buf.dL_dlogvar, 5.0);
        clipGradient(buf.dL_dh2, 5.0);
        clipGradient(buf.dL_dh1, 5.0);

        // Update parameters
        updateParametersDeep(
            buf.xc, buf.h1, buf.h2,
            buf.dL_dh1, buf.dL_dh2,
            buf.dL_dmu, buf.dL_dlogvar,
            buf.z, c, buf.d1, buf.d2,
            buf.dL_dDec1, buf.dL_dDec2, buf.dL_dxRecon
        );

        currentEpoch.incrementAndGet();
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
        updateMatrix(W_enc1, xc, dh1, getLearningRate());
        updateVector(b_enc1, dh1, getLearningRate());

        updateMatrix(W_enc2, h1, dh2, getLearningRate());
        updateVector(b_enc2, dh2, getLearningRate());

        updateMatrix(W_mu, h2, dmu, getLearningRate());
        updateVector(b_mu, dmu, getLearningRate());

        updateMatrix(W_logvar, h2, dlogvar, getLearningRate());
        updateVector(b_logvar, dlogvar, getLearningRate());

        // ----- Decoder -----
        double[] zc = concat(z, c);

        // Decoder Layer 1
        updateMatrix(W_dec1, zc, dL_dDec1, getLearningRate());
        updateVector(b_dec1, dL_dDec1, getLearningRate());

        // Decoder Layer 2
        updateMatrix(W_dec2, d1, dL_dDec2, getLearningRate());
        updateVector(b_dec2, dL_dDec2, getLearningRate());

        // Output Layer
        updateMatrix(W_decOut, d2, dL_dxRecon, getLearningRate());
        updateVector(b_decOut, dL_dxRecon, getLearningRate());
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
            z[i] = threadLocalRandom.get().nextGaussian(); // sample from N(0,1)
        }

        return decode(z, condition);
    }

    /**
     * Estimates reconstruction confidence by sampling multiple latent vectors
     * and measuring variance in the decoder's outputs.
     *
     * @param condition  Condition vector used during decoding (e.g. MDS
     *                   embedding)
     * @param numSamples Number of latent samples to draw (e.g. 10–100)
     * @return Variance vector (length inputDim) where lower values indicate
     * higher confidence
     */
    public double[] confidenceEstimate(double[] condition, int numSamples) {
        if (condition.length != conditionDim) {
            throw new IllegalArgumentException("Condition vector must have dimension " + conditionDim);
        }
        if (numSamples <= 1) {
            throw new IllegalArgumentException("Need at least 2 samples for confidence estimation.");
        }

        double[][] outputs = new double[numSamples][inputDim];
        Random rng = threadLocalRandom.get();

        // Sample reconstructions
        for (int i = 0; i < numSamples; i++) {
            double[] z = new double[latentDim];
            for (int j = 0; j < latentDim; j++) {
                z[j] = rng.nextGaussian();
            }
            outputs[i] = decode(z, condition);
        }

        // Compute mean
        double[] mean = new double[inputDim];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < inputDim; j++) {
                mean[j] += outputs[i][j];
            }
        }
        for (int j = 0; j < inputDim; j++) {
            mean[j] /= numSamples;
        }

        // Compute variance
        double[] variance = new double[inputDim];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < inputDim; j++) {
                double diff = outputs[i][j] - mean[j];
                variance[j] += diff * diff;
            }
        }
        for (int j = 0; j < inputDim; j++) {
            variance[j] /= (numSamples - 1); // unbiased estimate
        }

        return variance; // Lower variance = higher confidence
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
        Random rand = threadLocalRandom.get();
        // Step 2: Hidden layer 1
        double[] h1 = relu(add(dot(zc, W_dec1), b_dec1)); // [hiddenDim]
        if (useDropout && isIsTraining()) {
            h1 = applyDropout(h1, dropoutRate, rand);
        }

        // Step 3: Hidden layer 2
        double[] h2 = relu(add(dot(h1, W_dec2), b_dec2)); // [hiddenDim]
        if (useDropout && isIsTraining()) {
            h2 = applyDropout(h2, dropoutRate, rand);
        }

        // Step 4: Final linear output (no activation)
        double[] out = add(dot(h2, W_decOut), b_decOut); // [inputDim]

        return out;
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
        Random rand = threadLocalRandom.get();
        // Step 2: Hidden layer 1
        double[] h1 = relu(add(dot(xc, W_enc1), b_enc1)); // [hiddenDim]
        if (useDropout) {
            h1 = applyDropout(h1, dropoutRate, rand);
        }

        // Step 3: Hidden layer 2
        double[] h2 = relu(add(dot(h1, W_enc2), b_enc2)); // [hiddenDim]
        if (useDropout) {
            h2 = applyDropout(h2, dropoutRate, rand);
        }

        return h2;
    }

    /**
     * Deterministic inverse transform using zero latent vector. Useful for
     * evaluating the mean reconstruction.
     */
    public double[] inverseTransformZeroLatent(double[] condition) {
        double[] z = new double[latentDim]; // all zeros
        return decode(z, condition);
    }

    //<editor-fold defaultstate="collapsed" desc="Properties">

    /**
     * Set the current training epoch, used for KL annealing. Should be called
     * once per epoch from the training loop.
     *
     * @param epoch Current epoch number (0-based)
     */
    public void setCurrentEpoch(int epoch) {
        this.currentEpoch.set(epoch);
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
        this.currentEpoch.set(0);
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

    /**
     * @return the klSharpness
     */
    public double getKlSharpness() {
        return klSharpness;
    }

    /**
     * @param klSharpness the klSharpness to set
     */
    public void setKlSharpness(double klSharpness) {
        this.klSharpness = klSharpness;
    }

    /**
     * @return the learningRate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * @param learningRate the learningRate to set
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * @return the useCyclicalAnneal
     */
    public boolean isUseCyclicalAnneal() {
        return useCyclicalAnneal;
    }

    /**
     * @param useCyclicalAnneal the useCyclicalAnneal to set
     */
    public void setUseCyclicalAnneal(boolean useCyclicalAnneal) {
        this.useCyclicalAnneal = useCyclicalAnneal;
    }

    /**
     * @return the klAnnealCycleLength
     */
    public int getKlAnnealCycleLength() {
        return klAnnealCycleLength;
    }

    /**
     * @param klAnnealCycleLength the klAnnealCycleLength to set
     */
    public void setKlAnnealCycleLength(int klAnnealCycleLength) {
        this.klAnnealCycleLength = klAnnealCycleLength;
    }

    /**
     * @return the debugEpochCount
     */
    public int getDebugEpochCount() {
        return debugEpochCount;
    }

    /**
     * @param debugEpochCount the debugEpochCount to set
     */
    public void setDebugEpochCount(int debugEpochCount) {
        this.debugEpochCount = debugEpochCount;
    }

    /**
     * @return the debug
     */
    public boolean isDebug() {
        return debug;
    }

    /**
     * @param debug the debug to set
     */
    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    /**
     * @return the dropoutRate
     */
    public double getDropoutRate() {
        return dropoutRate;
    }

    /**
     * @param dropoutRate the dropoutRate to set
     */
    public void setDropoutRate(double dropoutRate) {
        this.dropoutRate = dropoutRate;
    }

    /**
     * @return the useDropout
     */
    public boolean isUseDropout() {
        return useDropout;
    }

    /**
     * @param useDropout the useDropout to set
     */
    public void setUseDropout(boolean useDropout) {
        this.useDropout = useDropout;
    }

    /**
     * @return the isTraining
     */
    public boolean isIsTraining() {
        return isTraining;
    }

    /**
     * @param isTraining the isTraining to set
     */
    public void setIsTraining(boolean isTraining) {
        this.isTraining = isTraining;
    }
    //</editor-fold>
}
