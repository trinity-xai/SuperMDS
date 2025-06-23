package SuperMDS;

import static SuperMDS.CVAEHelper.accumulateInto;
import static SuperMDS.CVAEHelper.accumulateOuterProduct;
import static SuperMDS.CVAEHelper.accumulateVector;
import static SuperMDS.CVAEHelper.add;
import static SuperMDS.CVAEHelper.addInPlace;
import static SuperMDS.CVAEHelper.applyDropout;
import static SuperMDS.CVAEHelper.applyDropoutInPlace;
import static SuperMDS.CVAEHelper.applyUpdate;
import static SuperMDS.CVAEHelper.clipGradient;
import static SuperMDS.CVAEHelper.concat;
import static SuperMDS.CVAEHelper.concatInPlace;
import static SuperMDS.CVAEHelper.dot;
import static SuperMDS.CVAEHelper.dotInPlace;
import static SuperMDS.CVAEHelper.dotTInPlace;
import static SuperMDS.CVAEHelper.getCyclicalKLWeightSigmoid;
import static SuperMDS.CVAEHelper.getKLWeight;
import static SuperMDS.CVAEHelper.initMatrix;
import static SuperMDS.CVAEHelper.initVector;
import static SuperMDS.CVAEHelper.mseGradientInPlace;
import static SuperMDS.CVAEHelper.mseLoss;
import static SuperMDS.CVAEHelper.relu;
import static SuperMDS.CVAEHelper.reluGradInPlace;
import static SuperMDS.CVAEHelper.reluInPlace;
import static SuperMDS.CVAEHelper.sampleLatentInPlace;
import static SuperMDS.CVAEHelper.scaleGradientsInPlace;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
public class CVAEInPlace {

    private int inputDim;      // Dimensionality of input vector
    private int conditionDim;  // Dimensionality of condition vector
    private int latentDim;     // Dimensionality of latent space
    private int hiddenDim;     // Number of hidden units in each hidden layer

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
    private AtomicInteger currentEpoch = new AtomicInteger(0);
    private int klWarmupEpochs = 1000;
    private double maxKLWeight = 0.5;
    private double klSharpness = 10.0;
    private double learningRate = 0.001;
    private boolean useCyclicalAnneal = false;
    private int klAnnealCycleLength = 100;

    private double dropoutRate = 0.01; // 20% dropout is typical
    private boolean useDropout = false;
    private boolean isTraining = false;
    private boolean useSIMD = false;

    ThreadLocal<Random> threadLocalRandom;

    private int debugEpochCount = 10000;
    private boolean debug = false;

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
    public CVAEInPlace(int inputDim, int conditionDim, int latentDim, int hiddenDim) {
        this.inputDim = inputDim;
        this.conditionDim = conditionDim;
        this.latentDim = latentDim;
        this.hiddenDim = hiddenDim;

        threadLocalRandom = ThreadLocal.withInitial(() 
            -> new Random(System.nanoTime()));
    
        int encInputDim = inputDim + conditionDim;    // Encoder input: [x | c]
        int decInputDim = latentDim + conditionDim;   // Decoder input: [z | c]

        // === Encoder ===
        W_enc1 = initMatrix(encInputDim, hiddenDim, true);  // He init for ReLU
        b_enc1 = initVector(hiddenDim);

        W_enc2 = initMatrix(hiddenDim, hiddenDim, true);    // He init
        b_enc2 = initVector(hiddenDim);

        W_mu = initMatrix(hiddenDim, latentDim, false);     // Xavier init
        b_mu = initVector(latentDim);

        W_logvar = initMatrix(hiddenDim, latentDim, false); // Xavier init
        b_logvar = initVector(latentDim);

        // === Decoder ===
        W_dec1 = initMatrix(decInputDim, hiddenDim, true);  // He init
        b_dec1 = initVector(hiddenDim);

        W_dec2 = initMatrix(hiddenDim, hiddenDim, true);    // He init
        b_dec2 = initVector(hiddenDim);

        //Row Major way
        W_decOut = initMatrix(hiddenDim, inputDim, false);  // Xavier init
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

    public double trainBatchInPlaceParallel(double[][] xBatch, double[][] cBatch) {
       int batchSize = xBatch.length;
       // Step 0: Allocate per-sample buffers once (outside parallel loop)
       List<BufferSet> bufferCopies = IntStream.range(0, batchSize)
           .mapToObj(i -> new BufferSet(inputDim, conditionDim, latentDim, hiddenDim))
           .collect(Collectors.toList());
       // Step 1: Master gradient accumulator (cleared before use)
       BufferSet agg = new BufferSet(inputDim, conditionDim, latentDim, hiddenDim);
       agg.resetGradients(); // clear aggregated gradients
       // Step 2: Parallel per-sample gradient accumulation
       DoubleAdder totalLossAdder = new DoubleAdder();
       IntStream.range(0, batchSize).parallel().forEach(i -> {
           double[] x = xBatch[i];
           double[] c = cBatch[i];
           BufferSet buf = bufferCopies.get(i);
           // Thread-local forward resets
           buf.resetForwardBuffers();
           Random rng = threadLocalRandom.get();
           double loss = trainSampleAccumulateGradients(x, c, buf, rng);
           totalLossAdder.add(loss);
       });
       // Step 3: Aggregate gradients from all per-sample buffers into master
       for (BufferSet buf : bufferCopies) {
           accumulateInto(agg, buf);
       }

       // Step 3: Scale gradients
       double invBatchSize = 1.0 / batchSize;
       scaleGradientsInPlace(agg, invBatchSize);

        // Step 5: Apply updates to model weights
       applyUpdate(W_enc1, agg.grad_W_enc1, learningRate);
       applyUpdate(b_enc1, agg.grad_b_enc1, learningRate);

       applyUpdate(W_enc2, agg.grad_W_enc2, learningRate);
       applyUpdate(b_enc2, agg.grad_b_enc2, learningRate);

       applyUpdate(W_mu, agg.grad_W_mu, learningRate);
       applyUpdate(b_mu, agg.grad_b_mu, learningRate);

       applyUpdate(W_logvar, agg.grad_W_logvar, learningRate);
       applyUpdate(b_logvar, agg.grad_b_logvar, learningRate);

       applyUpdate(W_dec1, agg.grad_W_dec1, learningRate);
       applyUpdate(b_dec1, agg.grad_b_dec1, learningRate);

       applyUpdate(W_dec2, agg.grad_W_dec2, learningRate);
       applyUpdate(b_dec2, agg.grad_b_dec2, learningRate);

       applyUpdate(W_decOut, agg.grad_W_decOut, learningRate);
       applyUpdate(b_decOut, agg.grad_b_decOut, learningRate);

       // Step 6: Advance epoch
       currentEpoch.incrementAndGet();
       return totalLossAdder.sum() / batchSize;
    }

    public double trainSampleAccumulateGradients(double[] x, double[] c, BufferSet buf, Random rand) {
        // ===== Forward Pass =====
        concatInPlace(x, c, buf.xc);
        dotInPlace(buf.xc, W_enc1, buf.h1);
        addInPlace(buf.h1, b_enc1);
        reluInPlace(buf.h1);
        if (useDropout && isIsTraining()) {
            applyDropoutInPlace(buf.h1, dropoutRate, rand);
        }

        dotInPlace(buf.h1, W_enc2, buf.h2);
        addInPlace(buf.h2, b_enc2);
        reluInPlace(buf.h2);
        if (useDropout && isIsTraining()) {
            applyDropoutInPlace(buf.h2, dropoutRate, rand);
        }

        dotInPlace(buf.h2, W_mu, buf.mu);
        addInPlace(buf.mu, b_mu);
        dotInPlace(buf.h2, W_logvar, buf.logvar);
        addInPlace(buf.logvar, b_logvar);

        for (int i = 0; i < latentDim; i++) {
            buf.safeLogvar[i] = Math.max(Math.min(buf.logvar[i], 10.0), -10.0);
        }
        sampleLatentInPlace(buf.mu, buf.safeLogvar, buf.z, rand);
        for (int i = 0; i < buf.z.length; i++) {
            buf.z[i] = Math.max(Math.min(buf.z[i], 10.0), -10.0);
        }

        concatInPlace(buf.z, c, buf.zc);
        dotInPlace(buf.zc, W_dec1, buf.d1);
        addInPlace(buf.d1, b_dec1);
        reluInPlace(buf.d1);
        if (useDropout && isIsTraining()) {
            applyDropoutInPlace(buf.d1, dropoutRate, rand);
        }

        dotInPlace(buf.d1, W_dec2, buf.d2);
        addInPlace(buf.d2, b_dec2);
        reluInPlace(buf.d2);
        if (useDropout && isIsTraining()) {
            applyDropoutInPlace(buf.d2, dropoutRate, rand);
        }

        dotInPlace(buf.d2, W_decOut, buf.xRecon);
        addInPlace(buf.xRecon, b_decOut);

        // Clamp outputs
        for (int i = 0; i < buf.xRecon.length; i++) {
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
            throw new RuntimeException("NaN or Inf loss in CVAE training");
        }

        // ===== Backward Pass =====
        mseGradientInPlace(buf.xRecon, x, buf.grad_xRecon);
        dotTInPlace(buf.grad_xRecon, W_decOut, buf.dL_dDec2);
        reluGradInPlace(buf.d2, buf.dL_dDec2, buf.dL_dDec2);
        dotTInPlace(buf.dL_dDec2, W_dec2, buf.dL_dDec1);
        reluGradInPlace(buf.d1, buf.dL_dDec1, buf.dL_dDec1);
        dotTInPlace(buf.dL_dDec1, W_dec1, buf.dL_dZC);

        System.arraycopy(buf.dL_dZC, 0, buf.dz, 0, latentDim);

        for (int i = 0; i < latentDim; i++) {
            double sigma = Math.exp(0.5 * buf.safeLogvar[i]);
            double eps = (buf.z[i] - buf.mu[i]) / sigma;

            buf.grad_mu[i] = buf.dz[i] + klWeight * buf.mu[i];
            buf.grad_logvar[i] = 0.5 * buf.dz[i] * eps + klWeight * 0.5 * (Math.exp(buf.safeLogvar[i]) - 1);
        }

        dotTInPlace(buf.grad_mu, W_mu, buf.dmu_dh2);
        dotTInPlace(buf.grad_logvar, W_logvar, buf.dlogvar_dh2);
        addInPlace(buf.dmu_dh2, buf.dlogvar_dh2, buf.dL_dh2);
        reluGradInPlace(buf.h2, buf.dL_dh2, buf.dL_dh2);
        dotTInPlace(buf.dL_dh2, W_enc2, buf.dL_dh1);
        reluGradInPlace(buf.h1, buf.dL_dh1, buf.dL_dh1);

        // Gradient Clipping
        clipGradient(buf.grad_xRecon, 5.0);
        clipGradient(buf.grad_mu, 5.0);
        clipGradient(buf.grad_logvar, 5.0);
        clipGradient(buf.dL_dh2, 5.0);
        clipGradient(buf.dL_dh1, 5.0);

        // ===== Accumulate Gradients =====
        // --- Encoder ---
        accumulateOuterProduct(buf.grad_W_enc1, buf.xc, buf.dL_dh1);
        accumulateVector(buf.grad_b_enc1, buf.dL_dh1);

        accumulateOuterProduct(buf.grad_W_enc2, buf.h1, buf.dL_dh2);
        accumulateVector(buf.grad_b_enc2, buf.dL_dh2);

        accumulateOuterProduct(buf.grad_W_mu, buf.h2, buf.grad_mu);
        accumulateVector(buf.grad_b_mu, buf.grad_mu);

        accumulateOuterProduct(buf.grad_W_logvar, buf.h2, buf.grad_logvar);
        accumulateVector(buf.grad_b_logvar, buf.grad_logvar);

        // --- Decoder ---
        concatInPlace(buf.z, c, buf.zc);

        accumulateOuterProduct(buf.grad_W_dec1, buf.zc, buf.dL_dDec1);
        accumulateVector(buf.grad_b_dec1, buf.dL_dDec1);

        accumulateOuterProduct(buf.grad_W_dec2, buf.d1, buf.dL_dDec2);
        accumulateVector(buf.grad_b_dec2, buf.dL_dDec2);

        accumulateOuterProduct(buf.grad_W_decOut, buf.d2, buf.grad_xRecon);
        accumulateVector(buf.grad_b_decOut, buf.grad_xRecon);

        return loss;
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
        Random rand = threadLocalRandom.get();
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

    /**
     * @return the useSIMD
     */
    public boolean isUseSIMD() {
        return useSIMD;
    }

    /**
     * @param useSIMD the useSIMD to set
     */
    public void setUseSIMD(boolean useSIMD) {
        this.useSIMD = useSIMD;
    }
    //</editor-fold>    
}
