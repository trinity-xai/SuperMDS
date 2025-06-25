package SuperMDS;

import static SuperMDS.CVAEHelper.accumulateOuterProduct;
import static SuperMDS.CVAEHelper.accumulateVector;
import static SuperMDS.CVAEHelper.add;
import static SuperMDS.CVAEHelper.addInPlace;
import static SuperMDS.CVAEHelper.applyDropout;
import static SuperMDS.CVAEHelper.applyUpdate;
import static SuperMDS.CVAEHelper.clipGradientInPlace;
import static SuperMDS.CVAEHelper.concat;
import static SuperMDS.CVAEHelper.concatInPlace;
import static SuperMDS.CVAEHelper.dot;
import static SuperMDS.CVAEHelper.dotT;
import static SuperMDS.CVAEHelper.getCyclicalKLWeightSigmoid;
import static SuperMDS.CVAEHelper.getKLWeight;
import static SuperMDS.CVAEHelper.hasNaNsOrInfs;
import static SuperMDS.CVAEHelper.initMatrix;
import static SuperMDS.CVAEHelper.initVector;
import static SuperMDS.CVAEHelper.mseGradient;
import static SuperMDS.CVAEHelper.mseLoss;
import static SuperMDS.CVAEHelper.relu;
import static SuperMDS.CVAEHelper.reluGradInPlace;
import static SuperMDS.CVAEHelper.reluInPlace;
import static SuperMDS.CVAEHelper.sampleLatentInPlace;
import static SuperMDS.CVAEHelper.updateMatrix;
import static SuperMDS.CVAEHelper.updateVector;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
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

    // Encoder weights 
    private double[][] W_enc1, W_enc2; // Weights for encoder layers
    private double[] b_enc1, b_enc2;   // Biases for encoder layers
    private double[][] W_mu;       // Weights: hiddenDim x latentDim
    private double[] b_mu;         // Biases: latentDim
    private double[][] W_logvar;   // Weights: hiddenDim x latentDim
    private double[] b_logvar;     // Biases: latentDim

    // Decoder weight matrices for 2 hidden layers + output layer
    private double[][] W_dec1, W_dec2, W_decOut; // Weights for decoder layers
    private double[] b_dec1, b_dec2, b_decOut; // Biases for decoder layers
    ThreadLocal<Random> threadLocalRandom;    
    ThreadLocal<BufferSet> threadLocalBuffers;
    ThreadLocal<GradientBuffer> threadLocalGrads;
    
    
    // === Annealing settings ===
    private AtomicInteger currentEpoch = new AtomicInteger(0);
    private int klWarmupEpochs = 100;
    private double maxKLWeight = 0.5;
    private double klSharpness = 10.0;
    private double learningRate = 0.0001;
    private boolean useCyclicalAnneal = false;
    private int klAnnealCycleLength = 100;
    private double dropoutRate = 0.01; // 20% dropout is typical
    private boolean useDropout = true;    
    private boolean isTraining = false;
    private long seed = 42L;
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
            -> new Random(seed));
        threadLocalBuffers = ThreadLocal.withInitial(() ->
            new BufferSet(inputDim, conditionDim, latentDim, hiddenDim));    
        threadLocalGrads = ThreadLocal.withInitial(() -> 
            new GradientBuffer(inputDim, conditionDim, latentDim, hiddenDim));

         int encInputDim = inputDim + conditionDim;    // Encoder input: [x | c]
         int decInputDim = latentDim + conditionDim;   // Decoder input: [z | c]
         Random rand = threadLocalRandom.get();
         // Encoder 
         W_enc1 = initMatrix(encInputDim, hiddenDim, true, rand);  // He init for ReLU
         b_enc1 = initVector(hiddenDim);

         W_enc2 = initMatrix(hiddenDim, hiddenDim, true, rand);    // He init
         b_enc2 = initVector(hiddenDim);

         W_mu = initMatrix(hiddenDim, latentDim, false, rand);     // Xavier init
         b_mu = initVector(latentDim);

         W_logvar = initMatrix(hiddenDim, latentDim, false, rand); // Xavier init
         b_logvar = initVector(latentDim);

         // Decoder
         W_dec1 = initMatrix(decInputDim, hiddenDim, true, rand);  // He init
         b_dec1 = initVector(hiddenDim);

         W_dec2 = initMatrix(hiddenDim, hiddenDim, true, rand);    // He init
         b_dec2 = initVector(hiddenDim);

         W_decOut = initMatrix(hiddenDim, inputDim, false, rand);  // Xavier init
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
 * Trains a batch of (x, c) pairs using forward-backward and accumulates gradients in parallel.
 * Each thread uses thread-local buffers and gradients to avoid contention.
 * Final gradients are averaged and applied once after the batch.
 *
 * @param xBatch   Input batch (batchSize × inputDim)
 * @param cBatch   Condition batch (batchSize × conditionDim)
 * @return Average loss across the batch
 */
public double trainBatchParallel(double[][] xBatch, double[][] cBatch) {
    int batchSize = xBatch.length;

    // Master gradient buffer for final averaged update
    GradientBuffer masterGrad = new GradientBuffer(inputDim, conditionDim, latentDim, hiddenDim);

//    // Thread-local buffers
//    ThreadLocal<CVAEBufferSet> localBuf = ThreadLocal.withInitial(() ->
//        new CVAEBufferSet(inputDim, conditionDim, latentDim, hiddenDim)
//    );
//
//    ThreadLocal<GradientBuffer> localGrad = ThreadLocal.withInitial(() ->
//        new GradientBuffer(inputDim, conditionDim, latentDim, hiddenDim)
//    );
//
//    ThreadLocal<Random> localRng = ThreadLocal.withInitial(() ->
//        new Random(seed ^ Thread.currentThread().getId())
//    );
//    ThreadLocal<Random> threadLocalRandom;    
//    ThreadLocal<BufferSet> threadLocalBuffers;
//    ThreadLocal<GradientBuffer> threadLocalGrads;
    
    double totalLoss = IntStream.range(0, batchSize).parallel().mapToDouble(i -> {
        double[] x = xBatch[i];
        double[] c = cBatch[i];
        BufferSet buf = threadLocalBuffers.get();
        GradientBuffer grad = threadLocalGrads.get();
        Random rng = threadLocalRandom.get();

        // Zero out gradients from last run
        grad.clear();

        // Compute forward + backward pass and accumulate gradients locally
        double loss = forwardBackward(x, c, buf, grad, rng);

        // Add local gradients to shared master
        synchronized (masterGrad) {
            masterGrad.accumulate(grad);
        }

        return loss;
    }).sum();

    // Apply averaged gradients
    applyGradients(masterGrad, 1.0 / batchSize);

    return totalLoss / batchSize;
}


/**
 * Performs the forward and backward pass on a single (x, c) input pair using reusable buffers.
 * Computes loss and accumulates gradients into the provided GradientBuffer.
 *
 * @param x         Input vector
 * @param c         Condition vector
 * @param buf       Thread-local buffer set for forward/backward intermediate values
 * @param grad      Thread-local gradient buffer (accumulated into)
 * @param rng       Random number generator (thread-local)
 * @return Loss for the single (x, c) sample
 */
public double forwardBackward(
    double[] x,
    double[] c,
    BufferSet buf,
    GradientBuffer grad,
    Random rng
) {
    // === Forward pass ===
    concatInPlace(x, c, buf.xc); // buf.xc = [x | c]

    dot(buf.xc, W_enc1, buf.h1);
    addInPlace(buf.h1, b_enc1);
    reluInPlace(buf.h1);

    dot(buf.h1, W_enc2, buf.h2);
    addInPlace(buf.h2, b_enc2);
    reluInPlace(buf.h2);

    dot(buf.h2, W_mu, buf.mu);
    addInPlace(buf.mu, b_mu);

    dot(buf.h2, W_logvar, buf.logvar);
    addInPlace(buf.logvar, b_logvar);

    for (int i = 0; i < latentDim; i++) {
        buf.safeLogvar[i] = Math.max(Math.min(buf.logvar[i], 10.0), -10.0);
    }

    sampleLatentInPlace(buf.mu, buf.safeLogvar, buf.z, rng);
    for (int i = 0; i < buf.z.length; i++) {
        buf.z[i] = Math.max(Math.min(buf.z[i], 10.0), -10.0);
    }

    concatInPlace(buf.z, c, buf.zc);

    dot(buf.zc, W_dec1, buf.d1);
    addInPlace(buf.d1, b_dec1);
    reluInPlace(buf.d1);

    dot(buf.d1, W_dec2, buf.d2);
    addInPlace(buf.d2, b_dec2);
    reluInPlace(buf.d2);

    dot(buf.d2, W_decOut, buf.xRecon);
    addInPlace(buf.xRecon, b_decOut);

    for (int i = 0; i < buf.xRecon.length; i++) {
        buf.xRecon[i] = Math.max(Math.min(buf.xRecon[i], 1e6), -1e6);
    }

    // === Loss ===
    double reconLoss = mseLoss(x, buf.xRecon);
    double klLoss = 0.0;
    for (int i = 0; i < latentDim; i++) {
        double var = Math.exp(buf.safeLogvar[i]);
        klLoss += -0.5 * (1 + buf.safeLogvar[i] - buf.mu[i] * buf.mu[i] - var);
    }

    double klWeight = isUseCyclicalAnneal()
        ? getCyclicalKLWeightSigmoid(currentEpoch.get(), klAnnealCycleLength, maxKLWeight, klSharpness)
        : getKLWeight(currentEpoch.get(), klWarmupEpochs, maxKLWeight, klSharpness);

    double totalLoss = reconLoss + klWeight * klLoss;

    if (Double.isNaN(totalLoss) || Double.isInfinite(totalLoss)) {
        throw new RuntimeException("NaN or Inf encountered in forwardBackward()");
    }

    // === Backward pass ===
    mseGradient(buf.xRecon, x, buf.grad_xRecon);
    dotT(buf.grad_xRecon, W_decOut, buf.dL_dDec2);
    reluGradInPlace(buf.d2, buf.dL_dDec2, buf.dL_dDec2);

    dotT(buf.dL_dDec2, W_dec2, buf.dL_dDec1);
    reluGradInPlace(buf.d1, buf.dL_dDec1, buf.dL_dDec1);

    dotT(buf.dL_dDec1, W_dec1, buf.dL_dZC);
    System.arraycopy(buf.dL_dZC, 0, buf.dz, 0, latentDim);

    for (int i = 0; i < latentDim; i++) {
        double sigma = Math.exp(0.5 * buf.safeLogvar[i]);
        double eps = (buf.z[i] - buf.mu[i]) / sigma;
        buf.grad_mu[i] = buf.dz[i] + klWeight * buf.mu[i];
        buf.grad_logvar[i] = 0.5 * buf.dz[i] * eps + klWeight * 0.5 * (Math.exp(buf.safeLogvar[i]) - 1);
    }

    dotT(buf.grad_mu, W_mu, buf.dmu_dh2);
    dotT(buf.grad_logvar, W_logvar, buf.dlogvar_dh2);
    addInPlace(buf.dmu_dh2, buf.dlogvar_dh2, buf.dL_dh2);

    reluGradInPlace(buf.h2, buf.dL_dh2, buf.dL_dh2);
    dotT(buf.dL_dh2, W_enc2, buf.dL_dh1);
    reluGradInPlace(buf.h1, buf.dL_dh1, buf.dL_dh1);

    // === Clip gradients ===
    clipGradientInPlace(buf.grad_xRecon, 5.0);
    clipGradientInPlace(buf.grad_mu, 5.0);
    clipGradientInPlace(buf.grad_logvar, 5.0);
    clipGradientInPlace(buf.dL_dh2, 5.0);
    clipGradientInPlace(buf.dL_dh1, 5.0);

    // === Accumulate gradients ===
    accumulateOuterProduct(grad.grad_W_enc1, buf.xc, buf.dL_dh1);
    accumulateVector(grad.grad_b_enc1, buf.dL_dh1);

    accumulateOuterProduct(grad.grad_W_enc2, buf.h1, buf.dL_dh2);
    accumulateVector(grad.grad_b_enc2, buf.dL_dh2);

    accumulateOuterProduct(grad.grad_W_mu, buf.h2, buf.grad_mu);
    accumulateVector(grad.grad_b_mu, buf.grad_mu);

    accumulateOuterProduct(grad.grad_W_logvar, buf.h2, buf.grad_logvar);
    accumulateVector(grad.grad_b_logvar, buf.grad_logvar);

    accumulateOuterProduct(grad.grad_W_dec1, buf.zc, buf.dL_dDec1);
    accumulateVector(grad.grad_b_dec1, buf.dL_dDec1);

    accumulateOuterProduct(grad.grad_W_dec2, buf.d1, buf.dL_dDec2);
    accumulateVector(grad.grad_b_dec2, buf.dL_dDec2);

    accumulateOuterProduct(grad.grad_W_decOut, buf.d2, buf.grad_xRecon);
    accumulateVector(grad.grad_b_decOut, buf.grad_xRecon);

    return totalLoss;
}

private void applyGradients(GradientBuffer grad, double lr) {
    applyUpdate(W_enc1, grad.grad_W_enc1, lr);
    applyUpdate(b_enc1, grad.grad_b_enc1, lr);
    applyUpdate(W_enc2, grad.grad_W_enc2, lr);
    applyUpdate(b_enc2, grad.grad_b_enc2, lr);
    applyUpdate(W_mu, grad.grad_W_mu, lr);
    applyUpdate(b_mu, grad.grad_b_mu, lr);
    applyUpdate(W_logvar, grad.grad_W_logvar, lr);
    applyUpdate(b_logvar, grad.grad_b_logvar, lr);
    applyUpdate(W_dec1, grad.grad_W_dec1, lr);
    applyUpdate(b_dec1, grad.grad_b_dec1, lr);
    applyUpdate(W_dec2, grad.grad_W_dec2, lr);
    applyUpdate(b_dec2, grad.grad_b_dec2, lr);
    applyUpdate(W_decOut, grad.grad_W_decOut, lr);
    applyUpdate(b_decOut, grad.grad_b_decOut, lr);
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
        BufferSet buf = threadLocalBuffers.get(); // Reuse buffer set
        Random rng = threadLocalRandom.get();

        // ===== Forward Pass =====
        concatInPlace(x, c, buf.xc);
        dot(buf.xc, W_enc1, buf.h1);
        addInPlace(buf.h1, b_enc1);
        reluInPlace(buf.h1);

        dot(buf.h1, W_enc2, buf.h2);
        addInPlace(buf.h2, b_enc2);
        reluInPlace(buf.h2);

        dot(buf.h2, W_mu, buf.mu);
        addInPlace(buf.mu, b_mu);

        dot(buf.h2, W_logvar, buf.logvar);
        addInPlace(buf.logvar, b_logvar);

        for (int i = 0; i < latentDim; i++) {
            buf.safeLogvar[i] = Math.max(Math.min(buf.logvar[i], 10.0), -10.0);
        }

        sampleLatentInPlace(buf.mu, buf.safeLogvar, buf.z, rng);
        for (int i = 0; i < buf.z.length; i++) {
            buf.z[i] = Math.max(Math.min(buf.z[i], 10.0), -10.0);
        }

        concatInPlace(buf.z, c, buf.zc);
        dot(buf.zc, W_dec1, buf.d1);
        addInPlace(buf.d1, b_dec1);
        reluInPlace(buf.d1);

        dot(buf.d1, W_dec2, buf.d2);
        addInPlace(buf.d2, b_dec2);
        reluInPlace(buf.d2);

        dot(buf.d2, W_decOut, buf.xRecon);
        addInPlace(buf.xRecon, b_decOut);

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
                ? getCyclicalKLWeightSigmoid(currentEpoch.get(), klAnnealCycleLength, maxKLWeight, klSharpness)
                : getKLWeight(currentEpoch.get(), klWarmupEpochs, maxKLWeight, klSharpness);

        double loss = reconLoss + klWeight * klLoss;

        if (Double.isNaN(loss) || Double.isInfinite(loss)) {
            if (debug) debugMe(currentEpoch.get(), reconLoss, klLoss, klWeight, loss);
            throw new RuntimeException("Training loss became NaN or Infinite");
        }

        // ===== Backward Pass =====
        mseGradient(buf.xRecon, x, buf.grad_xRecon);

        dotT(buf.grad_xRecon, W_decOut, buf.dL_dDec2);
        reluGradInPlace(buf.d2, buf.dL_dDec2, buf.dL_dDec2);

        dotT(buf.dL_dDec2, W_dec2, buf.dL_dDec1);
        reluGradInPlace(buf.d1, buf.dL_dDec1, buf.dL_dDec1);

        dotT(buf.dL_dDec1, W_dec1, buf.dL_dZC);
        System.arraycopy(buf.dL_dZC, 0, buf.dz, 0, latentDim);

        for (int i = 0; i < latentDim; i++) {
            double sigma = Math.exp(0.5 * buf.safeLogvar[i]);
            double eps = (buf.z[i] - buf.mu[i]) / sigma;
            buf.grad_mu[i] = buf.dz[i] + klWeight * buf.mu[i];
            buf.grad_logvar[i] = 0.5 * buf.dz[i] * eps + klWeight * 0.5 * (Math.exp(buf.safeLogvar[i]) - 1);
        }

        dotT(buf.grad_mu, W_mu, buf.dmu_dh2);
        dotT(buf.grad_logvar, W_logvar, buf.dlogvar_dh2);
        addInPlace(buf.dmu_dh2, buf.dlogvar_dh2, buf.dL_dh2);

        reluGradInPlace(buf.h2, buf.dL_dh2, buf.dL_dh2);
        dotT(buf.dL_dh2, W_enc2, buf.dL_dh1);
        reluGradInPlace(buf.h1, buf.dL_dh1, buf.dL_dh1);

        // Clip grads
        clipGradientInPlace(buf.grad_xRecon, 5.0);
        clipGradientInPlace(buf.grad_mu, 5.0);
        clipGradientInPlace(buf.grad_logvar, 5.0);
        clipGradientInPlace(buf.dL_dh2, 5.0);
        clipGradientInPlace(buf.dL_dh1, 5.0);

        // ===== Parameter Updates =====
        updateParametersDeep(
                buf.xc, buf.h1, buf.h2,
                buf.dL_dh1, buf.dL_dh2,
                buf.grad_mu, buf.grad_logvar,
                buf.z, c, buf.d1, buf.d2,
                buf.dL_dDec1, buf.dL_dDec2, buf.grad_xRecon
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
    private void debugMe(double currentEpoch, double reconLoss, double klLoss, double klWeight, double loss){
//        System.out.println("x = " + Arrays.toString(x));
//        System.out.println("c = " + Arrays.toString(c));
//        System.out.println("xRecon = " + Arrays.toString(xRecon));
//        System.out.println("mu = " + Arrays.toString(mu));
//        System.out.println("logvar = " + Arrays.toString(logvar));
//        System.out.println("safeLogvar = " + Arrays.toString(safeLogvar));
//        System.out.println("z = " + Arrays.toString(z));
        if (currentEpoch % getDebugEpochCount() == 0) {
            System.out.printf("Epoch %d — Recon: %.6f, KL: %.6f (weight %.3f), Total: %.6f\n",
                currentEpoch, reconLoss, klLoss, klWeight, loss);
        }
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
        if (useDropout && isIsTraining()) h1 = applyDropout(h1, dropoutRate, rand);
        
        // Step 3: Hidden layer 2
        double[] h2 = relu(add(dot(h1, W_dec2), b_dec2)); // [hiddenDim]
        if (useDropout && isIsTraining()) h2 = applyDropout(h2, dropoutRate, rand);
        
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
        if (useDropout) h1 = applyDropout(h1, dropoutRate, rand);
        
        // Step 3: Hidden layer 2
        double[] h2 = relu(add(dot(h1, W_enc2), b_enc2)); // [hiddenDim]
        if (useDropout) h2 = applyDropout(h2, dropoutRate, rand);
        
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
