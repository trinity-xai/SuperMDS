package SuperMDS;

import static SuperMDS.CVAEHelper.add;
import static SuperMDS.CVAEHelper.addInPlace;
import static SuperMDS.CVAEHelper.applyDropout;
import static SuperMDS.CVAEHelper.applyDropoutInPlace;
import static SuperMDS.CVAEHelper.clipGradient;
import static SuperMDS.CVAEHelper.concat;
import static SuperMDS.CVAEHelper.concatInPlace;
import static SuperMDS.CVAEHelper.dot;
import static SuperMDS.CVAEHelper.dotInPlace;
import static SuperMDS.CVAEHelper.dotT;
import static SuperMDS.CVAEHelper.dotTInPlace;
import static SuperMDS.CVAEHelper.getCyclicalKLWeightSigmoid;
import static SuperMDS.CVAEHelper.getKLWeight;
import static SuperMDS.CVAEHelper.hasNaNsOrInfs;
import static SuperMDS.CVAEHelper.initMatrix;
import static SuperMDS.CVAEHelper.initVector;
import static SuperMDS.CVAEHelper.mseGradient;
import static SuperMDS.CVAEHelper.mseGradientInPlace;
import static SuperMDS.CVAEHelper.mseLoss;
import static SuperMDS.CVAEHelper.relu;
import static SuperMDS.CVAEHelper.reluGrad;
import static SuperMDS.CVAEHelper.reluGradInPlace;
import static SuperMDS.CVAEHelper.reluInPlace;
import static SuperMDS.CVAEHelper.sampleLatent;
import static SuperMDS.CVAEHelper.sampleLatentInPlace;
import static SuperMDS.CVAEHelper.slice;
import static SuperMDS.CVAEHelper.updateMatrix;
import static SuperMDS.CVAEHelper.updateVector;
import static SuperMDS.SIMDMath.transposeMatrix;
import static SuperMDS.SIMDMath.updateMatrix_SIMD;
import static SuperMDS.SIMDMath.updateVector_SIMD;
import java.util.Arrays;
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
public class CVAESIMD {

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
    
    // Transposed versions for SIMD-friendly access
    private double[][] W_enc1_T, W_enc2_T;
    private double[][] W_mu_T, W_logvar_T;
    private double[][] W_dec1_T, W_dec2_T, W_decOut_T;

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
    private boolean useSIMD = false;
    
    private final ThreadLocal<BufferSet> threadBuffers;
    private final ThreadLocal<RandomBuffer> threadRngBuffer;
    ThreadLocal<Random> threadLocalRandom;    
    long seed = 42L;
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
    public CVAESIMD(int inputDim, int conditionDim, int latentDim, int hiddenDim) {
        this.inputDim = inputDim;
         this.conditionDim = conditionDim;
         this.latentDim = latentDim;
         this.hiddenDim = hiddenDim;

         threadBuffers = ThreadLocal.withInitial(() ->
             new BufferSet(inputDim, conditionDim, latentDim, hiddenDim));
         threadRngBuffer = ThreadLocal.withInitial(() -> new RandomBuffer(4096));
        threadLocalRandom = ThreadLocal.withInitial(() 
            -> new Random(seed));
        Random rand = threadLocalRandom.get();
         int encInputDim = inputDim + conditionDim;    // Encoder input: [x | c]
         int decInputDim = latentDim + conditionDim;   // Decoder input: [z | c]

         // === Encoder ===
         W_enc1 = initMatrix(encInputDim, hiddenDim, true, rand);  // He init for ReLU
         W_enc1_T = transposeMatrix(W_enc1);
         b_enc1 = initVector(hiddenDim);

         W_enc2 = initMatrix(hiddenDim, hiddenDim, true, rand);    // He init
         W_enc2_T = transposeMatrix(W_enc2);
         b_enc2 = initVector(hiddenDim);

         W_mu = initMatrix(hiddenDim, latentDim, false, rand);     // Xavier init
         W_mu_T = transposeMatrix(W_mu);
         b_mu = initVector(latentDim);

         W_logvar = initMatrix(hiddenDim, latentDim, false, rand); // Xavier init
         W_logvar_T = transposeMatrix(W_logvar);
         b_logvar = initVector(latentDim);

         // === Decoder ===
         W_dec1 = initMatrix(decInputDim, hiddenDim, true, rand);  // He init
         W_dec1_T = transposeMatrix(W_dec1);
         b_dec1 = initVector(hiddenDim);

         W_dec2 = initMatrix(hiddenDim, hiddenDim, true, rand);    // He init
         W_dec2_T = transposeMatrix(W_dec2);
         b_dec2 = initVector(hiddenDim);

         //Row Major way
        //W_decOut = initMatrix(hiddenDim, inputDim, false);  // Xavier init
        //Column Major Way (needed for SIMD)
        W_decOut = initMatrix(inputDim, hiddenDim, false, rand); // ✅
        W_decOut_T = transposeMatrix(W_decOut);
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
        if (useDropout) h1 = applyDropout(h1, dropoutRate, rand);
        
        // Step 3: Hidden layer 2
        double[] h2 = relu(add(dot(h1, W_enc2), b_enc2)); // [hiddenDim]
        if (useDropout) h2 = applyDropout(h2, dropoutRate, rand);
        
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
        if (useDropout && isIsTraining()) h1 = applyDropout(h1, dropoutRate, rand);
        
        // Step 3: Hidden layer 2
        double[] h2 = relu(add(dot(h1, W_dec2), b_dec2)); // [hiddenDim]
        if (useDropout && isIsTraining()) h2 = applyDropout(h2, dropoutRate, rand);
        
        // Step 4: Final linear output (no activation)
        double[] out = add(dot(h2, W_decOut), b_decOut); // [inputDim]

        return out;
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

        double[] xc = concat(x, c);

        // ===== Forward Pass =====
        // Encoder
        double[] h1 = relu(add(dot(xc, W_enc1), b_enc1));
        double[] h2 = relu(add(dot(h1, W_enc2), b_enc2));

        double[] mu = add(dot(h2, W_mu), b_mu);
        double[] logvar = add(dot(h2, W_logvar), b_logvar);

        double[] safeLogvar = new double[latentDim];
        for (int i = 0; i < latentDim; i++) {
            safeLogvar[i] = Math.max(Math.min(logvar[i], 10.0), -10.0);
        }

        double[] z = sampleLatent(mu, safeLogvar, threadLocalRandom.get());
        for (int i = 0; i < z.length; i++) {
            if (Math.abs(z[i]) > 10.0) {
                z[i] = Math.signum(z[i]) * 10.0;
            }
        }

        double[] zc = concat(z, c);

        // Decoder
        double[] d1 = relu(add(dot(zc, W_dec1), b_dec1));
        double[] d2 = relu(add(dot(d1, W_dec2), b_dec2));
        double[] xRecon = add(dot(d2, W_decOut), b_decOut);  // Linear output

        for (int i = 0; i < xRecon.length; i++) {
            xRecon[i] = Math.max(Math.min(xRecon[i], 1e6), -1e6);
        }

        // ===== Loss =====
        double reconLoss = mseLoss(x, xRecon);
        double klLoss = 0.0;
        for (int i = 0; i < latentDim; i++) {
            double var = Math.exp(safeLogvar[i]);
            klLoss += -0.5 * (1 + safeLogvar[i] - mu[i] * mu[i] - var);
        }

        //should we use Sawtooth annealing (cyclical) or a monotonic rampup
        double klWeight = isUseCyclicalAnneal()
                ? getCyclicalKLWeightSigmoid(currentEpoch.get(), getKlAnnealCycleLength(), maxKLWeight, getKlSharpness())
                : getKLWeight(currentEpoch.get(), klWarmupEpochs, maxKLWeight, getKlSharpness());

        double loss = reconLoss + klWeight * klLoss;

        if (Double.isNaN(loss) || Double.isInfinite(loss)) {
            if(debug) {
                System.out.println("x = " + Arrays.toString(x));
                System.out.println("c = " + Arrays.toString(c));
                System.out.println("xRecon = " + Arrays.toString(xRecon));
                System.out.println("mu = " + Arrays.toString(mu));
                System.out.println("logvar = " + Arrays.toString(logvar));
                System.out.println("safeLogvar = " + Arrays.toString(safeLogvar));
                System.out.println("z = " + Arrays.toString(z));
            }
            throw new RuntimeException("Training loss became NaN or Infinite — check input data or model stability.");
        }

        // Optional debug output
        if (debug && currentEpoch.get() % getDebugEpochCount() == 0) {
            System.out.printf("Epoch %d — Recon: %.6f, KL: %.6f (weight %.3f), Total: %.6f\n",
                currentEpoch.get(), reconLoss, klLoss, klWeight, loss);
        }

        // ===== Backward Pass =====
        double[] dL_dxRecon = mseGradient(xRecon, x);

        // Decoder
        double[] dL_dDecOut = dotT(dL_dxRecon, W_decOut);
        double[] dL_dDec2 = reluGrad(d2, dL_dDecOut);
        double[] dL_dDec1 = reluGrad(d1, dotT(dL_dDec2, W_dec2));
        double[] dL_dZC = dotT(dL_dDec1, W_dec1);

        double[] dz = slice(dL_dZC, 0, latentDim);

        // ===== KL Divergence Backpropagation (Fixed) =====
        double[] dL_dmu = new double[latentDim];
        double[] dL_dlogvar = new double[latentDim];
        for (int i = 0; i < latentDim; i++) {
            double sigma = Math.exp(0.5 * safeLogvar[i]);
            double eps = (z[i] - mu[i]) / sigma;

            // Backprop from decoder + KL term
            dL_dmu[i] = dz[i] + klWeight * mu[i];  // ∂KL/∂μ = μ
            dL_dlogvar[i] = 0.5 * dz[i] * eps + klWeight * 0.5 * (Math.exp(safeLogvar[i]) - 1);
        }

        // Encoder
        double[] dmu_dh2 = dotT(dL_dmu, W_mu);
        double[] dlogvar_dh2 = dotT(dL_dlogvar, W_logvar);
        double[] dL_dh2 = reluGrad(h2, add(dmu_dh2, dlogvar_dh2));
        double[] dL_dh1 = reluGrad(h1, dotT(dL_dh2, W_enc2));

        // ===== Gradient Clipping =====
        clipGradient(dL_dxRecon, 5.0);
        clipGradient(dL_dmu, 5.0);
        clipGradient(dL_dlogvar, 5.0);
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

        currentEpoch.incrementAndGet();
        return loss;
    }

    public double trainBatchInPlaceDynamic(boolean useSIMD, double[][] xBatch, double[][] cBatch) {
        return useSIMD
            ? trainBatchInPlaceSIMD(xBatch, cBatch)
            : trainBatchInPlace(xBatch, cBatch);
    }    
    //In place versions
    public double trainBatchInPlace(double[][] xBatch, double[][] cBatch) {
        int batchSize = xBatch.length;
        double totalLoss = IntStream.range(0, batchSize).parallel()
            .mapToDouble(i -> trainInPlace(xBatch[i], cBatch[i]))
            .sum();
        return totalLoss / batchSize;
    }    
    public double trainInPlace(double[] x, double[] c) {
        BufferSet buf = threadBuffers.get();
        buf.reset();
        Random rand = threadLocalRandom.get();
        // ========== Forward Pass ==========
        concatInPlace(x, c, buf.xc);  // buf.xc = x ⊕ c
        dotInPlace(buf.xc, W_enc1, buf.h1);           // h1 = relu(W1·xc + b1)
        addInPlace(buf.h1, b_enc1);
        reluInPlace(buf.h1);
        if (useDropout && isIsTraining()) applyDropoutInPlace(buf.h1, dropoutRate, rand);

        dotInPlace(buf.h1, W_enc2, buf.h2);           // h2 = relu(W2·h1 + b2)
        addInPlace(buf.h2, b_enc2);
        reluInPlace(buf.h2);
        if (useDropout && isIsTraining()) applyDropoutInPlace(buf.h2, dropoutRate, rand);

        dotInPlace(buf.h2, W_mu, buf.mu);             // mu = W_mu·h2 + b_mu
        addInPlace(buf.mu, b_mu);

        dotInPlace(buf.h2, W_logvar, buf.logvar);     // logvar = W_logvar·h2 + b_logvar
        addInPlace(buf.logvar, b_logvar);

        // Clamp logvar and sample z using reparam trick
        for (int i = 0; i < latentDim; i++) {
            buf.safeLogvar[i] = Math.max(Math.min(buf.logvar[i], 10.0), -10.0);
        }
        sampleLatentInPlace(buf.mu, buf.safeLogvar, buf.z, threadLocalRandom.get());
        for (int i = 0; i < latentDim; i++) {
            if (Math.abs(buf.z[i]) > 10.0)
                buf.z[i] = Math.signum(buf.z[i]) * 10.0;
        }

        concatInPlace(buf.z, c, buf.zc);              // zc = z ⊕ c
        dotInPlace(buf.zc, W_dec1, buf.d1);           // d1 = relu(W1·zc + b1)
        addInPlace(buf.d1, b_dec1);
        reluInPlace(buf.d1);
        if (useDropout && isIsTraining()) applyDropoutInPlace(buf.d1, dropoutRate, rand);

        dotInPlace(buf.d1, W_dec2, buf.d2);           // d2 = relu(W2·d1 + b2)
        addInPlace(buf.d2, b_dec2);
        reluInPlace(buf.d2);
        if (useDropout && isIsTraining()) applyDropoutInPlace(buf.d2, dropoutRate, rand);

        dotInPlace(buf.d2, W_decOut, buf.xRecon);     // recon = W3·d2 + b3
        addInPlace(buf.xRecon, b_decOut);

        for (int i = 0; i < buf.xRecon.length; i++) {
            buf.xRecon[i] = Math.max(Math.min(buf.xRecon[i], 1e6), -1e6);
        }

        // ======== Loss =========
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
                System.out.println("NaN loss at epoch " + currentEpoch.get());
                System.out.println("x = " + Arrays.toString(x));
                System.out.println("c = " + Arrays.toString(c));
                System.out.println("xRecon = " + Arrays.toString(buf.xRecon));
            }
            throw new RuntimeException("NaN/Inf in loss");
        }

        if (debug && currentEpoch.get() % getDebugEpochCount() == 0) {
            System.out.printf("Epoch %d — Recon: %.6f, KL: %.6f (w=%.3f), Total: %.6f\n",
                currentEpoch.get(), reconLoss, klLoss, klWeight, loss);
        }

        // ========== Backward Pass ==========
        mseGradientInPlace(buf.xRecon, x, buf.grad_xRecon);
        dotTInPlace(buf.grad_xRecon, W_decOut, buf.dL_dDec2);
        reluGradInPlace(buf.d2, buf.dL_dDec2, buf.dL_dDec2);

        dotTInPlace(buf.dL_dDec2, W_dec2, buf.dL_dDec1);
        reluGradInPlace(buf.d1, buf.dL_dDec1, buf.dL_dDec1);

        dotTInPlace(buf.dL_dDec1, W_dec1, buf.dL_dZC);
        System.arraycopy(buf.dL_dZC, 0, buf.dz, 0, latentDim);

        // KL Gradients
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
    
    //SIMD In Place versions
    public double trainBatchInPlaceSIMD(double[][] xBatch, double[][] cBatch) {
        int batchSize = xBatch.length;
//        double totalLoss = IntStream.range(0, batchSize).parallel()
//            .mapToDouble(i -> trainInPlaceSIMD(xBatch[i], cBatch[i]))
//            .sum();
        double totalLoss = 0.0;
        for (int i = 0; i < batchSize; i++) {
            totalLoss += trainInPlaceSIMD(xBatch[i], cBatch[i]);
        }
        return totalLoss / batchSize;
    }
    public double trainInPlaceSIMD(double[] x, double[] c) {
        BufferSet buf = threadBuffers.get();
        RandomBuffer rng = threadRngBuffer.get();
        buf.reset();

        // ======= Forward Pass =======
        concatInPlace(x, c, buf.xc);
        SIMDMath.dotSIMD(buf.xc, W_enc1, buf.h1);
        SIMDMath.addInPlaceSIMD(buf.h1, b_enc1, buf.h1);
        SIMDMath.reluInPlaceSIMD(buf.h1);
        if (useDropout && isIsTraining()) {
            SIMDMath.applyDropoutInPlaceSIMD(buf.h1, buf.h1, dropoutRate, rng);
        }

        SIMDMath.dotSIMD(buf.h1, W_enc2, buf.h2);
        SIMDMath.addInPlaceSIMD(buf.h2, b_enc2, buf.h2);
        SIMDMath.reluInPlaceSIMD(buf.h2);
        if (useDropout && isIsTraining()) {
            SIMDMath.applyDropoutInPlaceSIMD(buf.h2, buf.h2, dropoutRate, rng);
        }

        SIMDMath.dotSIMD(buf.h2, W_mu, buf.mu);
        SIMDMath.addInPlaceSIMD(buf.mu, b_mu, buf.mu);

        SIMDMath.dotSIMD(buf.h2, W_logvar, buf.logvar);
        SIMDMath.addInPlaceSIMD(buf.logvar, b_logvar, buf.logvar);

        for (int i = 0; i < latentDim; i++) {
            buf.safeLogvar[i] = Math.max(Math.min(buf.logvar[i], 10.0), -10.0);
        }

        SIMDMath.sampleLatentInPlaceSIMD(buf.mu, buf.safeLogvar, buf.z, rng);
        for (int i = 0; i < latentDim; i++) {
            if (Math.abs(buf.z[i]) > 10.0) {
                buf.z[i] = Math.signum(buf.z[i]) * 10.0;
            }
        }

        concatInPlace(buf.z, c, buf.zc);
        SIMDMath.dotSIMD(buf.zc, W_dec1, buf.d1);
        SIMDMath.addInPlaceSIMD(buf.d1, b_dec1, buf.d1);
        SIMDMath.reluInPlaceSIMD(buf.d1);
        if (useDropout && isIsTraining()) {
            SIMDMath.applyDropoutInPlaceSIMD(buf.d1, buf.d1, dropoutRate, rng);
        }

        SIMDMath.dotSIMD(buf.d1, W_dec2, buf.d2);
        SIMDMath.addInPlaceSIMD(buf.d2, b_dec2, buf.d2);
        SIMDMath.reluInPlaceSIMD(buf.d2);
        if (useDropout && isIsTraining()) {
            SIMDMath.applyDropoutInPlaceSIMD(buf.d2, buf.d2, dropoutRate, rng);
        }

        //Row Major way
        //IMDMath.dotSIMD(buf.d2, W_decOut, buf.xRecon);
        //Column Major way
        SIMDMath.dotSIMD(buf.d2, W_decOut_T, buf.xRecon); 
        SIMDMath.addInPlaceSIMD(buf.xRecon, b_decOut, buf.xRecon);
        for (int i = 0; i < buf.xRecon.length; i++) {
            buf.xRecon[i] = Math.max(Math.min(buf.xRecon[i], 1e6), -1e6);
        }

        // ======== Loss (SIMD version) ========
        double reconLoss = SIMDMath.mseLossSIMD(x, buf.xRecon);

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
                System.out.println("NaN loss at epoch " + currentEpoch.get());
                System.out.println("x = " + Arrays.toString(x));
                System.out.println("c = " + Arrays.toString(c));
                System.out.println("xRecon = " + Arrays.toString(buf.xRecon));
            }
            throw new RuntimeException("NaN/Inf in loss");
        }

        if (debug && currentEpoch.get() % getDebugEpochCount() == 0) {
            System.out.printf("Epoch %d — Recon: %.6f, KL: %.6f (w=%.3f), Total: %.6f\n",
                currentEpoch.get(), reconLoss, klLoss, klWeight, loss);
        }

        // ======= Backward Pass (SIMD) =======
        SIMDMath.mseGradient_SIMD(buf.xRecon, x, buf.grad_xRecon);
        SIMDMath.dotSIMD_T(buf.grad_xRecon, W_decOut_T, buf.dL_dDec2);
        SIMDMath.reluGradInPlaceSIMD(buf.d2, buf.dL_dDec2, buf.dL_dDec2);

        SIMDMath.dotSIMD_T(buf.dL_dDec2, W_dec2_T, buf.dL_dDec1);
        SIMDMath.reluGradInPlaceSIMD(buf.d1, buf.dL_dDec1, buf.dL_dDec1);

//System.out.printf("W_dec1 dims = [%d][%d]\n", W_dec1.length, W_dec1[0].length);
//System.out.printf("W_dec1_T dims = [%d][%d]\n", W_dec1_T.length, W_dec1_T[0].length);
//System.out.printf("buf.dL_dDec1 length = %d\n", buf.dL_dDec1.length);

        SIMDMath.dotSIMD_T(buf.dL_dDec1, W_dec1_T, buf.dL_dZC);
        System.arraycopy(buf.dL_dZC, 0, buf.dz, 0, latentDim);

        for (int i = 0; i < latentDim; i++) {
            double sigma = Math.exp(0.5 * buf.safeLogvar[i]);
            double eps = (buf.z[i] - buf.mu[i]) / sigma;

            buf.grad_mu[i] = buf.dz[i] + klWeight * buf.mu[i];
            buf.grad_logvar[i] = 0.5 * buf.dz[i] * eps + klWeight * 0.5 * (Math.exp(buf.safeLogvar[i]) - 1);
        }

        SIMDMath.dotSIMD_T(buf.grad_mu, W_mu_T, buf.dmu_dh2);
        SIMDMath.dotSIMD_T(buf.grad_logvar, W_logvar_T, buf.dlogvar_dh2);
        SIMDMath.addInPlaceSIMD(buf.dmu_dh2, buf.dlogvar_dh2, buf.dL_dh2);

        SIMDMath.reluGradInPlaceSIMD(buf.h2, buf.dL_dh2, buf.dL_dh2);
        SIMDMath.dotSIMD_T(buf.dL_dh2, W_enc2_T, buf.dL_dh1);
        SIMDMath.reluGradInPlaceSIMD(buf.h1, buf.dL_dh1, buf.dL_dh1);

        // ======= Gradient Clipping =======
        SIMDMath.clipGradient_SIMD(buf.grad_xRecon, 5.0);
        SIMDMath.clipGradient_SIMD(buf.grad_mu, 5.0);
        SIMDMath.clipGradient_SIMD(buf.grad_logvar, 5.0);
        SIMDMath.clipGradient_SIMD(buf.dL_dh2, 5.0);
        SIMDMath.clipGradient_SIMD(buf.dL_dh1, 5.0);

        updateParametersDeepSIMD(
            buf.xc, buf.h1, buf.h2,
            buf.dL_dh1, buf.dL_dh2,
            buf.grad_mu, buf.grad_logvar,
            buf.z, c, buf.d1, buf.d2,
            buf.dL_dDec1, buf.dL_dDec2, buf.grad_xRecon
        );

        currentEpoch.incrementAndGet();
        return loss;
    }
    private void updateParametersDeepSIMD(
        double[] xc, double[] h1, double[] h2,
        double[] dh1, double[] dh2,
        double[] dmu, double[] dlogvar,
        double[] z, double[] c,
        double[] d1, double[] d2,
        double[] dL_dDec1, double[] dL_dDec2, double[] dL_dxRecon
) {
    // ----- Encoder -----
    updateMatrix_SIMD(W_enc1, xc, dh1, getLearningRate());
    updateVector_SIMD(b_enc1, dh1, getLearningRate());

    updateMatrix_SIMD(W_enc2, h1, dh2, getLearningRate());
    updateVector_SIMD(b_enc2, dh2, getLearningRate());

    updateMatrix_SIMD(W_mu, h2, dmu, getLearningRate());
    updateVector_SIMD(b_mu, dmu, getLearningRate());

    updateMatrix_SIMD(W_logvar, h2, dlogvar, getLearningRate());
    updateVector_SIMD(b_logvar, dlogvar, getLearningRate());

    // ----- Decoder -----
    double[] zc = concat(z, c);

    updateMatrix_SIMD(W_dec1, zc, dL_dDec1, getLearningRate());
    updateVector_SIMD(b_dec1, dL_dDec1, getLearningRate());

    updateMatrix_SIMD(W_dec2, d1, dL_dDec2, getLearningRate());
    updateVector_SIMD(b_dec2, dL_dDec2, getLearningRate());

    updateMatrix_SIMD(W_decOut, d2, dL_dxRecon, getLearningRate());
    updateVector_SIMD(b_decOut, dL_dxRecon, getLearningRate());

    // ----- Refresh Transposed Matrices for SIMD -----
    W_enc1_T = transposeMatrix(W_enc1);
    W_enc2_T = transposeMatrix(W_enc2);
    W_mu_T = transposeMatrix(W_mu);
    W_logvar_T = transposeMatrix(W_logvar);

    W_dec1_T = transposeMatrix(W_dec1);
    W_dec2_T = transposeMatrix(W_dec2);
    W_decOut_T = transposeMatrix(W_decOut);
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
public void verifyNetworkShapes() {
    int encInputDim = inputDim + conditionDim;
    int decInputDim = latentDim + conditionDim;

    // Check encoder shapes
    assertShape(W_enc1, encInputDim, hiddenDim, "W_enc1");
    assertShape(W_enc1_T, hiddenDim, encInputDim, "W_enc1_T");

    assertShape(W_enc2, hiddenDim, hiddenDim, "W_enc2");
    assertShape(W_enc2_T, hiddenDim, hiddenDim, "W_enc2_T");

    assertShape(W_mu, hiddenDim, latentDim, "W_mu");
    assertShape(W_mu_T, latentDim, hiddenDim, "W_mu_T");

    assertShape(W_logvar, hiddenDim, latentDim, "W_logvar");
    assertShape(W_logvar_T, latentDim, hiddenDim, "W_logvar_T");

    // Check decoder shapes
    assertShape(W_dec1, decInputDim, hiddenDim, "W_dec1");
    assertShape(W_dec1_T, hiddenDim, decInputDim, "W_dec1_T");

    assertShape(W_dec2, hiddenDim, hiddenDim, "W_dec2");
    assertShape(W_dec2_T, hiddenDim, hiddenDim, "W_dec2_T");

    assertShape(W_decOut, inputDim, hiddenDim, "W_decOut");
    assertShape(W_decOut_T, hiddenDim, inputDim, "W_decOut_T");

    // Check bias vectors
    assertLength(b_enc1, hiddenDim, "b_enc1");
    assertLength(b_enc2, hiddenDim, "b_enc2");
    assertLength(b_mu, latentDim, "b_mu");
    assertLength(b_logvar, latentDim, "b_logvar");

    assertLength(b_dec1, hiddenDim, "b_dec1");
    assertLength(b_dec2, hiddenDim, "b_dec2");
    assertLength(b_decOut, inputDim, "b_decOut");

    // Sample forward buffer checks
    BufferSet buf = threadBuffers.get();
    assertLength(buf.xc, encInputDim, "buf.xc");
    assertLength(buf.h1, hiddenDim, "buf.h1");
    assertLength(buf.h2, hiddenDim, "buf.h2");
    assertLength(buf.mu, latentDim, "buf.mu");
    assertLength(buf.logvar, latentDim, "buf.logvar");
    assertLength(buf.z, latentDim, "buf.z");
    assertLength(buf.zc, decInputDim, "buf.zc");
    assertLength(buf.d1, hiddenDim, "buf.d1");
    assertLength(buf.d2, hiddenDim, "buf.d2");
    assertLength(buf.xRecon, inputDim, "buf.xRecon");

    System.out.println("CVAE shape verification passed.");
}
private void assertShape(double[][] matrix, int expectedRows, int expectedCols, String name) {
    if (matrix.length != expectedRows || matrix[0].length != expectedCols) {
        throw new IllegalArgumentException(
            String.format("Shape mismatch in %s: expected [%d][%d], found [%d][%d]",
                name, expectedRows, expectedCols, matrix.length, matrix[0].length));
    }
}

private void assertLength(double[] vector, int expectedLength, String name) {
    if (vector.length != expectedLength) {
        throw new IllegalArgumentException(
            String.format("Length mismatch in %s: expected %d, found %d",
                name, expectedLength, vector.length));
    }
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
