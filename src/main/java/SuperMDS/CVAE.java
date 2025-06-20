package SuperMDS;

import static SuperMDS.CVAEHelper.add;
import static SuperMDS.CVAEHelper.applyDropout;
import static SuperMDS.CVAEHelper.clipGradient;
import static SuperMDS.CVAEHelper.concat;
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
import static SuperMDS.CVAEHelper.reluGrad;
import static SuperMDS.CVAEHelper.sampleLatent;
import static SuperMDS.CVAEHelper.slice;
import static SuperMDS.CVAEHelper.updateMatrix;
import static SuperMDS.CVAEHelper.updateVector;
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
//        double totalLoss = 0.0;
//        for (int i = 0; i < batchSize; i++) {
//            totalLoss += train(xBatch[i], cBatch[i]);
//        }
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

        double[] z = sampleLatent(mu, safeLogvar);
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
    //</editor-fold>
}
