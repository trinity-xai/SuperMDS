package com.github.trinity.supermds;

/**
 * @author Sean Phillips
 */
public class CVAEBufferSet {
    // Forward pass
    public final double[] xc;       // input + condition
    public final double[] h1;       // encoder hidden layer 1
    public final double[] h2;       // encoder hidden layer 2
    public final double[] mu;       // mean vector (encoder output)
    public final double[] logvar;   // log variance vector
    public final double[] safeLogvar; // clipped logvar for numerical stability
    public final double[] z;        // sampled latent vector
    public final double[] zc;       // latent + condition (decoder input)
    public final double[] d1;       // decoder hidden layer 1
    public final double[] d2;       // decoder hidden layer 2
    public final double[] xRecon;   // reconstruction output

    // Gradients from loss
    public final double[] dL_dxRecon;

    // Decoder backward
    public final double[] dL_dDecOut; // grad wrt decoder output
    public final double[] dL_dDec2;
    public final double[] dL_dDec1;
    public final double[] dL_dZC;
    public final double[] dz;

    // KL backward
    public final double[] dL_dmu;
    public final double[] dL_dlogvar;

    // Encoder backward
    public final double[] dmu_dh2;
    public final double[] dlogvar_dh2;
    public final double[] dL_dh2;
    public final double[] dL_dh1;

    /**
     * Allocate all buffers needed for a single-threaded forward/backward pass.
     * These are reused during parallel training to avoid garbage collection.
     */
    public CVAEBufferSet(int inputDim, int conditionDim, int latentDim, int hiddenDim) {
        int encInputDim = inputDim + conditionDim;
        int decInputDim = latentDim + conditionDim;

        xc = new double[encInputDim];
        h1 = new double[hiddenDim];
        h2 = new double[hiddenDim];
        mu = new double[latentDim];
        logvar = new double[latentDim];
        safeLogvar = new double[latentDim];
        z = new double[latentDim];
        zc = new double[decInputDim];
        d1 = new double[hiddenDim];
        d2 = new double[hiddenDim];
        xRecon = new double[inputDim];

        dL_dxRecon = new double[inputDim];

        dL_dDecOut = new double[hiddenDim];
        dL_dDec2 = new double[hiddenDim];
        dL_dDec1 = new double[hiddenDim];
        dL_dZC = new double[decInputDim];
        dz = new double[latentDim];

        dL_dmu = new double[latentDim];
        dL_dlogvar = new double[latentDim];

        dmu_dh2 = new double[hiddenDim];
        dlogvar_dh2 = new double[hiddenDim];
        dL_dh2 = new double[hiddenDim];
        dL_dh1 = new double[hiddenDim];
    }
}
