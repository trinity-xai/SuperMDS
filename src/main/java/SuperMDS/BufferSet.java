package SuperMDS;

import java.util.Arrays;

/**
 *
 * @author Sean Phillips
 */
public class BufferSet {
    // Dimensions
    private final int inputDim, conditionDim, latentDim, hiddenDim;

    // --- Forward/Backward Buffers (same as before) ---
    public final double[] xc, h1, h2, mu, logvar, safeLogvar, z, zc;
    public final double[] d1, d2, xRecon;
    public final double[] grad_xRecon, dz, grad_mu, grad_logvar;
    public final double[] dmu_dh2, dlogvar_dh2, dL_dh2, dL_dh1, dL_dZC;
    public final double[] dL_dDec1, dL_dDec2;

    // --- Gradient Accumulators ---
    public final double[][] grad_W_enc1, grad_W_enc2, grad_W_mu, grad_W_logvar;
    public final double[]   grad_b_enc1, grad_b_enc2, grad_b_mu, grad_b_logvar;

    public final double[][] grad_W_dec1, grad_W_dec2, grad_W_decOut;
    public final double[]   grad_b_dec1, grad_b_dec2, grad_b_decOut;
    
    public BufferSet(int inputDim, int conditionDim, int latentDim, int hiddenDim) {
        this.inputDim = inputDim;
        this.conditionDim = conditionDim;
        this.latentDim = latentDim;
        this.hiddenDim = hiddenDim;

        int encInputDim = inputDim + conditionDim;
        int decInputDim = latentDim + conditionDim;

        // --- Forward/Backward ---
        this.xc = new double[encInputDim];
        this.h1 = new double[hiddenDim];
        this.h2 = new double[hiddenDim];
        this.mu = new double[latentDim];
        this.logvar = new double[latentDim];
        this.safeLogvar = new double[latentDim];
        this.z = new double[latentDim];
        this.zc = new double[decInputDim];

        this.d1 = new double[hiddenDim];
        this.d2 = new double[hiddenDim];
        this.xRecon = new double[inputDim];

        this.grad_xRecon = new double[inputDim];
        this.dz = new double[latentDim];
        this.grad_mu = new double[latentDim];
        this.grad_logvar = new double[latentDim];

        this.dmu_dh2 = new double[hiddenDim];
        this.dlogvar_dh2 = new double[hiddenDim];
        this.dL_dh2 = new double[hiddenDim];
        this.dL_dh1 = new double[hiddenDim];
        this.dL_dZC = new double[latentDim + conditionDim];

        this.dL_dDec1 = new double[hiddenDim];
        this.dL_dDec2 = new double[hiddenDim];

        // --- Gradient Accumulators ---
        this.grad_W_enc1 = new double[encInputDim][hiddenDim];
        this.grad_b_enc1 = new double[hiddenDim];

        this.grad_W_enc2 = new double[hiddenDim][hiddenDim];
        this.grad_b_enc2 = new double[hiddenDim];

        this.grad_W_mu = new double[hiddenDim][latentDim];
        this.grad_b_mu = new double[latentDim];

        this.grad_W_logvar = new double[hiddenDim][latentDim];
        this.grad_b_logvar = new double[latentDim];

        this.grad_W_dec1 = new double[decInputDim][hiddenDim];
        this.grad_b_dec1 = new double[hiddenDim];

        this.grad_W_dec2 = new double[hiddenDim][hiddenDim];
        this.grad_b_dec2 = new double[hiddenDim];

        this.grad_W_decOut = new double[hiddenDim][inputDim];
        this.grad_b_decOut = new double[inputDim];
    }
    public void resetForwardBuffers() {
        Arrays.fill(xc, 0); Arrays.fill(h1, 0); Arrays.fill(h2, 0); 
        Arrays.fill(mu, 0); Arrays.fill(logvar, 0); Arrays.fill(safeLogvar, 0);
        Arrays.fill(z, 0); Arrays.fill(zc, 0);
        Arrays.fill(d1, 0); Arrays.fill(d2, 0); Arrays.fill(xRecon, 0);
        Arrays.fill(grad_xRecon, 0); Arrays.fill(dz, 0); 
        Arrays.fill(grad_mu, 0); Arrays.fill(grad_logvar, 0);
        Arrays.fill(dmu_dh2, 0); Arrays.fill(dlogvar_dh2, 0); 
        Arrays.fill(dL_dh2, 0); Arrays.fill(dL_dh1, 0); 
        Arrays.fill(dL_dZC, 0); Arrays.fill(dL_dDec1, 0); Arrays.fill(dL_dDec2, 0);
    }

    public void resetGradients() {
        zeroMatrix(grad_W_enc1); zeroVector(grad_b_enc1);
        zeroMatrix(grad_W_enc2); zeroVector(grad_b_enc2);
        zeroMatrix(grad_W_mu); zeroVector(grad_b_mu);
        zeroMatrix(grad_W_logvar); zeroVector(grad_b_logvar);
        zeroMatrix(grad_W_dec1); zeroVector(grad_b_dec1);
        zeroMatrix(grad_W_dec2); zeroVector(grad_b_dec2);
        zeroMatrix(grad_W_decOut); zeroVector(grad_b_decOut);
    }
    public void reset() {
        Arrays.fill(xc, 0);
        Arrays.fill(h1, 0);
        Arrays.fill(h2, 0);
        Arrays.fill(mu, 0);
        Arrays.fill(logvar, 0);
        Arrays.fill(safeLogvar, 0);
        Arrays.fill(z, 0);
        Arrays.fill(zc, 0);
        Arrays.fill(d1, 0);
        Arrays.fill(d2, 0);
        Arrays.fill(xRecon, 0);
        Arrays.fill(grad_xRecon, 0);
        Arrays.fill(dz, 0);
        Arrays.fill(grad_mu, 0);
        Arrays.fill(grad_logvar, 0);
        Arrays.fill(dmu_dh2, 0);
        Arrays.fill(dlogvar_dh2, 0);
        Arrays.fill(dL_dh2, 0);
        Arrays.fill(dL_dh1, 0);
        Arrays.fill(dL_dZC, 0);
        Arrays.fill(dL_dDec1, 0);
        Arrays.fill(dL_dDec2, 0);

        zeroMatrix(grad_W_enc1);
        zeroVector(grad_b_enc1);
        zeroMatrix(grad_W_enc2);
        zeroVector(grad_b_enc2);
        zeroMatrix(grad_W_mu);
        zeroVector(grad_b_mu);
        zeroMatrix(grad_W_logvar);
        zeroVector(grad_b_logvar);
        zeroMatrix(grad_W_dec1);
        zeroVector(grad_b_dec1);
        zeroMatrix(grad_W_dec2);
        zeroVector(grad_b_dec2);
        zeroMatrix(grad_W_decOut);
        zeroVector(grad_b_decOut);
    }

    private void zeroMatrix(double[][] mat) {
        for (double[] row : mat) Arrays.fill(row, 0.0);
    }

    private void zeroVector(double[] vec) {
        Arrays.fill(vec, 0.0);
    }
}
 