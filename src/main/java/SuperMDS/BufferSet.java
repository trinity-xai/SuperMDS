package SuperMDS;

import java.util.Arrays;

/**
 *
 * @author Sean Phillips
 */
public class BufferSet {
    public final double[] xc, h1, h2;
    public final double[] mu, logvar, safeLogvar, z, zc;
    public final double[] d1, d2, xRecon;
    public final double[] grad_xRecon, grad_mu, grad_logvar;
    public final double[] dmu_dh2, dlogvar_dh2, dL_dh2, dL_dh1;
    public final double[] dz, dL_dDec1, dL_dDec2, dL_dZC;

    public BufferSet(int inputDim, int condDim, int latentDim, int hiddenDim) {
        int xcLen = inputDim + condDim;
        int zcLen = latentDim + condDim;

        this.xc = new double[xcLen];
        this.h1 = new double[hiddenDim];
        this.h2 = new double[hiddenDim];
        this.mu = new double[latentDim];
        this.logvar = new double[latentDim];
        this.safeLogvar = new double[latentDim];
        this.z = new double[latentDim];
        this.zc = new double[zcLen];
        this.d1 = new double[hiddenDim];
        this.d2 = new double[hiddenDim];
        this.xRecon = new double[inputDim];
        this.grad_xRecon = new double[inputDim];
        this.grad_mu = new double[latentDim];
        this.grad_logvar = new double[latentDim];
        this.dmu_dh2 = new double[hiddenDim];
        this.dlogvar_dh2 = new double[hiddenDim];
        this.dL_dh2 = new double[hiddenDim];
        this.dL_dh1 = new double[hiddenDim];
        this.dz = new double[latentDim];
        this.dL_dDec1 = new double[hiddenDim];
        this.dL_dDec2 = new double[hiddenDim];
        this.dL_dZC = new double[zcLen];
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
        Arrays.fill(grad_mu, 0);
        Arrays.fill(grad_logvar, 0);
        Arrays.fill(dmu_dh2, 0);
        Arrays.fill(dlogvar_dh2, 0);
        Arrays.fill(dL_dh2, 0);
        Arrays.fill(dL_dh1, 0);
        Arrays.fill(dz, 0);
        Arrays.fill(dL_dDec1, 0);
        Arrays.fill(dL_dDec2, 0);
        Arrays.fill(dL_dZC, 0);
    }
}
