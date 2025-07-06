package com.github.trinity.supermds;

import java.util.Arrays;

/**
 * @author Sean Phillips
 */
public class GradientBuffer {
    public final double[][] grad_W_enc1, grad_W_enc2, grad_W_mu, grad_W_logvar;
    public final double[] grad_b_enc1, grad_b_enc2, grad_b_mu, grad_b_logvar;
    public final double[][] grad_W_dec1, grad_W_dec2, grad_W_decOut;
    public final double[] grad_b_dec1, grad_b_dec2, grad_b_decOut;

    public GradientBuffer(int inputDim, int condDim, int latentDim, int hiddenDim) {
        int encInput = inputDim + condDim;
        int decInput = latentDim + condDim;

        grad_W_enc1 = new double[encInput][hiddenDim];
        grad_b_enc1 = new double[hiddenDim];
        grad_W_enc2 = new double[hiddenDim][hiddenDim];
        grad_b_enc2 = new double[hiddenDim];
        grad_W_mu = new double[hiddenDim][latentDim];
        grad_b_mu = new double[latentDim];
        grad_W_logvar = new double[hiddenDim][latentDim];
        grad_b_logvar = new double[latentDim];

        grad_W_dec1 = new double[decInput][hiddenDim];
        grad_b_dec1 = new double[hiddenDim];
        grad_W_dec2 = new double[hiddenDim][hiddenDim];
        grad_b_dec2 = new double[hiddenDim];
        grad_W_decOut = new double[hiddenDim][inputDim];
        grad_b_decOut = new double[inputDim];
    }

    public void clear() {
        zero(grad_W_enc1);
        zero(grad_b_enc1);
        zero(grad_W_enc2);
        zero(grad_b_enc2);
        zero(grad_W_mu);
        zero(grad_b_mu);
        zero(grad_W_logvar);
        zero(grad_b_logvar);
        zero(grad_W_dec1);
        zero(grad_b_dec1);
        zero(grad_W_dec2);
        zero(grad_b_dec2);
        zero(grad_W_decOut);
        zero(grad_b_decOut);
    }

    private void zero(double[][] m) {
        for (double[] row : m) Arrays.fill(row, 0.0);
    }

    private void zero(double[] v) {
        Arrays.fill(v, 0.0);
    }

    public void scale(double factor) {
        scale(grad_W_enc1, factor);
        scale(grad_b_enc1, factor);
        scale(grad_W_enc2, factor);
        scale(grad_b_enc2, factor);
        scale(grad_W_mu, factor);
        scale(grad_b_mu, factor);
        scale(grad_W_logvar, factor);
        scale(grad_b_logvar, factor);
        scale(grad_W_dec1, factor);
        scale(grad_b_dec1, factor);
        scale(grad_W_dec2, factor);
        scale(grad_b_dec2, factor);
        scale(grad_W_decOut, factor);
        scale(grad_b_decOut, factor);
    }

    private void scale(double[][] m, double factor) {
        for (int i = 0; i < m.length; i++)
            for (int j = 0; j < m[i].length; j++)
                m[i][j] *= factor;
    }

    private void scale(double[] v, double factor) {
        for (int i = 0; i < v.length; i++)
            v[i] *= factor;
    }

    public void accumulate(GradientBuffer other) {
        add(grad_W_enc1, other.grad_W_enc1);
        add(grad_b_enc1, other.grad_b_enc1);
        add(grad_W_enc2, other.grad_W_enc2);
        add(grad_b_enc2, other.grad_b_enc2);
        add(grad_W_mu, other.grad_W_mu);
        add(grad_b_mu, other.grad_b_mu);
        add(grad_W_logvar, other.grad_W_logvar);
        add(grad_b_logvar, other.grad_b_logvar);
        add(grad_W_dec1, other.grad_W_dec1);
        add(grad_b_dec1, other.grad_b_dec1);
        add(grad_W_dec2, other.grad_W_dec2);
        add(grad_b_dec2, other.grad_b_dec2);
        add(grad_W_decOut, other.grad_W_decOut);
        add(grad_b_decOut, other.grad_b_decOut);
    }

    private void add(double[][] a, double[][] b) {
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < a[i].length; j++)
                a[i][j] += b[i][j];
    }

    private void add(double[] a, double[] b) {
        for (int i = 0; i < a.length; i++)
            a[i] += b[i];
    }
}
