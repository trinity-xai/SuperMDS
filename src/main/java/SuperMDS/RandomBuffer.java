package SuperMDS;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Provides thread-safe, pre-buffered random values for SIMD operations.
 * @author Sean Phillips
 */
public class RandomBuffer {
    private final double[] uniformBuffer;
    private final double[] gaussianBuffer;
    private int cursorUniform = 0;
    private int cursorGaussian = 0;
    private final int size;

    public RandomBuffer(int size) {
        this.size = size;
        this.uniformBuffer = new double[size];
        this.gaussianBuffer = new double[size];
        refillUniform();
        refillGaussian();
    }

    // ===== UNIFORM =====
    public void refillUniform() {
        ThreadLocalRandom rng = ThreadLocalRandom.current();
        for (int i = 0; i < size; i++) {
            uniformBuffer[i] = rng.nextDouble(); // in [0, 1)
        }
        cursorUniform = 0;
    }

    public void nextUniformBatch(double[] out) {
        int len = out.length;
        if (cursorUniform + len > size) {
            refillUniform();
        }
        System.arraycopy(uniformBuffer, cursorUniform, out, 0, len);
        cursorUniform += len;
    }

    // ===== GAUSSIAN =====
    public void refillGaussian() {
        ThreadLocalRandom rng = ThreadLocalRandom.current();
        for (int i = 0; i < size; i++) {
            gaussianBuffer[i] = rng.nextGaussian(); // mean 0, std 1
        }
        cursorGaussian = 0;
    }

    public void nextGaussianBatch(double[] out) {
        int len = out.length;
        if (cursorGaussian + len > size) {
            refillGaussian();
        }
        System.arraycopy(gaussianBuffer, cursorGaussian, out, 0, len);
        cursorGaussian += len;
    }

    // ===== Single Value Access (Optional) =====
    public double nextUniform() {
        if (cursorUniform >= size) {
            refillUniform();
        }
        return uniformBuffer[cursorUniform++];
    }
    public double next() {
        return nextUniform();
    }

    public double nextGaussian() {
        if (cursorGaussian >= size) {
            refillGaussian();
        }
        return gaussianBuffer[cursorGaussian++];
    }
}

