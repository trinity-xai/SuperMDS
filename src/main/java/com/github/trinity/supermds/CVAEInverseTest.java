package com.github.trinity.supermds;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

import static com.github.trinity.supermds.CVAEHelper.*;

/**
 * @author Sean Phillips
 */

public class CVAEInverseTest {
    private static final Logger LOG = LoggerFactory.getLogger(CVAEInverseTest.class);

    public static void main(String[] args) {
        // Example synthetic test case
        int numPoints = 1000;
        int inputDim = 10;      // Original high-dimensional space
        int embeddingDim = 3;    // From SMACOF MDS
        int latentDim = 16;
        int hiddenDim = 64;
        int batchSize = 128;
        int epochs = 2000;

        // Generate dummy original data (e.g., MDS input)
        double[][] originalData = generateRandomData(numPoints, inputDim);
        // Optional: generate weights... for equal weighting use all 1.0s
        LOG.info("Initializing weights...");
        long startTime = System.nanoTime();
        double[][] weights = new double[originalData.length][originalData.length];
        for (int i = 0; i < originalData.length; i++) {
            Arrays.fill(weights[i], 1.0);
        }
        printTotalTime(startTime);
        // Build params
        SuperMDS.Params params = new SuperMDS.Params();
        params.outputDim = embeddingDim;
        params.mode = SuperMDS.Mode.PARALLEL;          // Try CLASSICAL, SUPERVISED, LANDMARK, etc.
        params.useSMACOF = true;                     // Enable SMACOF optimization
        params.weights = weights;                   // No weighting
        params.autoSymmetrize = true;             // Auto symmetrization of distance matrix
        params.useKMeansForLandmarks = true;         // If LANDMARK mode is selected
        params.classLabels = null;                 // Only used by SUPERVISED mode
        params.numLandmarks = 20;                    // Used if LANDMARK mode is active
        params.useParallel = false;               // Toggle parallelized SMACOF
        params.useStressSampling = true;         // allows SMACOF to drastically reduce iterations
        params.stressSampleCount = 1000; //number of stress samples per SMACOF interation

        // Run com.github.trinity.supermds/SMACOF to get embeddings
        LOG.info("Running SMACOF MDS...");
        startTime = System.nanoTime();
        double[][] symmetricDistanceMatrix = SuperMDS.ensureSymmetricDistanceMatrix(originalData);
        //normalize
        double[][] normalizedDistanceMatrix = SuperMDSHelper.normalizeDistancesParallel(symmetricDistanceMatrix);
        double[][] mdsEmbedding = SuperMDS.runMDS(normalizedDistanceMatrix, params);
        printTotalTime(startTime);

        //Sanity check on CVAE
        Normalizer normalizer = new Normalizer(originalData, Normalizer.Type.Z_SCORE);
        double[][] normalizedData = normalizer.normalizeAll(originalData);

        // Sanity check: set conditional to first 3 dimensions of original input
        Normalizer embeddingNormalizer = new Normalizer(mdsEmbedding, Normalizer.Type.Z_SCORE);
        double[][] normalizedEmbedding = embeddingNormalizer.normalizeAll(mdsEmbedding);
        double[][] conditions = new double[numPoints][embeddingDim];
        // full 3D embedding as condition
        System.arraycopy(normalizedEmbedding, 0, conditions, 0, numPoints);

        for (int outerLoop = 0; outerLoop < 10; outerLoop++) {
            // Initialize CVAE
            CVAE cvae = new CVAE(inputDim, embeddingDim, latentDim, hiddenDim);
            cvae.setDebug(false);
            cvae.setUseDropout(false);
            cvae.setIsTraining(true);
            // Train the CVAE
            LOG.info("Training CVAE...");
            startTime = System.nanoTime();

            for (int epoch = 0; epoch < epochs; epoch++) {
                double totalLoss = 0.0;
                int numBatches = numPoints / batchSize;

                // Shuffle the dataset at the beginning of each epoch
                int[] indices = shuffledIndices(numPoints, cvae.threadLocalRandom.get());

                for (int b = 0; b < numBatches; b++) {
                    double[][] xBatch = new double[batchSize][inputDim];
                    double[][] cBatch = new double[batchSize][embeddingDim];

                    for (int i = 0; i < batchSize; i++) {
                        int idx = indices[b * batchSize + i];
                        xBatch[i] = normalizedData[idx];
                        cBatch[i] = conditions[idx];
                    }
                    totalLoss += cvae.trainBatch(xBatch, cBatch);
                }
            }
            cvae.setIsTraining(false);

            printTotalTime(startTime);
            // Test inverseTransform
            double totalReconError = 0.0;
            double totalMeanVar = 0.0;
            for (int i = 0; i < numPoints; i++) {
                double[] recon = cvae.inverseTransform(conditions[i]);
                //            double[] recon = cvae.inverseTransform(mdsEmbedding[i]);
                //            double mse = mseLoss(originalData[i], recon);
                double mse = mseLoss(normalizedData[i], recon);
                totalReconError += mse;

                double[] var = cvae.confidenceEstimate(conditions[i], 50);
                totalMeanVar += Arrays.stream(var).average().orElse(Double.NaN);
//                LOG.info("Condition {}: Mean variance = {}", i, String.format("%.6f", meanVar));
            }

            double avgReconError = totalReconError / numPoints;
            double avgMeanVariance = totalMeanVar / numPoints;
            LOG.info("Average Reconstruction MSE: {}", String.format("%.6f", avgReconError));
            LOG.info("Average Mean Variance: {}", String.format("%.6f", avgMeanVariance));
        }
    }

    public static void printTotalTime(long startTime) {
        LOG.info(totalTimeString(startTime));
    }

    public static String totalTimeString(long startTime) {
        long estimatedTime = System.nanoTime() - startTime;
        long totalNanos = estimatedTime;
        long s = totalNanos / 1000000000;
        totalNanos -= s * 1000000000;
        long ms = totalNanos / 1000000;
        totalNanos -= ms * 1000000;

        long us = totalNanos / 1000;
        totalNanos -= us * 1000;
        return "Total elapsed time: " + s + ":s:" + ms + ":ms:" + us + ":us:" + totalNanos + ":ns";
    }
}
