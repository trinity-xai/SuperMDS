package SuperMDS;

import static SuperMDS.CVAEHelper.generateRandomData;
import static SuperMDS.CVAEHelper.mseLoss;
import static SuperMDS.SuperMDSApp.printTotalTime;
import java.util.Arrays;

/**
 *
 * @author Sean Phillips
 */

public class CVAEInverseTest {

    public static void main(String[] args) {
         
        // Example synthetic test case
        int numPoints = 1000;
        int inputDim = 10;      // Original high-dimensional space
        int embeddingDim = 3;    // From SMACOF MDS
        int latentDim = 16;
        int hiddenDim = 64;
        
        // Generate dummy original data (e.g., MDS input)
        double[][] originalData = generateRandomData(numPoints, inputDim);
        // Optional: generate weights... for equal weighting use all 1.0s
        System.out.println("Initializing weights...");
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
        
        // Run SuperMDS/SMACOF to get embeddings
        System.out.println("Running SMACOF MDS...");
        startTime = System.nanoTime();
        double[][] symmetricDistanceMatrix = SuperMDS.ensureSymmetricDistanceMatrix(originalData);
        //normalize
        double[][] normalizedDistanceMatrix = SuperMDSHelper.normalizeDistancesParallel(symmetricDistanceMatrix);
        double[][] mdsEmbedding = SuperMDS.runMDS(normalizedDistanceMatrix, params);
        printTotalTime(startTime);

        //Sanity check on CVAE
        Normalizer normalizer = new Normalizer(originalData, Normalizer.Type.Z_SCORE);
        double[][] normalizedData = normalizer.normalizeAll(originalData);
        // Initialize CVAE
        CVAE cvae = new CVAE(inputDim, embeddingDim, latentDim, hiddenDim);
        // Sanity check: set conditional to first 3 dimensions of original input
        double[][] conditions = new double[numPoints][3];
        for (int i = 0; i < numPoints; i++) {
//            System.arraycopy(originalData[i], 0, conditions[i], 0, 3);
            System.arraycopy(normalizedData[i], 0, conditions[i], 0, 3);
        }
        // Train the CVAE
        System.out.println("Training CVAE...");
        startTime = System.nanoTime();
        int epochs = 2000;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            for (int i = 0; i < numPoints; i++) {
                //totalLoss += cvae.train(originalData[i], mdsEmbedding[i]);
                totalLoss += cvae.train(normalizedData[i], conditions[i]);
            }
//            if (epoch % 50 == 0 || epoch == epochs - 1) {
//                System.out.printf("Epoch %d, Avg Loss: %.6f%n", epoch, totalLoss / numPoints);
//            }
        }
        printTotalTime(startTime);
        // Test inverseTransform
        System.out.println("\nEvaluating reconstruction error...");
        double totalReconError = 0.0;
        for (int i = 0; i < numPoints; i++) {
//            double[] recon = cvae.inverseTransform(mdsEmbedding[i]);
            double[] recon = cvae.inverseTransform(conditions[i]);
//            double mse = mseLoss(originalData[i], recon);
            double mse = mseLoss(normalizedData[i], recon);
            totalReconError += mse;
            System.out.printf("Point %3d - MSE: %.6f%n", i, mse);
        }

        double avgReconError = totalReconError / numPoints;
        System.out.printf("\nAverage Reconstruction MSE: %.6f%n", avgReconError);
    }
}