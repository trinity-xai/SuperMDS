package SuperMDS;

import SuperMDS.MultilaterationConfig.OptimizerType;
import SuperMDS.SuperMDSAnchors.AnchorSetRecord;
import SuperMDS.SuperMDSAnchors.Strategy;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

import java.util.Arrays;
import java.util.List;
import javafx.collections.FXCollections;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.Spinner;
import javafx.scene.layout.Background;
import javafx.scene.layout.BackgroundFill;
import javafx.scene.layout.CornerRadii;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;
/**
 * @author Sean Phillips
 */
public class SuperMDSApp extends Application {
    
    //make transparent so it doesn't interfere with subnode transparency effects
    Background transBack = new Background(new BackgroundFill(
        Color.TRANSPARENT, CornerRadii.EMPTY, Insets.EMPTY));

    @Override
    public void init() {

    }

    @Override
    public void start(Stage stage) {

        BorderPane borderPane = new BorderPane();
        borderPane.setBackground(transBack);

        Spinner<Integer> numPointsSpinner = new Spinner(100, 10000, 100, 100);
        numPointsSpinner.setEditable(true);
        numPointsSpinner.setPrefWidth(100);

        VBox numPoints = new VBox(5, 
            new Label("Number of Points"), 
            numPointsSpinner
        );
        numPoints.setPrefWidth(200);
        
        Spinner<Integer> inputDimensionsSpinner = new Spinner(3, 1000, 10, 5);
        inputDimensionsSpinner.setEditable(true);
        inputDimensionsSpinner.setPrefWidth(100);

        VBox inputDim = new VBox(5, 
            new Label("Input Dimensions"), 
            inputDimensionsSpinner
        );

        Spinner<Integer> outputDimensionsSpinner = new Spinner(2, 1000, 3, 5);
        outputDimensionsSpinner.setEditable(true);
        outputDimensionsSpinner.setPrefWidth(100);        
        
        VBox outputDim = new VBox(5, 
            new Label("Output Dimensions"), 
            outputDimensionsSpinner
        );
        
        Spinner<Integer> numLandmarksSpinner = new Spinner(2, 1000, 10, 10);
        numLandmarksSpinner.setEditable(true);
        numLandmarksSpinner.setPrefWidth(100); 
        
        VBox numLandmarks = new VBox(5, 
            new Label("Number of Landmarks"), 
            numLandmarksSpinner
        );

        ChoiceBox<Strategy> inverseAnchorChoiceBox = new ChoiceBox<>(
            FXCollections.observableArrayList(SuperMDSAnchors.Strategy.values()));
        inverseAnchorChoiceBox.getSelectionModel().selectFirst();
        VBox inverseAnchors = new VBox(5, 
            new Label("Inverse Anchor Selection Strategy"),
            inverseAnchorChoiceBox
        );
        ChoiceBox<OptimizerType> optimizerChoiceBox = new ChoiceBox<>(
            FXCollections.observableArrayList(MultilaterationConfig.OptimizerType.values()));
        optimizerChoiceBox.getSelectionModel().selectFirst();
        VBox optimizer = new VBox(5, 
            new Label("Multilateration Optimizer"),
            optimizerChoiceBox
        );

        Button testButton = new Button("Test MDS");
        testButton.setOnAction(e -> {
            testMDS(
                numPointsSpinner.getValue(), 
                inputDimensionsSpinner.getValue(),
                outputDimensionsSpinner.getValue(),
                numLandmarksSpinner.getValue(),
                inverseAnchorChoiceBox.getValue(),
                optimizerChoiceBox.getValue()
            );
        });

        VBox centerVBox = new VBox(10, 
            testButton, numPoints, inputDim, outputDim, numLandmarks, 
            inverseAnchors, optimizer
        );
        centerVBox.setAlignment(Pos.CENTER_LEFT);
        borderPane.setCenter(new Pane(centerVBox));
        borderPane.getStyleClass().add("trinity-pane");
        
        Scene scene = new Scene(borderPane, 300, 600, Color.BLACK);

        //Make everything pretty
        String CSS = StyleResourceProvider.getResource("styles.css").toExternalForm();
        scene.getStylesheets().add(CSS);

        stage.setTitle("SuperMDS");
        stage.setScene(scene);
        stage.show();
    }

    public void testMDS(int nPoints, int inputDim, int outputDim, int numberOfLandmarks, 
        Strategy strategy, OptimizerType optimizer) {
        long startTime = System.nanoTime();
        // Generate synthetic data
        double[][] rawInputData = SuperMDSValidator.generateSphereData(nPoints, inputDim, 42);
        printTotalTime(startTime);
        
        // Optional: generate weights... for equal weighting use all 1.0s
        System.out.println("Initializing weights...");
        startTime = System.nanoTime();
        double[][] weights = new double[rawInputData.length][rawInputData.length]; 
        for (int i = 0; i < rawInputData.length; i++) {
            Arrays.fill(weights[i], 1.0);
        }
        printTotalTime(startTime);  

        // Optional: Generate synthetic class labels
        System.out.println("Initializing Labels...");
        startTime = System.nanoTime();
        int[] labels = SuperMDSValidator.generateSyntheticLabels(nPoints, 3); // 3 classes
        printTotalTime(startTime);
        
        // Build params
        SuperMDS.Params params = new SuperMDS.Params();
        params.outputDim = outputDim;
        params.mode = SuperMDS.Mode.PARALLEL;          // Try CLASSICAL, SUPERVISED, LANDMARK, etc.
        params.useSMACOF = true;                     // Enable SMACOF optimization
        params.weights = weights;                   // No weighting
        params.autoSymmetrize = true;             // Auto symmetrization of distance matrix
        params.useKMeansForLandmarks = true;         // If LANDMARK mode is selected
        params.classLabels = labels;                 // Only used by SUPERVISED mode
        params.numLandmarks = 20;                    // Used if LANDMARK mode is active
        params.useParallel = false;               // Toggle parallelized SMACOF
        params.useStressSampling = true;         // allows SMACOF to drastically reduce iterations
        params.stressSampleCount = 1000; //number of stress samples per SMACOF interation
        
        
        System.out.println("ensuring Symmetric Distance Matrix and normalizing...");
        startTime = System.nanoTime();
        double [][] classicalSquaredDistances = SuperMDSHelper.computeSquaredEuclideanDistanceMatrix(rawInputData);
        double[][] symmetricDistanceMatrix = SuperMDS.ensureSymmetricDistanceMatrix(rawInputData);
        //normalize
        double[][] normalizedDistanceMatrix = SuperMDSHelper.normalizeDistancesParallel(symmetricDistanceMatrix);
        printTotalTime(startTime);        

        System.out.println("Number of Points: " + nPoints);
        System.out.println("Input Dimensions: " + inputDim);
        System.out.println("Output Dimensions: " + outputDim);
        System.out.println("Number of Landmarks: " + numberOfLandmarks);
        System.out.println("Initializing data...");

        System.out.println("Testing Classical MDS...");
        startTime = System.nanoTime();
        double[][] classicalEmbeddings = SuperMDS.classicalMDS(classicalSquaredDistances, outputDim);
        printTotalTime(startTime);        

        System.out.println("Computing Error and Stress Metrics for Classical MDS...");
        SuperMDSValidator.computeStressMetricsClassic(rawInputData, classicalEmbeddings);        

        
        System.out.println("Testing Landmark MDS...");
        startTime = System.nanoTime();
        double[][] landmarkEmbeddings = SuperMDS.landmarkMDS(
            rawInputData, outputDim, numberOfLandmarks, false, 42);
        printTotalTime(startTime);        
        System.out.printf("Results for Landmark MDS on synthetic data (%d points, %dD → %dD):\n",
                nPoints, inputDim, outputDim);
        SuperMDSValidator.computeStressMetricsClassic(rawInputData, landmarkEmbeddings);

        System.out.println("Testing Approximate Landmark MDS...");
        startTime = System.nanoTime();
        double[][] approximateLandmarkEmbeddings = SuperMDS.approximateMDSViaLandmarks(
            rawInputData, outputDim, numberOfLandmarks, false, 42);
        printTotalTime(startTime);        
        System.out.printf("Results for Approximate Landmark MDS on synthetic data (%d points, %dD → %dD):\n",
                nPoints, inputDim, outputDim);
        SuperMDSValidator.computeStressMetricsClassic(rawInputData, approximateLandmarkEmbeddings);

        // Run MDS
        System.out.println("Running SMACOF MDS...");
        startTime = System.nanoTime();

        double[][] embeddings = SuperMDS.runMDS(normalizedDistanceMatrix, params);
        printTotalTime(startTime);

        System.out.println("Computing Error and Stress Metrics...");
        startTime = System.nanoTime();
        double [][] reconstructed = SuperMDSHelper.computeReconstructedDistances(embeddings);
        double maxError = SuperMDSValidator.maxDistanceError(normalizedDistanceMatrix, reconstructed);
        double mse = SuperMDSValidator.meanSquaredError(normalizedDistanceMatrix, reconstructed);
        double rawStress = SuperMDSValidator.rawStress(normalizedDistanceMatrix, reconstructed, weights);
        printTotalTime(startTime);
        
        System.out.printf("Results for SMACOF MDS on synthetic data (%d points, %dD → %dD):\n",
                nPoints, inputDim, outputDim);
        System.out.printf("Max error: %.6f\n", maxError);
        System.out.printf("MSE:       %.6f\n", mse);
        System.out.printf("Raw stress: %.6f\n", rawStress);        

        SuperMDSValidator.StressMetrics smacofStressMetrics = 
            SuperMDSValidator.computeStressMetrics(normalizedDistanceMatrix, reconstructed);
        System.out.println(smacofStressMetrics);  
        
        
//        System.out.println("Testing OSE...");
//        System.out.println("Generating synthetic test data...");
//        startTime = System.nanoTime();
//        double[][] testData = SuperMDSValidator.generateSyntheticData(5, inputDim); // Normally distributed
//        printTotalTime(startTime);
//
//        // Embed the new points4
//        System.out.println("Using OSE to project test data...");
//        startTime = System.nanoTime();
//        double[] testDataWeights = new double[rawInputData.length];
//        Arrays.fill(testDataWeights, 1.0);
//
//        for(int i=0;i<testData.length;i++) {
//            double[] distances = SuperMDSHelper.distancesToNewPoint(testData[i], rawInputData);
//            double[] normalizedDistances = SuperMDSHelper.normalizeDistances(distances);
//            double[] embeddedNewPoint = SuperMDS.embedPointOSEParallel(
//                embeddings, normalizedDistances, testDataWeights, params);
//            double oseStress = SuperMDSValidator.computeOSEStress(embeddings, embeddedNewPoint, normalizedDistances);
//            System.out.printf("Embedding stress for new point: %.6f%n", oseStress);            
//            double oseGoodnessOfFit = SuperMDSValidator.computeOSEGoodnessOfFit(
//                embeddings, embeddedNewPoint, normalizedDistances);
//            System.out.printf("Goodness-of-Fit for new point: %.6f <-------------------- %n", oseGoodnessOfFit);            
//        }
//        printTotalTime(startTime);        
        
        AnchorSetRecord anchorSet = SuperMDSAnchors.selectAnchors(rawInputData, 
            numberOfLandmarks, strategy, 42);
        double [][] landmarksLowD = SuperMDSAnchors.extractByIndices(embeddings, anchorSet.indices());
        
        double [][] inverseMappedData = new double[embeddings.length][embeddings[0].length];

        MultilaterationConfig config = new MultilaterationConfig();
       
        config.regularizationLambda = 0.01;
        config.maxIterations = 10000;
        config.maxEvaluations = 10000;
        config.optimizer = optimizer;
        for(int i=0;i<rawInputData.length; i++) {
            double[] reconstructedHighD = SuperMDSInverter.invertViaMultilateration(
                anchorSet.anchors(),
                landmarksLowD,
                embeddings[i],
                config
            );
            inverseMappedData[i] = Arrays.copyOf(reconstructedHighD, reconstructedHighD.length);
        }

        SuperMDSValidator.ValidationResults results = SuperMDSValidator.validateInversion(
            rawInputData, inverseMappedData, anchorSet.anchors()
        );
        System.out.println(results.toString());
        
        System.out.println("Sanity Check for Multilateration...");
        double[] x_orig = anchorSet.anchors()[0];
        double[] x_emb = embeddings[0];
        double[] x_recovered = SuperMDSInverter.invertViaMultilateration(
            anchorSet.anchors(), embeddings, x_emb, config);
        System.out.println("Error: " + SuperMDSHelper.euclideanDistance(x_orig, x_recovered));
        
        System.out.println("NEW AND IMPROVED Sanity Checks for Inversions...");
        //identity test: Set the high-D and low-D space equal and skip MDS. 
        double[][] sanityAnchors = SuperMDSValidator.generateSphereData(nPoints, inputDim, 42);
        double[][] sanityEmbeddings = sanityAnchors;  // Identity mapping
        int[] anchorIndices = new int[] { 0, 2, 4, 6, 8, 10 };       
        System.out.println("Multilateration Inversion...");
        SuperMDSValidator.runMultilaterationSanityCheck(
            sanityAnchors, sanityEmbeddings, config, List.of(0, 1, 2)
        );       
        System.out.println("Pseudo Inversion...");
        SuperMDSValidator.runPseudoinverseInversionSanityCheck(sanityAnchors, sanityEmbeddings, anchorIndices);  // Identity        
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

    public static void printTotalTime(long startTime) {
        System.out.println(totalTimeString(startTime));
    }   
    public static void main(String[] args) {
        launch(args);
    }
}
