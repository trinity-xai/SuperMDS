package SuperMDS;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.input.DragEvent;
import javafx.scene.input.Dragboard;
import javafx.scene.input.TransferMode;
import javafx.scene.layout.BorderPane;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.layout.Background;
import javafx.scene.layout.BackgroundFill;
import javafx.scene.layout.CornerRadii;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.DirectoryChooser;
/**
 * @author Sean Phillips
 */
public class SuperMDSApp extends Application {
    
    //make transparent so it doesn't interfere with subnode transparency effects
    Background transBack = new Background(new BackgroundFill(
        Color.TRANSPARENT, CornerRadii.EMPTY, Insets.EMPTY));

    @Override
    public void init() {
//        cocoAnnotationPane = new CocoAnnotationPane();
//        controls = new CocoAnnotationControlBox();
//        basePathTextField = new TextField("");
//        controls.imageBasePathProperty.bind(basePathTextField.textProperty());
    }

    @Override
    public void start(Stage stage) {
//        BorderPane borderPane = new BorderPane(cocoAnnotationPane);
        BorderPane borderPane = new BorderPane();
        borderPane.setBackground(transBack);
        borderPane.addEventHandler(DragEvent.DRAG_OVER, event -> {
            if (ResourceUtils.canDragOver(event)) {
                event.acceptTransferModes(TransferMode.COPY);
            } else {
                event.consume();
            }
        });
        borderPane.addEventHandler(DragEvent.DRAG_DROPPED, event -> {
            Dragboard db = event.getDragboard();
            if (db.hasFiles()) {
                final File file = db.getFiles().get(0);
//                try {
////                    if (CocoAnnotationFile.isCocoAnnotationFile(file)) {
////                        System.out.println("Detected CocoAnnotation File...");
////                        loadCocoFile(file);
////                        controls.populateControls(cocoObject);
//                    }
//                } catch (Exception ex) {
//                    Logger.getLogger(SuperMDSApp.class.getName()).log(Level.SEVERE, null, ex);
//                }
            }
        });
//        ScrollPane scrollPane = new ScrollPane(controls);
//        // hide scrollpane scrollbars
//        scrollPane.setVbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
//        scrollPane.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
//        scrollPane.setPadding(Insets.EMPTY); 
//        scrollPane.setPannable(true);
        
//        controls.heightProperty().addListener(cl-> {
//            scrollPane.setVvalue(scrollPane.getVmax());
//        });
      
        
//        Button browseButton = new Button("Browse");
//        browseButton.setOnAction(e -> {
//            DirectoryChooser dc = new DirectoryChooser();
//            File f = new File(basePathTextField.getText());
//            if(f.isDirectory())
//                dc.setInitialDirectory(f);
//            dc.setTitle("Browse to imagery base path...");
//            File dir = dc.showDialog(null);
//            if(null != dir && dir.isDirectory()) {
//                basePathTextField.setText(dir.getPath());
//            }
//        });
//        HBox basePathHBox = new HBox(10, browseButton, basePathTextField);
//        basePathHBox.setAlignment(Pos.CENTER_LEFT);
//        HBox.setHgrow(basePathTextField, Priority.ALWAYS);
//        basePathTextField.setPrefHeight(40);
//        VBox basePathVBox = new VBox(5, 
//            new Label("Imagery Base Path"), basePathHBox);
//        basePathVBox.setPadding(new Insets(5));
//        borderPane.setTop(basePathVBox);
//        borderPane.setLeft(scrollPane);
        borderPane.getStyleClass().add("trinity-pane");
        
        Scene scene = new Scene(borderPane, 600, 600, Color.BLACK);

        //Make everything pretty
        String CSS = StyleResourceProvider.getResource("styles.css").toExternalForm();
        scene.getStylesheets().add(CSS);

        stage.setTitle("SuperMDS");
        stage.setScene(scene);
        stage.show();
        
        int nPoints = 10000;
        int inputDim = 1000;
        int outputDim = 3;
        System.out.println("Initializing data, labels, weights and params...");
        long startTime = System.nanoTime();
        // Generate synthetic data
        double[][] rawInputData = generateSyntheticData(nPoints, inputDim);
        printTotalTime(startTime);

        startTime = System.nanoTime();
        double[][] distanceMatrix = SuperMDS.ensureSymmetricDistanceMatrix(rawInputData);
        printTotalTime(startTime);

        // Optional: Generate synthetic class labels
        startTime = System.nanoTime();
        int[] labels = generateSyntheticLabels(nPoints, 3); // 3 classes
        printTotalTime(startTime);

        // Optional: generate weights... for no weighting use all 1.0s
        startTime = System.nanoTime();
        double[][] weights = new double[rawInputData.length][rawInputData.length]; 
        for (int i = 0; i < rawInputData.length; i++) {
            Arrays.fill(weights[i], 1.0);
        }
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
        // Run MDS
        System.out.println("Running MDS...");
        startTime = System.nanoTime();

        double[][] embedding = SuperMDS.runMDS(distanceMatrix, params);
        printTotalTime(startTime);
        
        // Print first 5 embedded points
        System.out.println("First 5 output coordinates:");
        for (int i = 0; i < 5; i++) {
            System.out.printf("Point %d: (", i);
            for (int j = 0; j < embedding[i].length; j++) {
                System.out.printf("%.4f", embedding[i][j]);
                if (j < embedding[i].length - 1) System.out.print(", ");
            }
            System.out.println(")");
        }

        System.out.println("Computing Error and Stress Metrics...");
        startTime = System.nanoTime();
        double[][] reconstructed = computeReconstructedDistances(embedding);
        double maxError = maxDistanceError(distanceMatrix, reconstructed);
        double mse = meanSquaredError(distanceMatrix, reconstructed);
        double rawStress = rawStress(distanceMatrix, reconstructed, weights);
        printTotalTime(startTime);
        
        System.out.printf("Results for SMACOF MDS on synthetic data (%d points, %dD → %dD):\n",
                nPoints, inputDim, outputDim);
        System.out.printf("Max error: %.6f\n", maxError);
        System.out.printf("MSE:       %.6f\n", mse);
        System.out.printf("Raw stress: %.6f\n", rawStress);        

        SuperMDSValidator.StressMetrics metrics = 
            SuperMDSValidator.computeStressMetrics(distanceMatrix, reconstructed);
        System.out.println(metrics);  
        
        
        System.out.println("Testing OSE...");
        System.out.println("Generating synthetic test data...");
        startTime = System.nanoTime();
        double[][] testData = generateSyntheticData(100, outputDim); // Normally distributed
        printTotalTime(startTime);

//originalData = raw high-dimensional vectors (e.g. 10000 × 10)
//
//existingEmbedding = low-dimensional output (e.g. 10000 × 2)
//
//newPoint = one raw input vector to embed
//
//Result: embedded coordinates of newPoint using existingEmbedding as reference
        
        // Embed the new points4
        System.out.println("Using OSE to project test data...");
        startTime = System.nanoTime();
        double[] testDataWeights = new double[rawInputData.length];
        Arrays.fill(testDataWeights, 1.0);

        for(int i=0;i<testData.length;i++) {
            double[] distances = distancesToNewPoint(testData[i], rawInputData);
            double[] embeddedNewPoint = SuperMDS.embedPointOSEParallel(
                embedding, distances, testDataWeights, params);
            double oseStress = computeOSEStress(embedding, embeddedNewPoint, distances);
            System.out.printf("Embedding stress for new point: %.6f%n", oseStress);            
        }
        printTotalTime(startTime);
      
    }
    /**
     * Computes Euclidean distances from a new point to each row in the training dataset.
     *
     * @param newPoint     The new point as a 1D array (length = dim).
     * @param trainingData The training dataset as a 2D array (shape: n x dim).
     * @return A 1D array of distances from the new point to each training point.
     */
    public static double[] distancesToNewPoint(double[] newPoint, double[][] trainingData) {
        int n = trainingData.length;
        double[] distances = new double[n];

        for (int i = 0; i < n; i++) {
            distances[i] = SuperMDS.euclideanDistance(newPoint, trainingData[i]);
        }

        return distances;
    }
    public static double[][] computeReconstructedDistances(double[][] embedding) {
        int n = embedding.length;
        double[][] distances = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dist = 0.0;
                for (int d = 0; d < embedding[i].length; d++) {
                    double diff = embedding[i][d] - embedding[j][d];
                    dist += diff * diff;
                }
                dist = Math.sqrt(dist);
                distances[i][j] = dist;
                distances[j][i] = dist; // ensure symmetry
            }
        }

        return distances;
    } 
    public static double computeOSEStress(double[][] embeddings, double[] newPointCoords, double[] distancesToNew) {
        double stress = 0.0;
        for (int i = 0; i < embeddings.length; i++) {
            double dHat = SuperMDS.euclideanDistance(newPointCoords, embeddings[i]);
            double diff = distancesToNew[i] - dHat;
            stress += diff * diff;
        }
        return stress;
    }
    
    public static double maxDistanceError(double[][] D, double[][] reconstructed) {
        int n = D.length;
        double maxError = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double error = Math.abs(D[i][j] - reconstructed[i][j]);
                maxError = Math.max(maxError, error);
            }
        }
        return maxError;
    }

    public static double meanSquaredError(double[][] D, double[][] reconstructed) {
        int n = D.length;
        double sum = 0.0;
        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double diff = D[i][j] - reconstructed[i][j];
                sum += diff * diff;
                count++;
            }
        }
        return sum / count;
    }

    public static double rawStress(double[][] D, double[][] reconstructed, double[][] weights) {
        int n = D.length;
        double stress = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double diff = reconstructed[i][j] - D[i][j];
                double w = (weights != null) ? weights[i][j] : 1.0;
                stress += w * diff * diff;
            }
        }
        return stress;
    }
    
    // Generate random N x D data
    public static double[][] generateSyntheticData(int n, int dim) {
        double[][] data = new double[n][dim];
        Random rand = new Random(42); // deterministic seed

        for (int i = 0; i < n; i++)
            for (int j = 0; j < dim; j++)
                data[i][j] = rand.nextGaussian(); // normal distribution

        return data;
    }

    // Generate synthetic integer class labels (0 to numClasses-1)
    public static int[] generateSyntheticLabels(int n, int numClasses) {
        int[] labels = new int[n];
        Random rand = new Random(42);
        for (int i = 0; i < n; i++)
            labels[i] = rand.nextInt(numClasses);
        return labels;
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
