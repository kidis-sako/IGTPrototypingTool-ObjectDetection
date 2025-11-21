package controller;

import algorithm.UltrasoundObjectDetector;
import algorithm.ImageDataManager;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.logging.Logger;

import nu.pattern.OpenCV;

/**
 * Controller for demonstrating object detection algorithms on ultrasound images
 *
 * @author Moritz Koehn, Kidis Sako
 */
public class ObjectDetectionController {
    
    @FXML
    private ImageView originalImageView;
    
    @FXML
    private ImageView detectionImageView;
    
    @FXML
    private ComboBox<String> detectionMethodCombo;
    
    @FXML
    private Button detectButton;
    
    @FXML
    private TextArea resultsTextArea;
    
    @FXML
    private Slider cannyThreshold1Slider;
    
    @FXML
    private Slider cannyThreshold2Slider;
    
    @FXML
    private Slider houghThresholdSlider;
    
    @FXML
    private Slider minRadiusSlider;
    
    @FXML
    private Slider maxRadiusSlider;
    
    @FXML
    private Label cannyThreshold1Label;
    
    @FXML
    private Label cannyThreshold2Label;
    
    @FXML
    private Label houghThresholdLabel;
    
    @FXML
    private Label minRadiusLabel;
    
    @FXML
    private Label maxRadiusLabel;
    
    @FXML
    private Slider peakHeightRatioSlider;
    
    @FXML
    private Label peakHeightRatioLabel;
    
    @FXML
    private CheckBox visualizeCheckBox;
    
    @FXML
    private TitledPane parametersPanel;
    
    @FXML
    private Button autoThresholdsButton;
    
    @FXML
    private Label cannyLabel;
    
    @FXML
    private Label houghLabel;
    
    @FXML
    private Label circleLabel;
    
    @FXML
    private Label interfaceLabel;
    
    private UltrasoundObjectDetector detector;
    private ImageDataManager imageDataManager;
    private Mat currentImage;
    private String defaultImagePath = "/circle.jpg";  // Path to default image resource
    private Logger logger = Logger.getLogger(this.getClass().getName());
    
    /**
     * Initialize the controller
     */
    @FXML
    public void initialize() {
        // Load OpenCV native libraries
        OpenCV.loadLocally();
        
        detector = new UltrasoundObjectDetector();
        
        // Populate detection methods
        detectionMethodCombo.getItems().addAll(
            "Hough Line Detection",
            "RANSAC Line Detection",
            "Hough Circle Detection",
            "Blob Sphere Detection",
            "Horizontal Interfaces Detection"
        );
        detectionMethodCombo.setValue("RANSAC Line Detection");
        
        // Set up parameter sliders
        setupParameterListeners();
        
        // Set up detection method change listener to enable/disable relevant parameters
        setupDetectionMethodListener();
        
        // Set initial threshold values (optimized for ultrasound)
        if (cannyThreshold1Slider != null && cannyThreshold2Slider != null) {
            detector.setCannyThresholds(
                cannyThreshold1Slider.getValue(), 
                cannyThreshold2Slider.getValue()
            );
        }
        
        // Initially visualize is checked
        visualizeCheckBox.setSelected(true);
        
        logger.info("Object Detection Controller initialized (ultrasound-optimized defaults: 30/90)");
    }
    
    /**
     * Set up listeners for parameter sliders
     */
    private void setupParameterListeners() {
        if (cannyThreshold1Slider != null) {
            cannyThreshold1Slider.valueProperty().addListener((obs, oldVal, newVal) -> {
                cannyThreshold1Label.setText(String.format("%.0f", newVal.doubleValue()));
                detector.setCannyThresholds(newVal.doubleValue(), cannyThreshold2Slider.getValue());
            });
        }
        
        if (cannyThreshold2Slider != null) {
            cannyThreshold2Slider.valueProperty().addListener((obs, oldVal, newVal) -> {
                cannyThreshold2Label.setText(String.format("%.0f", newVal.doubleValue()));
                detector.setCannyThresholds(cannyThreshold1Slider.getValue(), newVal.doubleValue());
            });
        }
        
        if (houghThresholdSlider != null) {
            houghThresholdSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
                houghThresholdLabel.setText(String.format("%.0f", newVal.doubleValue()));
                detector.setHoughThreshold(newVal.intValue());
            });
        }
        
        if (minRadiusSlider != null) {
            minRadiusSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
                minRadiusLabel.setText(String.format("%.0f", newVal.doubleValue()));
            });
        }
        
        if (maxRadiusSlider != null) {
            maxRadiusSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
                maxRadiusLabel.setText(String.format("%.0f", newVal.doubleValue()));
            });
        }
        
        if (peakHeightRatioSlider != null) {
            peakHeightRatioSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
                peakHeightRatioLabel.setText(String.format("%.2f", newVal.doubleValue()));
                detector.setMinPeakHeightRatio(newVal.doubleValue());
            });
        }
    }
    
    /**
     * Set up listener for detection method selection to enable/disable relevant parameters
     */
    private void setupDetectionMethodListener() {
        if (detectionMethodCombo != null) {
            detectionMethodCombo.valueProperty().addListener((obs, oldVal, newVal) -> {
                updateParameterAvailability(newVal);
            });
            
            // Initialize parameter availability for the default selection
            updateParameterAvailability(detectionMethodCombo.getValue());
        }
    }
    
    /**
     * Enable/disable parameters based on selected detection method
     * 
     * Parameter usage by algorithm:
     * - Hough Line Detection: Canny thresholds + Hough threshold
     * - RANSAC Line Detection: Canny thresholds only
     * - Hough Circle Detection: Circle parameters (min/max radius)
     * - Blob Sphere Detection: None (uses auto thresholding)
     * - Horizontal Interfaces Detection: Canny thresholds + Peak Height Ratio
     */
    private void updateParameterAvailability(String selectedMethod) {
        if (selectedMethod == null) return;
        
        // Default: disable all
        boolean cannyEnabled = false;
        boolean houghEnabled = false;
        boolean circleEnabled = false;
        boolean peakHeightEnabled = false;
        
        // Enable parameters based on selected method
        switch (selectedMethod) {
            case "Hough Line Detection":
                cannyEnabled = true;
                houghEnabled = true;
                break;
                
            case "RANSAC Line Detection":
                cannyEnabled = true;
                break;
                
            case "Hough Circle Detection":
                circleEnabled = true;
                break;
                
            case "Blob Sphere Detection":
                // No configurable parameters - uses auto thresholding
                break;
                
            case "Horizontal Interfaces Detection":
                cannyEnabled = true;
                peakHeightEnabled = true;
                break;
        }
        
        // Apply enable/disable state to Canny threshold controls
        setControlsEnabled(cannyEnabled, 
            cannyLabel, cannyThreshold1Slider, cannyThreshold1Label,
            cannyThreshold2Slider, cannyThreshold2Label,
            autoThresholdsButton);
        
        // Apply enable/disable state to Hough threshold controls
        setControlsEnabled(houghEnabled, 
            houghLabel, houghThresholdSlider, houghThresholdLabel);
        
        // Apply enable/disable state to Circle detection controls
        setControlsEnabled(circleEnabled, 
            circleLabel, minRadiusSlider, minRadiusLabel,
            maxRadiusSlider, maxRadiusLabel);
        
        // Apply enable/disable state to Peak Height Ratio controls
        setControlsEnabled(peakHeightEnabled, 
            interfaceLabel, peakHeightRatioSlider, peakHeightRatioLabel);
        
        logger.info("Updated parameter availability for: " + selectedMethod);
    }
    
    /**
     * Helper method to enable/disable multiple controls at once
     */
    private void setControlsEnabled(boolean enabled, javafx.scene.Node... controls) {
        for (javafx.scene.Node control : controls) {
            if (control != null) {
                control.setDisable(!enabled);
                // Optionally adjust opacity for visual feedback
                control.setOpacity(enabled ? 1.0 : 0.5);
            }
        }
    }
    
    /**
     * Set the image data manager
     */
    public void setImageDataManager(ImageDataManager manager) {
        this.imageDataManager = manager;
    }
    
    /**
     * Auto-calculate optimal Canny thresholds based on current image
     */
    @FXML
    public void autoCalculateThresholds() {
        if (currentImage == null || currentImage.empty()) {
            showAlert("No Image", "Please load an image first before auto-calculating thresholds");
            return;
        }
        
        try {
            // Calculate optimal thresholds using the detector's algorithm
            double[] thresholds = detector.estimateCannyThresholds(currentImage);
            
            // Update UI sliders
            Platform.runLater(() -> {
                cannyThreshold1Slider.setValue(thresholds[0]);
                cannyThreshold2Slider.setValue(thresholds[1]);
                
                resultsTextArea.clear();
                resultsTextArea.appendText("Auto-Calculated Thresholds\n");
                resultsTextArea.appendText("═══════════════════════════\n\n");
                resultsTextArea.appendText(String.format("Lower Threshold: %.1f\n", thresholds[0]));
                resultsTextArea.appendText(String.format("Upper Threshold: %.1f\n\n", thresholds[1]));
                resultsTextArea.appendText("Thresholds have been automatically adjusted\n");
                resultsTextArea.appendText("based on image gradient statistics.\n\n");
                resultsTextArea.appendText("You can now run detection or manually\n");
                resultsTextArea.appendText("fine-tune the values using the sliders.");
            });
            
            logger.info(String.format("Auto-calculated thresholds: %.1f / %.1f", thresholds[0], thresholds[1]));
            
        } catch (Exception e) {
            logger.warning("Failed to auto-calculate thresholds: " + e.getMessage());
            e.printStackTrace();
            showAlert("Error", "Failed to calculate thresholds: " + e.getMessage());
        }
    }
    
    /**
     * Load and display the current frame
     */
    @FXML
    public void loadCurrentFrame() {
        if (imageDataManager != null) {
            Mat latestFrame = imageDataManager.readMat();
            if (latestFrame != null && !latestFrame.empty()) {
                // Clone to freeze the frame so detection uses the exact same image
                currentImage = latestFrame.clone();
                Image img = MatToImageConverter.matToImage(currentImage);
                originalImageView.setImage(img);
                logger.info("Loaded current frame for detection");
            }
        }
    }
    
    /**
     * Load test image (needle.png) for detection and display
     */
    @FXML
    public void loadTestFrame() {
        try {
            // Load as Mat for detection first
            InputStream is = getClass().getResourceAsStream(defaultImagePath);
            if (is != null) {
                File tempFile = File.createTempFile("needle_temp", ".png");
                tempFile.deleteOnExit();
                Files.copy(is, tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                is.close();
                
                currentImage = Imgcodecs.imread(tempFile.getAbsolutePath());
                
                if (currentImage == null || currentImage.empty()) {
                    showAlert("Error", "Failed to load image as Mat. OpenCV may not be initialized.");
                    logger.warning("Failed to load image as Mat from: " + tempFile.getAbsolutePath());
                    return;
                }
                
                // Load for display
                Image testImg = new Image(getClass().getResourceAsStream(defaultImagePath));
                originalImageView.setImage(testImg);
                
                logger.info("Loaded test frame for detection (Size: " + currentImage.cols() + "x" + currentImage.rows() + ")");
            } else {
                showAlert("Error", "Could not find needle.png in resources");
            }
        } catch (Exception e) {
            logger.warning("Failed to load test image: " + e.getMessage());
            e.printStackTrace();
            showAlert("Error", "Failed to load test image: " + e.getMessage());
        }
    }
    
    /**
     * Handle detect button click
     */
    @FXML
    public void handleDetect(ActionEvent event) {
        if (currentImage == null || currentImage.empty()) {
            showAlert("No Image", "Please load an image first (use 'Load Current Frame' or 'Load Test Frame')");
            return;
        }
        
        String method = detectionMethodCombo.getValue();
        boolean visualize = visualizeCheckBox.isSelected();
        
        resultsTextArea.clear();
        resultsTextArea.appendText("Running: " + method + "...\n\n");
        
        // Run detection in background thread to avoid UI freezing
        new Thread(() -> {
            try {
                switch (method) {
                    case "Hough Line Detection":
                        detectHoughLines(visualize);
                        break;
                    case "RANSAC Line Detection":
                        detectRANSACLines(visualize);
                        break;
                    case "Hough Circle Detection":
                        detectHoughCircles(visualize);
                        break;
                    case "Blob Sphere Detection":
                        detectBlobSpheres(visualize);
                        break;
                    case "Horizontal Interfaces Detection":
                        detectHorizontalInterfaces(visualize);
                        break;
                    default:
                        Platform.runLater(() -> 
                            resultsTextArea.appendText("Unknown detection method\n"));
                }
            } catch (Exception e) {
                logger.severe("Detection error: " + e.getMessage());
                e.printStackTrace();
                Platform.runLater(() -> 
                    resultsTextArea.appendText("Error: " + e.getMessage() + "\n"));
            }
        }).start();
    }
    
    private void detectHoughLines(boolean visualize) {
        UltrasoundObjectDetector.LineDetectionResult result = 
            detector.detectLinesHough(currentImage, visualize);
        
        Platform.runLater(() -> {
            resultsTextArea.appendText("Found " + result.lineCount + " lines\n\n");
            
            for (int i = 0; i < result.lines.size(); i++) {
                double[] line = result.lines.get(i);
                double angle = Math.toDegrees(Math.atan2(line[3] - line[1], line[2] - line[0]));
                double length = Math.sqrt(Math.pow(line[2] - line[0], 2) + Math.pow(line[3] - line[1], 2));
                
                resultsTextArea.appendText(String.format("Line %d:\n", i + 1));
                resultsTextArea.appendText(String.format("  Start: (%.0f, %.0f)\n", line[0], line[1]));
                resultsTextArea.appendText(String.format("  End: (%.0f, %.0f)\n", line[2], line[3]));
                resultsTextArea.appendText(String.format("  Angle: %.1f°\n", angle));
                resultsTextArea.appendText(String.format("  Length: %.1f pixels\n\n", length));
            }
            
            if (visualize && result.visualizedImage != null) {
                Image img = MatToImageConverter.matToImage(result.visualizedImage);
                detectionImageView.setImage(img);
            }
        });
    }
    
    private void detectRANSACLines(boolean visualize) {
        UltrasoundObjectDetector.LineDetectionResult result = 
            detector.detectLineRANSAC(currentImage, visualize);
        
        Platform.runLater(() -> {
            resultsTextArea.appendText("Found " + result.lineCount + " lines\n\n");
            
            if (result.lines.isEmpty()) {
                resultsTextArea.appendText("No lines detected\n");
            } else {
                for (int i = 0; i < result.lines.size(); i++) {
                    double[] line = result.lines.get(i);
                    double angle = Math.toDegrees(Math.atan2(line[3] - line[1], line[2] - line[0]));
                    double length = Math.sqrt(Math.pow(line[2] - line[0], 2) + Math.pow(line[3] - line[1], 2));
                    
                    resultsTextArea.appendText(String.format("Line %d:\n", i + 1));
                    resultsTextArea.appendText(String.format("  Start: (%.0f, %.0f)\n", line[0], line[1]));
                    resultsTextArea.appendText(String.format("  End: (%.0f, %.0f)\n", line[2], line[3]));
                    resultsTextArea.appendText(String.format("  Angle: %.1f°\n", angle));
                    resultsTextArea.appendText(String.format("  Length: %.1f pixels\n", length));
                    
                    // Add confidence information
                    if (i < result.inlierCounts.size() && i < result.confidenceScores.size()) {
                        resultsTextArea.appendText(String.format("  Inliers: %d points\n", result.inlierCounts.get(i)));
                        resultsTextArea.appendText(String.format("  Confidence: %.1f%%\n", result.confidenceScores.get(i)));
                    }
                    resultsTextArea.appendText("\n");
                }
            }
            
            if (visualize && result.visualizedImage != null) {
                Image img = MatToImageConverter.matToImage(result.visualizedImage);
                detectionImageView.setImage(img);
            }
        });
    }
    
    private void detectHoughCircles(boolean visualize) {
        // Update circle parameters from sliders (improved for ultrasound)
        if (minRadiusSlider != null && maxRadiusSlider != null) {
            detector.setCircleParameters(1.2, 80, 100, 20, 
                (int) minRadiusSlider.getValue(), 
                (int) maxRadiusSlider.getValue());
        }
        
        UltrasoundObjectDetector.CircleDetectionResult result = 
            detector.detectCircles(currentImage, visualize);
        
        Platform.runLater(() -> {
            resultsTextArea.appendText("Found " + result.circleCount + " circles\n\n");
            
            for (int i = 0; i < result.circles.size(); i++) {
                double[] circle = result.circles.get(i);
                resultsTextArea.appendText(String.format("Circle %d:\n", i + 1));
                resultsTextArea.appendText(String.format("  Center: (%.0f, %.0f)\n", circle[0], circle[1]));
                resultsTextArea.appendText(String.format("  Radius: %.0f pixels\n\n", circle[2]));
            }
            
            if (visualize && result.visualizedImage != null) {
                Image img = MatToImageConverter.matToImage(result.visualizedImage);
                detectionImageView.setImage(img);
            }
        });
    }
    
    private void detectBlobSpheres(boolean visualize) {
        UltrasoundObjectDetector.CircleDetectionResult result = 
            detector.detectSpheresBlob(currentImage, visualize);
        
        Platform.runLater(() -> {
            resultsTextArea.appendText("Found " + result.circleCount + " blob spheres\n\n");
            
            for (int i = 0; i < result.circles.size(); i++) {
                double[] blob = result.circles.get(i);
                resultsTextArea.appendText(String.format("Sphere %d:\n", i + 1));
                resultsTextArea.appendText(String.format("  Center: (%.0f, %.0f)\n", blob[0], blob[1]));
                resultsTextArea.appendText(String.format("  Radius: %.0f pixels\n", blob[2]));
                
                // Add circularity score (quality metric)
                if (i < result.circularityScores.size()) {
                    resultsTextArea.appendText(String.format("  Circularity: %.2f (1.0 = perfect circle)\n", 
                        result.circularityScores.get(i)));
                }
                resultsTextArea.appendText("\n");
            }
            
            if (visualize && result.visualizedImage != null) {
                Image img = MatToImageConverter.matToImage(result.visualizedImage);
                detectionImageView.setImage(img);
            }
        });
    }
    
    private void detectHorizontalInterfaces(boolean visualize) {
        UltrasoundObjectDetector.LineDetectionResult result = 
            detector.detectHorizontalInterfaces(currentImage, visualize);
        
        Platform.runLater(() -> {
            resultsTextArea.appendText("Found " + result.lineCount + " horizontal interfaces\n\n");
            
            for (int i = 0; i < result.lines.size(); i++) {
                double[] line = result.lines.get(i);
                resultsTextArea.appendText(String.format("Interface %d at Y = %.0f\n", 
                    i + 1, line[1]));
            }
            
            if (visualize && result.visualizedImage != null) {
                Image img = MatToImageConverter.matToImage(result.visualizedImage);
                detectionImageView.setImage(img);
            }
        });
    }
    
    private void showAlert(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}

