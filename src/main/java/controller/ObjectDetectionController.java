package controller;

import algorithm.UltrasoundObjectDetector;
import algorithm.ImageDataManager;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.input.MouseEvent;
import javafx.geometry.Point2D;
import javafx.scene.paint.Color;
import javafx.scene.Cursor;
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
 * @author Kidis Sako
 */
public class ObjectDetectionController {

    @FXML
    private ImageView originalImageView;

    @FXML
    private ImageView detectionImageView;
    @FXML
    private Canvas originalOverlayCanvas;
    @FXML
    private Canvas detectionOverlayCanvas;

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
    private CheckBox visualizeCheckBox;

    @FXML
    private TitledPane parametersPanel;
    @FXML
    private ToggleButton measurementToggleButton;
    @FXML
    private Button clearMeasurementButton;
    @FXML
    private Label measurementStatusLabel;
    @FXML
    private Label originalMeasurementLabel;
    @FXML
    private Label detectionMeasurementLabel;
    @FXML
    private CheckBox calibrationModeCheckBox;
    @FXML
    private Label calibrationStatusLabel;
    @FXML
    private Label calibrationReferenceLabel;
    @FXML
    private Label calibrationScaleLabel;
    @FXML
    private TextField calibrationMmField;
    @FXML
    private Button applyCalibrationButton;

    private UltrasoundObjectDetector detector;
    private ImageDataManager imageDataManager;
    private Mat currentImage;
    private String defaultImagePath = "/needle.png";  // Path to default image resource
    private Logger logger = Logger.getLogger(this.getClass().getName());
    private static final String ORIGINAL_CONTEXT = "Original Image";
    private static final String DETECTION_CONTEXT = "Detection Image";
    private final MeasurementState originalMeasurementState = new MeasurementState();
    private final MeasurementState detectionMeasurementState = new MeasurementState();
    private final CalibrationState calibrationState = new CalibrationState();

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

        // Measurement tools
        initializeMeasurementControls();
        updateCalibrationUi();

        // Initially visualize is checked
        visualizeCheckBox.setSelected(true);

        logger.info("Object Detection Controller initialized");
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
    }

    /**
     * Set the image data manager
     */
    public void setImageDataManager(ImageDataManager manager) {
        this.imageDataManager = manager;
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
                resetMeasurementForContext(originalMeasurementState, originalOverlayCanvas, originalMeasurementLabel, ORIGINAL_CONTEXT);
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

                resetMeasurementForContext(originalMeasurementState, originalOverlayCanvas, originalMeasurementLabel, ORIGINAL_CONTEXT);
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
                resetMeasurementForContext(detectionMeasurementState, detectionOverlayCanvas, detectionMeasurementLabel, DETECTION_CONTEXT);
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
                resetMeasurementForContext(detectionMeasurementState, detectionOverlayCanvas, detectionMeasurementLabel, DETECTION_CONTEXT);
            }
        });
    }

    private void detectHoughCircles(boolean visualize) {
        // Update circle parameters from sliders
        if (minRadiusSlider != null && maxRadiusSlider != null) {
            detector.setCircleParameters(1.2, 50, 100, 30,
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
                resetMeasurementForContext(detectionMeasurementState, detectionOverlayCanvas, detectionMeasurementLabel, DETECTION_CONTEXT);
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
                resetMeasurementForContext(detectionMeasurementState, detectionOverlayCanvas, detectionMeasurementLabel, DETECTION_CONTEXT);
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
                resetMeasurementForContext(detectionMeasurementState, detectionOverlayCanvas, detectionMeasurementLabel, DETECTION_CONTEXT);
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

    private void initializeMeasurementControls() {
        configureOverlayCanvas(originalOverlayCanvas, originalImageView);
        configureOverlayCanvas(detectionOverlayCanvas, detectionImageView);
        if (measurementToggleButton != null) {
            measurementToggleButton.selectedProperty().addListener((obs, oldVal, enabled) -> {
                measurementToggleButton.setText(enabled ? "Measurement Enabled" : "Enable Measurement");
                measurementStatusLabel.setText(enabled
                        ? "Measurement enabled. Click two points in either image."
                        : "Measurement disabled");
            });
        }
        if (clearMeasurementButton != null) {
            clearMeasurementButton.setOnAction(e -> {
                resetAllMeasurements();
                if (calibrationState.isActive()) {
                    calibrationState.clearMeasurementReference();
                    calibrationReferenceLabel.setText("Reference: –");
                }
            });
        }
        if (originalOverlayCanvas != null) {
            originalOverlayCanvas.addEventHandler(MouseEvent.MOUSE_CLICKED,
                    event -> handleMeasurementEvent(event, originalImageView, originalOverlayCanvas,
                            originalMeasurementLabel, originalMeasurementState, ORIGINAL_CONTEXT));
        }
        if (detectionOverlayCanvas != null) {
            detectionOverlayCanvas.addEventHandler(MouseEvent.MOUSE_CLICKED,
                    event -> handleMeasurementEvent(event, detectionImageView, detectionOverlayCanvas,
                            detectionMeasurementLabel, detectionMeasurementState, DETECTION_CONTEXT));
        }
        resetAllMeasurements();
    }

    private void configureOverlayCanvas(Canvas canvas, ImageView view) {
        if (canvas == null || view == null) {
            return;
        }
        canvas.widthProperty().bind(view.fitWidthProperty());
        canvas.heightProperty().bind(view.fitHeightProperty());
        canvas.setMouseTransparent(false);
        canvas.setCursor(Cursor.CROSSHAIR);
    }

    private void handleMeasurementEvent(MouseEvent event, ImageView targetView, Canvas canvas,
                                        Label outputLabel, MeasurementState state, String contextLabel) {
        if (!measurementToggleButton.isSelected()) {
            return;
        }
        ImageTransform transform = createTransform(targetView, canvas);
        if (transform == null) {
            measurementStatusLabel.setText(contextLabel + ": No image available for measurement.");
            return;
        }
        Point2D imagePoint = transform.canvasToImage(event.getX(), event.getY());
        if (imagePoint == null) {
            measurementStatusLabel.setText(contextLabel + ": Click inside the image area.");
            return;
        }
        if (!state.hasFirstPoint()) {
            state.setFirstPoint(imagePoint);
            state.setSecondPoint(null);
            drawMeasurement(canvas, transform, state);
            measurementStatusLabel.setText(String.format("%s: First point (%.1f, %.1f) recorded.",
                    contextLabel, imagePoint.getX(), imagePoint.getY()));
            outputLabel.setText(contextLabel + ": Waiting for second point...");
            return;
        }
        if (!state.isComplete()) {
            state.setSecondPoint(imagePoint);
            drawMeasurement(canvas, transform, state);
            double distance = calculatePixelDistance(state);
            outputLabel.setText(String.format("%s: %.2f px", contextLabel, distance));
            appendMeasurementResult(contextLabel, state, distance);
            maybeHandleCalibration(distance);
            measurementStatusLabel.setText(String.format("%s measurement logged (%.2f px).",
                    contextLabel, distance));
            return;
        }
        // Third click starts a new measurement
        state.setFirstPoint(imagePoint);
        state.setSecondPoint(null);
        drawMeasurement(canvas, transform, state);
        measurementStatusLabel.setText(String.format("%s: First point (%.1f, %.1f) recorded.",
                contextLabel, imagePoint.getX(), imagePoint.getY()));
        outputLabel.setText(contextLabel + ": Waiting for second point...");
    }

    private double calculatePixelDistance(MeasurementState state) {
        Point2D p1 = state.getFirstPoint();
        Point2D p2 = state.getSecondPoint();
        if (p1 == null || p2 == null) {
            return 0;
        }
        return Math.hypot(p2.getX() - p1.getX(), p2.getY() - p1.getY());
    }

    private void drawMeasurement(Canvas canvas, ImageTransform transform, MeasurementState state) {
        if (canvas == null || transform == null) {
            return;
        }
        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.clearRect(0, 0, canvas.getWidth(), canvas.getHeight());
        if (!state.hasFirstPoint()) {
            return;
        }
        Point2D firstCanvas = transform.imageToCanvas(state.getFirstPoint());
        drawHandle(gc, firstCanvas);
        if (state.isComplete()) {
            Point2D secondCanvas = transform.imageToCanvas(state.getSecondPoint());
            gc.setStroke(Color.LIMEGREEN);
            gc.setLineWidth(2);
            gc.strokeLine(firstCanvas.getX(), firstCanvas.getY(), secondCanvas.getX(), secondCanvas.getY());
            drawHandle(gc, secondCanvas);
        }
    }

    private void drawHandle(GraphicsContext gc, Point2D point) {
        double size = 8;
        gc.setFill(Color.ORANGE);
        gc.fillOval(point.getX() - size / 2.0, point.getY() - size / 2.0, size, size);
    }

    private void appendMeasurementResult(String contextLabel, MeasurementState state, double distance) {
        if (resultsTextArea == null || state.getFirstPoint() == null || state.getSecondPoint() == null) {
            return;
        }
        resultsTextArea.appendText(String.format(
                "%s measurement: %.2f px (%.1f, %.1f) -> (%.1f, %.1f)%n",
                contextLabel,
                distance,
                state.getFirstPoint().getX(),
                state.getFirstPoint().getY(),
                state.getSecondPoint().getX(),
                state.getSecondPoint().getY()
        ));
        if (calibrationState.hasScale()) {
            double mmValue = distance * calibrationState.getScale();
            resultsTextArea.appendText(String.format("   => %.2f mm (Calibration active)%n", mmValue));
        }
    }

    private void resetMeasurementForContext(MeasurementState state, Canvas canvas, Label label, String prefix) {
        if (state == null) {
            return;
        }
        state.clear();
        if (canvas != null) {
            canvas.getGraphicsContext2D().clearRect(0, 0, canvas.getWidth(), canvas.getHeight());
        }
        if (label != null) {
            label.setText(prefix + ": –");
        }
    }

    private void resetAllMeasurements() {
        resetMeasurementForContext(originalMeasurementState, originalOverlayCanvas, originalMeasurementLabel, ORIGINAL_CONTEXT);
        resetMeasurementForContext(detectionMeasurementState, detectionOverlayCanvas, detectionMeasurementLabel, DETECTION_CONTEXT);
        if (measurementStatusLabel != null) {
            measurementStatusLabel.setText(measurementToggleButton != null && measurementToggleButton.isSelected()
                    ? "Measurement enabled. Click two points in either image."
                    : "Measurement disabled");
        }
        if (!calibrationState.isCalibrating()) {
            calibrationState.clearMeasurementReference();
            calibrationReferenceLabel.setText("Reference: –");
        }
    }

    private ImageTransform createTransform(ImageView imageView, Canvas canvas) {
        if (imageView == null || canvas == null || imageView.getImage() == null) {
            return null;
        }
        double canvasWidth = canvas.getWidth();
        double canvasHeight = canvas.getHeight();
        if (canvasWidth <= 0 || canvasHeight <= 0) {
            canvasWidth = imageView.getFitWidth();
            canvasHeight = imageView.getFitHeight();
        }
        if (canvasWidth <= 0 || canvasHeight <= 0) {
            return null;
        }
        return new ImageTransform(imageView.getImage(), canvasWidth, canvasHeight);
    }

    @FXML
    private void handleCalibrationToggle(ActionEvent event) {
        boolean enabled = calibrationModeCheckBox.isSelected();
        calibrationState.reset();
        calibrationState.setCalibrating(enabled);
        calibrationMmField.setDisable(!enabled);
        applyCalibrationButton.setDisable(!enabled);
        updateCalibrationUi();
    }

    @FXML
    private void handleApplyCalibration(ActionEvent event) {
        if (!calibrationState.hasReferenceDistance()) {
            showAlert("Calibration", "Please measure a reference distance in pixels first.");
            return;
        }
        String mmText = calibrationMmField.getText();
        if (mmText == null || mmText.isBlank()) {
            showAlert("Calibration", "Please enter the real distance in millimeters.");
            return;
        }
        double mmValue;
        try {
            mmValue = Double.parseDouble(mmText);
        } catch (NumberFormatException ex) {
            showAlert("Calibration", "Invalid millimeter value.");
            return;
        }
        if (mmValue <= 0) {
            showAlert("Calibration", "The distance must be greater than 0.");
            return;
        }
        if (calibrationState.getReferencePixels() <= 0) {
            showAlert("Calibration", "Reference distance invalid. Please measure again.");
            calibrationState.clearMeasurementReference();
            return;
        }
        double scale = mmValue / calibrationState.getReferencePixels();
        calibrationState.setScale(scale);
        calibrationState.setCalibrating(false);
        updateCalibrationUi();
        measurementStatusLabel.setText(String.format("Calibration active: 1 px = %.3f mm", scale));
    }

    private void maybeHandleCalibration(double distancePx) {
        if (!calibrationState.isCalibrating()) {
            return;
        }
        calibrationState.setReferencePixels(distancePx);
        calibrationReferenceLabel.setText(String.format("Reference: %.2f px", distancePx));
        measurementStatusLabel.setText("Reference distance stored. Please enter the millimeter value.");
    }

    private void updateCalibrationUi() {
        if (calibrationModeCheckBox == null) {
            return;
        }
        boolean enabled = calibrationModeCheckBox.isSelected();
        calibrationMmField.setDisable(!enabled);
        applyCalibrationButton.setDisable(!enabled);
        if (!enabled) {
            calibrationState.reset();
            calibrationStatusLabel.setText("Calibration disabled");
            calibrationReferenceLabel.setText("Reference: –");
            calibrationScaleLabel.setText("Scale: –");
            calibrationMmField.clear();
            return;
        }
        if (calibrationState.hasScale()) {
            calibrationStatusLabel.setText("Calibration active");
            calibrationScaleLabel.setText(String.format("Scale: 1 px = %.3f mm", calibrationState.getScale()));
        } else if (calibrationState.hasReferenceDistance()) {
            calibrationStatusLabel.setText("Reference stored – enter mm value");
            calibrationScaleLabel.setText("Scale: –");
        } else {
            calibrationStatusLabel.setText("Calibration active – pick reference");
            calibrationScaleLabel.setText("Scale: –");
        }
    }

    private static class CalibrationState {
        private boolean calibrating;
        private double referencePixels;
        private Double scale;

        boolean isCalibrating() {
            return calibrating;
        }

        void setCalibrating(boolean calibrating) {
            this.calibrating = calibrating;
        }

        boolean isActive() {
            return scale != null;
        }

        boolean hasScale() {
            return scale != null;
        }

        double getScale() {
            return scale == null ? 0 : scale;
        }

        void setScale(double scale) {
            this.scale = scale;
        }

        boolean hasReferenceDistance() {
            return referencePixels > 0;
        }

        void setReferencePixels(double referencePixels) {
            this.referencePixels = referencePixels;
        }

        double getReferencePixels() {
            return referencePixels;
        }

        void clearMeasurementReference() {
            this.referencePixels = 0;
        }

        void reset() {
            calibrating = false;
            referencePixels = 0;
            scale = null;
        }
    }

    private static class MeasurementState {
        private Point2D firstPoint;
        private Point2D secondPoint;

        void clear() {
            firstPoint = null;
            secondPoint = null;
        }

        boolean hasFirstPoint() {
            return firstPoint != null;
        }

        boolean isComplete() {
            return firstPoint != null && secondPoint != null;
        }

        Point2D getFirstPoint() {
            return firstPoint;
        }

        Point2D getSecondPoint() {
            return secondPoint;
        }

        void setFirstPoint(Point2D point) {
            this.firstPoint = point;
        }

        void setSecondPoint(Point2D point) {
            this.secondPoint = point;
        }
    }

    private static class ImageTransform {
        private final double ratio;
        private final double offsetX;
        private final double offsetY;
        private final double displayWidth;
        private final double displayHeight;

        ImageTransform(Image image, double canvasWidth, double canvasHeight) {
            double ratioX = canvasWidth / image.getWidth();
            double ratioY = canvasHeight / image.getHeight();
            this.ratio = Math.min(ratioX, ratioY);
            this.displayWidth = image.getWidth() * ratio;
            this.displayHeight = image.getHeight() * ratio;
            this.offsetX = (canvasWidth - displayWidth) / 2.0;
            this.offsetY = (canvasHeight - displayHeight) / 2.0;
        }

        Point2D canvasToImage(double canvasX, double canvasY) {
            double adjustedX = canvasX - offsetX;
            double adjustedY = canvasY - offsetY;
            if (adjustedX < 0 || adjustedY < 0 || adjustedX > displayWidth || adjustedY > displayHeight) {
                return null;
            }
            double imageX = adjustedX / ratio;
            double imageY = adjustedY / ratio;
            return new Point2D(imageX, imageY);
        }

        Point2D imageToCanvas(Point2D imagePoint) {
            double canvasX = imagePoint.getX() * ratio + offsetX;
            double canvasY = imagePoint.getY() * ratio + offsetY;
            return new Point2D(canvasX, canvasY);
        }
    }
}
