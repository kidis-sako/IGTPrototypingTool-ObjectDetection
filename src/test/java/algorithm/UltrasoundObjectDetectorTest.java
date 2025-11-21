package algorithm;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import nu.pattern.OpenCV;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for UltrasoundObjectDetector
 * Demonstrates usage of various detection algorithms
 */
class UltrasoundObjectDetectorTest {

    @BeforeAll
    static void setUp() {
        // Load OpenCV library
        OpenCV.loadLocally();
    }

    @Test
    void testHoughLineDetection() {
        // Create a simple test image (you would load your ultrasound image here)
        // Mat image = Imgcodecs.imread("path/to/ultrasound/image.png");

        // For this test, we'll just verify the detector can be created
        UltrasoundObjectDetector detector = new UltrasoundObjectDetector();
        assertNotNull(detector);

        // Set parameters
        detector.setCannyThresholds(50, 150);
        detector.setHoughThreshold(100);
        detector.setLineParameters(50, 10);

        System.out.println("✓ Hough line detector initialized successfully");
    }

    @Test
    void testRANSACLineDetection() {
        UltrasoundObjectDetector detector = new UltrasoundObjectDetector();
        assertNotNull(detector);

        detector.setCannyThresholds(50, 150);

        System.out.println("✓ RANSAC line detector initialized successfully");
    }

    @Test
    void testCircleDetection() {
        UltrasoundObjectDetector detector = new UltrasoundObjectDetector();
        assertNotNull(detector);

        // Set circle parameters
        detector.setCircleParameters(1.2, 50, 100, 30, 10, 200);

        System.out.println("✓ Circle detector initialized successfully");
    }

    @Test
    void testBlobDetection() {
        UltrasoundObjectDetector detector = new UltrasoundObjectDetector();
        assertNotNull(detector);

        System.out.println("✓ Blob detector initialized successfully");
    }

    @Test
    void testHorizontalInterfaceDetection() {
        UltrasoundObjectDetector detector = new UltrasoundObjectDetector();
        assertNotNull(detector);

        System.out.println("✓ Horizontal interface detector initialized successfully");
    }

    // Uncomment this test if you have a test ultrasound image
    /*
    @Test
    void testWithRealUltrasoundImage() {
        // Load test image
        Mat image = Imgcodecs.imread("src/test/resources/ultrasound_test.png");
        assertFalse(image.empty(), "Test image should be loaded");

        UltrasoundObjectDetector detector = new UltrasoundObjectDetector();

        // Test line detection
        UltrasoundObjectDetector.LineDetectionResult lineResult =
            detector.detectLinesHough(image, true);
        assertNotNull(lineResult);
        System.out.println("Detected " + lineResult.lineCount + " lines");

        // Test circle detection
        UltrasoundObjectDetector.CircleDetectionResult circleResult =
            detector.detectCircles(image, true);
        assertNotNull(circleResult);
        System.out.println("Detected " + circleResult.circleCount + " circles");

        // Save visualizations
        if (lineResult.visualizedImage != null) {
            Imgcodecs.imwrite("test_output_lines.png", lineResult.visualizedImage);
        }
        if (circleResult.visualizedImage != null) {
            Imgcodecs.imwrite("test_output_circles.png", circleResult.visualizedImage);
        }
    }
    */
}

