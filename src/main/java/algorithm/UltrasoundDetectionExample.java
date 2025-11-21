package algorithm;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Practical examples of using UltrasoundObjectDetector for common tasks
 *
 * @author Moritz Koehn, Kidis Sako
 */
public class UltrasoundDetectionExample {

    /**
     * Example 1: Detect water bath bottom in ultrasound image
     * Returns the Y-coordinate of the water bath bottom
     */
    public static double detectWaterBathBottom(Mat image) {
        UltrasoundObjectDetector detector = new UltrasoundObjectDetector();

        // Use RANSAC for robust multi-line detection
        // Works for lines at any angle
        // Auto-estimate thresholds for best results
        double[] thresholds = detector.estimateCannyThresholds(image);
        detector.setCannyThresholds(thresholds[0], thresholds[1]);
        
        UltrasoundObjectDetector.LineDetectionResult result =
            detector.detectLineRANSAC(image, false);

        if (!result.lines.isEmpty()) {
            // Get first line (most prominent, often water bath bottom)
            double[] line = result.lines.get(0);
            // Return average Y coordinate
            return (line[1] + line[3]) / 2.0;
        }

        return -1; // Not found
    }

    /**
     * Example 2: Detect needle in ultrasound image
     * Returns angle and endpoints of detected needle
     */
    public static class NeedleInfo {
        public double x1, y1, x2, y2;
        public double angle;
        public double length;

        public NeedleInfo(double x1, double y1, double x2, double y2) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.angle = Math.toDegrees(Math.atan2(y2 - y1, x2 - x1));
            this.length = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
        }

        @Override
        public String toString() {
            return String.format("Needle: (%.1f,%.1f) to (%.1f,%.1f), angle=%.1f°, length=%.1f px",
                               x1, y1, x2, y2, angle, length);
        }
    }

    public static NeedleInfo detectNeedle(Mat image) {
        UltrasoundObjectDetector detector = new UltrasoundObjectDetector();

        // Use auto-thresholds for adaptive detection
        double[] thresholds = detector.estimateCannyThresholds(image);
        detector.setCannyThresholds(thresholds[0], thresholds[1]);
        detector.setLineParameters(100, 15);  // Needles are usually long
        detector.setHoughThreshold(80);

        UltrasoundObjectDetector.LineDetectionResult result =
            detector.detectLinesHough(image, false);

        if (!result.lines.isEmpty()) {
            // Find longest line (likely the needle)
            double[] longestLine = result.lines.stream()
                .max((a, b) -> {
                    double lenA = Math.sqrt(Math.pow(a[2] - a[0], 2) + Math.pow(a[3] - a[1], 2));
                    double lenB = Math.sqrt(Math.pow(b[2] - b[0], 2) + Math.pow(b[3] - b[1], 2));
                    return Double.compare(lenA, lenB);
                })
                .orElse(null);

            if (longestLine != null) {
                return new NeedleInfo(longestLine[0], longestLine[1],
                                    longestLine[2], longestLine[3]);
            }
        }

        return null;
    }

    /**
     * Example 3: Detect calibration spheres and sort by size
     */
    public static class SphereInfo {
        public double x, y, radius;

        public SphereInfo(double x, double y, double radius) {
            this.x = x;
            this.y = y;
            this.radius = radius;
        }

        @Override
        public String toString() {
            return String.format("Sphere: center=(%.1f, %.1f), radius=%.1f px", x, y, radius);
        }
    }

    public static List<SphereInfo> detectCalibrationSpheres(Mat image,
                                                           int minRadius,
                                                           int maxRadius) {
        UltrasoundObjectDetector detector = new UltrasoundObjectDetector();

        // Set parameters for sphere detection (optimized for ultrasound)
        detector.setCircleParameters(1.2, 80, 100, 20, minRadius, maxRadius);

        UltrasoundObjectDetector.CircleDetectionResult result =
            detector.detectCircles(image, false);

        // Convert to SphereInfo objects and sort by radius
        return result.circles.stream()
            .map(c -> new SphereInfo(c[0], c[1], c[2]))
            .sorted((a, b) -> Double.compare(b.radius, a.radius))  // Largest first
            .collect(Collectors.toList());
    }

    /**
     * Example 4: Detect all horizontal interfaces (water layers, tissue boundaries)
     */
    public static List<Double> detectHorizontalLayers(Mat image) {
        UltrasoundObjectDetector detector = new UltrasoundObjectDetector();

        // Use auto-thresholds for best results
        double[] thresholds = detector.estimateCannyThresholds(image);
        detector.setCannyThresholds(thresholds[0], thresholds[1]);
        
        UltrasoundObjectDetector.LineDetectionResult result =
            detector.detectHorizontalInterfaces(image, false);

        // Return Y-coordinates sorted from top to bottom
        return result.lines.stream()
            .map(line -> line[1])  // Get Y coordinate
            .sorted()
            .collect(Collectors.toList());
    }

    /**
     * Example 5: Track needle tip over multiple frames
     */
    public static class NeedleTipTracker {
        private double lastX = -1;
        private double lastY = -1;
        private final double maxJumpDistance = 50; // Maximum pixel jump between frames

        public Point2D trackNeedleTip(Mat image) {
            UltrasoundObjectDetector detector = new UltrasoundObjectDetector();

            // Detect circles (needle tip often appears as bright spot)
            // Optimized for ultrasound with lower param2
            detector.setCircleParameters(1.2, 50, 100, 15, 3, 15);
            UltrasoundObjectDetector.CircleDetectionResult result =
                detector.detectCircles(image, false);

            if (result.circles.isEmpty()) {
                return null;
            }

            double[] bestCircle = result.circles.get(0);

            // If we have previous position, use proximity to track
            if (lastX >= 0) {
                bestCircle = result.circles.stream()
                    .min((a, b) -> {
                        double distA = Math.sqrt(Math.pow(a[0] - lastX, 2) + Math.pow(a[1] - lastY, 2));
                        double distB = Math.sqrt(Math.pow(b[0] - lastX, 2) + Math.pow(b[1] - lastY, 2));
                        return Double.compare(distA, distB);
                    })
                    .orElse(bestCircle);

                // Check if jump is too large (likely false detection)
                double distance = Math.sqrt(Math.pow(bestCircle[0] - lastX, 2) +
                                          Math.pow(bestCircle[1] - lastY, 2));
                if (distance > maxJumpDistance) {
                    return new Point2D(lastX, lastY);  // Return previous position
                }
            }

            lastX = bestCircle[0];
            lastY = bestCircle[1];

            return new Point2D(lastX, lastY);
        }

        public void reset() {
            lastX = -1;
            lastY = -1;
        }
    }

    public static class Point2D {
        public double x, y;

        public Point2D(double x, double y) {
            this.x = x;
            this.y = y;
        }

        @Override
        public String toString() {
            return String.format("(%.1f, %.1f)", x, y);
        }
    }

    /**
     * Example 6: Combined detection - try multiple algorithms
     */
    public static List<SphereInfo> detectSpheresRobust(Mat image) {
        UltrasoundObjectDetector detector = new UltrasoundObjectDetector();

        // Try Hough circles first (optimized for ultrasound)
        detector.setCircleParameters(1.2, 80, 100, 20, 10, 200);
        UltrasoundObjectDetector.CircleDetectionResult houghResult =
            detector.detectCircles(image, false);

        // If Hough finds spheres, use those
        if (houghResult.circleCount > 0) {
            System.out.println("Using Hough Circle detection: found " +
                             houghResult.circleCount + " spheres");
            return houghResult.circles.stream()
                .map(c -> new SphereInfo(c[0], c[1], c[2]))
                .collect(Collectors.toList());
        }

        // Otherwise try blob detection
        UltrasoundObjectDetector.CircleDetectionResult blobResult =
            detector.detectSpheresBlob(image, false);

        System.out.println("Using Blob detection: found " +
                         blobResult.circleCount + " spheres");

        return blobResult.circles.stream()
            .map(c -> new SphereInfo(c[0], c[1], c[2]))
            .collect(Collectors.toList());
    }

    /**
     * Example 7: Detect and classify objects by angle
     */
    public static void detectAndClassifyLines(Mat image) {
        UltrasoundObjectDetector detector = new UltrasoundObjectDetector();

        // Auto-calculate optimal thresholds
        double[] thresholds = detector.estimateCannyThresholds(image);
        detector.setCannyThresholds(thresholds[0], thresholds[1]);
        detector.setLineParameters(50, 10);
        UltrasoundObjectDetector.LineDetectionResult result =
            detector.detectLinesHough(image, false);

        int horizontalLines = 0;
        int verticalLines = 0;
        int diagonalLines = 0;

        for (double[] line : result.lines) {
            double angle = Math.abs(Math.toDegrees(
                Math.atan2(line[3] - line[1], line[2] - line[0])));

            if (angle < 10 || angle > 170) {
                horizontalLines++;
            } else if (angle > 80 && angle < 100) {
                verticalLines++;
            } else {
                diagonalLines++;
            }
        }

        System.out.println("Line Classification:");
        System.out.println("  Horizontal: " + horizontalLines);
        System.out.println("  Vertical: " + verticalLines);
        System.out.println("  Diagonal: " + diagonalLines);
    }

    /**
     * Main method with usage examples
     */
    public static void main(String[] args) {
        // Note: Replace with actual ultrasound image path
        String imagePath = "path/to/ultrasound/image.png";
        Mat image = Imgcodecs.imread(imagePath);

        if (image.empty()) {
            System.err.println("Could not load image from: " + imagePath);
            System.out.println("\nThis is a demonstration of available detection methods.");
            System.out.println("Load an ultrasound image to see actual results.");
            return;
        }

        System.out.println("=== Ultrasound Object Detection Examples ===\n");

        // Example 1: Water bath
        System.out.println("1. Detecting water bath bottom...");
        double bathY = detectWaterBathBottom(image);
        if (bathY >= 0) {
            System.out.println("   Water bath bottom at Y = " + bathY + " pixels\n");
        } else {
            System.out.println("   Water bath not detected\n");
        }

        // Example 2: Needle
        System.out.println("2. Detecting needle...");
        NeedleInfo needle = detectNeedle(image);
        if (needle != null) {
            System.out.println("   " + needle + "\n");
        } else {
            System.out.println("   Needle not detected\n");
        }

        // Example 3: Calibration spheres
        System.out.println("3. Detecting calibration spheres...");
        List<SphereInfo> spheres = detectCalibrationSpheres(image, 20, 100);
        if (!spheres.isEmpty()) {
            for (int i = 0; i < spheres.size(); i++) {
                System.out.println("   " + (i + 1) + ". " + spheres.get(i));
            }
            System.out.println();
        } else {
            System.out.println("   No spheres detected\n");
        }

        // Example 4: Horizontal layers
        System.out.println("4. Detecting horizontal interfaces...");
        List<Double> layers = detectHorizontalLayers(image);
        if (!layers.isEmpty()) {
            for (int i = 0; i < layers.size(); i++) {
                System.out.println("   Layer " + (i + 1) + " at Y = " + layers.get(i));
            }
            System.out.println();
        } else {
            System.out.println("   No interfaces detected\n");
        }

        // Example 7: Classify lines
        System.out.println("7. Classifying detected lines...");
        detectAndClassifyLines(image);
    }
}

