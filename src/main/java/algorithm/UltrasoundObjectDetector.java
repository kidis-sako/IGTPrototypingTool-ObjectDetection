package algorithm;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * UltrasoundObjectDetector provides various OpenCV algorithms for detecting objects
 * in ultrasound images including lines (water bath bottom, needles), circles (spheres),
 * and other geometric shapes.
 *
 * @author Moritz Koehn, Kidis Sako
 */
public class UltrasoundObjectDetector {

    private static final Logger logger = Logger.getLogger(UltrasoundObjectDetector.class.getName());

    // Detection parameters that can be adjusted (optimized defaults for ultrasound)
    private double cannyThreshold1 = 30;   // Lowered for ultrasound (was 50)
    private double cannyThreshold2 = 90;   // Lowered for ultrasound (was 150)
    private int houghThreshold = 100;
    private double minLineLength = 50;
    private double maxLineGap = 10;

    // Circle detection parameters (tuned for ultrasound)
    private double dp = 1.2;
    private double minDist = 80;  // Increased to avoid duplicates
    private int circleParam1 = 100;
    private int circleParam2 = 20;  // Lowered for ultrasound (was 30)
    private int minRadius = 10;
    private int maxRadius = 200;

    // Interface detection parameters
    private double minPeakHeightRatio = 0.15;  // Ratio of image width (0.1-0.2 typical)


    /**
     * Result class for line detection
     */
    public static class LineDetectionResult {
        public List<double[]> lines;  // Each line: [x1, y1, x2, y2]
        public Mat visualizedImage;
        public int lineCount;
        public List<Integer> inlierCounts;  // Number of inliers per line (for RANSAC)
        public List<Double> confidenceScores;  // Confidence percentage per line (for RANSAC)

        public LineDetectionResult(List<double[]> lines, Mat visualizedImage) {
            this.lines = lines;
            this.visualizedImage = visualizedImage;
            this.lineCount = lines.size();
            this.inlierCounts = new ArrayList<>();
            this.confidenceScores = new ArrayList<>();
        }
    }

    /**
     * Result class for circle detection
     */
    public static class CircleDetectionResult {
        public List<double[]> circles;  // Each circle: [x, y, radius]
        public Mat visualizedImage;
        public int circleCount;
        public List<Double> circularityScores;  // Circularity score per circle (for blob detection)

        public CircleDetectionResult(List<double[]> circles, Mat visualizedImage) {
            this.circles = circles;
            this.visualizedImage = visualizedImage;
            this.circleCount = circles.size();
            this.circularityScores = new ArrayList<>();
        }
    }

    /**
     * Detects lines using Hough Transform - good for straight needles and water bath bottom
     *
     * @param image Input ultrasound image (grayscale or color)
     * @param visualize If true, returns an image with detected lines drawn
     * @return LineDetectionResult containing detected lines
     */
    public LineDetectionResult detectLinesHough(Mat image, boolean visualize) {
        Mat gray = preprocessImage(image);
        Mat edges = new Mat();

        // Apply Canny edge detection
        Imgproc.Canny(gray, edges, cannyThreshold1, cannyThreshold2);

        // Detect lines using Probabilistic Hough Transform
        Mat lines = new Mat();
        Imgproc.HoughLinesP(edges, lines, 1, Math.PI / 180, houghThreshold,
                           minLineLength, maxLineGap);

        List<double[]> linesList = new ArrayList<>();
        Mat visualImage = null;

        if (visualize) {
            visualImage = image.clone();
            if (visualImage.channels() == 1) {
                Imgproc.cvtColor(visualImage, visualImage, Imgproc.COLOR_GRAY2BGR);
            }
        }

        // Extract lines
        for (int i = 0; i < lines.rows(); i++) {
            double[] line = lines.get(i, 0);
            linesList.add(line);

            if (visualize) {
                Point pt1 = new Point(line[0], line[1]);
                Point pt2 = new Point(line[2], line[3]);
                Imgproc.line(visualImage, pt1, pt2, new Scalar(0, 255, 0), 2);
            }
        }

        logger.info("Hough Line Detection: Found " + linesList.size() + " lines");

        edges.release();
        lines.release();
        gray.release();

        return new LineDetectionResult(linesList, visualImage);
    }

    /**
     * Detects lines using iterative RANSAC - robust for noisy ultrasound images
     * Works for lines at any angle (horizontal, vertical, diagonal)
     * Detects multiple lines by iteratively finding and removing inliers
     *
     * @param image Input ultrasound image
     * @param visualize If true, returns visualization
     * @return LineDetectionResult with all detected lines
     */
    public LineDetectionResult detectLineRANSAC(Mat image, boolean visualize) {
        Mat gray = preprocessImage(image);
        Mat edges = new Mat();

        // Apply Canny edge detection
        Imgproc.Canny(gray, edges, cannyThreshold1, cannyThreshold2);

        // Extract all edge points using Core.findNonZero (much faster than nested loops)
        Mat edgePointsMat = new Mat();
        Core.findNonZero(edges, edgePointsMat);
        
        List<Point> edgePoints = new ArrayList<>();
        if (edgePointsMat.rows() > 0) {
            for (int i = 0; i < edgePointsMat.rows(); i++) {
                double[] pt = edgePointsMat.get(i, 0);
                edgePoints.add(new Point(pt[0], pt[1]));
            }
        }
        edgePointsMat.release();

        logger.info("RANSAC: Starting with " + edgePoints.size() + " edge points");

        Mat visualImage = null;
        if (visualize) {
            visualImage = image.clone();
            if (visualImage.channels() == 1) {
                Imgproc.cvtColor(visualImage, visualImage, Imgproc.COLOR_GRAY2BGR);
            }
        }

        List<double[]> detectedLines = new ArrayList<>();
        List<Integer> inlierCounts = new ArrayList<>();
        List<Double> confidenceScores = new ArrayList<>();

        if (edgePoints.size() < 2) {
            logger.warning("RANSAC: Not enough edge points detected");
            edges.release();
            gray.release();
            LineDetectionResult result = new LineDetectionResult(detectedLines, visualImage);
            return result;
        }

        // Iterative RANSAC: Find multiple lines by repeatedly finding and removing inliers
        int maxLines = 20;  // Maximum number of lines to detect
        int minInliers = 50;  // Minimum inliers to consider a valid line
        // Adaptive inlier threshold based on image scale (3px for 640x480 baseline)
        double baselineSize = 640.0;
        double imageScale = Math.sqrt(image.cols() * image.rows()) / baselineSize;
        double inlierThreshold = 3.0 * imageScale;
        int ransacIterations = 1000;  // RANSAC iterations per line
        double minInlierRatio = 0.02;  // Early-exit if inlier ratio falls below 2%
        
        java.util.Random rand = new java.util.Random();
        Scalar[] lineColors = {
            new Scalar(0, 0, 255),    // Red
            new Scalar(0, 255, 0),    // Green  
            new Scalar(255, 0, 0),    // Blue
            new Scalar(0, 255, 255),  // Yellow
            new Scalar(255, 0, 255),  // Magenta
            new Scalar(255, 255, 0)   // Cyan
        };

        List<Point> remainingPoints = new ArrayList<>(edgePoints);
        
        for (int lineNum = 0; lineNum < maxLines && remainingPoints.size() > minInliers; lineNum++) {
            int bestInlierCount = 0;
            double[] bestLine = null;
            List<Point> bestInliers = null;
            double bestVx = 0, bestVy = 0;
            
            // Run RANSAC to find best line in remaining points
            for (int iter = 0; iter < ransacIterations; iter++) {
                // Randomly sample 2 points
                Point p1 = remainingPoints.get(rand.nextInt(remainingPoints.size()));
                Point p2 = remainingPoints.get(rand.nextInt(remainingPoints.size()));
                
                // Skip if points are too close
                double dist = Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
                if (dist < 50) continue;
                
                // Calculate line parameters from these 2 points
                double dx = p2.x - p1.x;
                double dy = p2.y - p1.y;
                double length = Math.sqrt(dx * dx + dy * dy);
                double vx_candidate = dx / length;
                double vy_candidate = dy / length;
                
                // Count inliers and collect them
                List<Point> inliers = new ArrayList<>();
                for (Point p : remainingPoints) {
                    // Distance from point to line
                    double lineDistance = Math.abs((p.y - p1.y) * vx_candidate - (p.x - p1.x) * vy_candidate);
                    if (lineDistance < inlierThreshold) {
                        inliers.add(p);
                    }
                }
                
                // Keep best line
                if (inliers.size() > bestInlierCount) {
                    bestInlierCount = inliers.size();
                    bestInliers = inliers;
                    bestVx = vx_candidate;
                    bestVy = vy_candidate;
                    
                    // Calculate line endpoints spanning the image
                    // Handle both horizontal and vertical lines properly
                    double x0 = (p1.x + p2.x) / 2.0;
                    double y0 = (p1.y + p2.y) / 2.0;
                    
                    // Extend line to image boundaries
                    // Use parametric form to handle all angles
                    double x1, y1, x2, y2;
                    
                    if (Math.abs(vx_candidate) > Math.abs(vy_candidate)) {
                        // More horizontal - extend to left and right edges
                        x1 = 0;
                        y1 = y0 - (x0 * vy_candidate / vx_candidate);
                        x2 = image.cols() - 1;
                        y2 = y0 + ((image.cols() - 1 - x0) * vy_candidate / vx_candidate);
                    } else {
                        // More vertical - extend to top and bottom edges
                        y1 = 0;
                        x1 = x0 - (y0 * vx_candidate / vy_candidate);
                        y2 = image.rows() - 1;
                        x2 = x0 + ((image.rows() - 1 - y0) * vx_candidate / vy_candidate);
                    }
                    
                    bestLine = new double[]{x1, y1, x2, y2};
                }
            }
            
            // Check if we found a valid line
            if (bestLine != null && bestInlierCount >= minInliers) {
                // Early-exit: stop if inlier ratio falls below minimum threshold
                double inlierRatio = (double) bestInlierCount / edgePoints.size();
                if (inlierRatio < minInlierRatio) {
                    logger.info(String.format("RANSAC: Early exit - inlier ratio %.3f%% below threshold %.1f%%",
                        inlierRatio * 100, minInlierRatio * 100));
                    break;
                }
                
                // Clip line endpoints to image bounds
                bestLine = clipLineToImage(bestLine, image.cols(), image.rows());
                
                detectedLines.add(bestLine);
                inlierCounts.add(bestInlierCount);
                double confidence = 100.0 * bestInlierCount / edgePoints.size();
                confidenceScores.add(confidence);
                
                // Remove inliers from remaining points
                remainingPoints.removeAll(bestInliers);
                
                double angle = Math.toDegrees(Math.atan2(bestVy, bestVx));
                logger.info(String.format("RANSAC: Line %d - %d inliers (%.1f%%), angle=%.1f°, %d points remaining",
                    lineNum + 1, bestInlierCount, 
                    confidence,
                    angle, remainingPoints.size()));
                
                // Visualize if requested
                if (visualize && visualImage != null) {
                    Point pt1 = new Point(bestLine[0], bestLine[1]);
                    Point pt2 = new Point(bestLine[2], bestLine[3]);
                    Scalar color = lineColors[lineNum % lineColors.length];
                    Imgproc.line(visualImage, pt1, pt2, color, 2);
                    
                    // Add line number label
                    Point midpoint = new Point((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2);
                    String label = String.format("L%d: %.1f°", lineNum + 1, angle);
                    Imgproc.putText(visualImage, label, midpoint,
                                  Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
                }
            } else {
                // No more significant lines found
                logger.info("RANSAC: No more significant lines found");
                break;
            }
        }

        logger.info("RANSAC: Found " + detectedLines.size() + " lines total");

        edges.release();
        gray.release();

        LineDetectionResult result = new LineDetectionResult(detectedLines, visualImage);
        result.inlierCounts = inlierCounts;
        result.confidenceScores = confidenceScores;
        return result;
    }

    /**
     * Detects circles/spheres using Hough Circle Transform
     * Good for detecting spherical phantoms, bubbles, or circular targets
     *
     * @param image Input ultrasound image
     * @param visualize If true, returns visualization
     * @return CircleDetectionResult with detected circles
     */
    public CircleDetectionResult detectCircles(Mat image, boolean visualize) {
        Mat gray = preprocessImage(image);

        // Apply Gaussian blur to reduce noise
        Imgproc.GaussianBlur(gray, gray, new Size(9, 9), 2, 2);

        // Detect circles using Hough Circle Transform
        Mat circles = new Mat();
        Imgproc.HoughCircles(gray, circles, Imgproc.HOUGH_GRADIENT,
                            dp, minDist, circleParam1, circleParam2,
                            minRadius, maxRadius);

        List<double[]> circlesList = new ArrayList<>();
        Mat visualImage = null;

        if (visualize) {
            visualImage = image.clone();
            if (visualImage.channels() == 1) {
                Imgproc.cvtColor(visualImage, visualImage, Imgproc.COLOR_GRAY2BGR);
            }
        }

        // Extract circles
        for (int i = 0; i < circles.cols(); i++) {
            double[] circle = circles.get(0, i);
            circlesList.add(circle);

            if (visualize) {
                Point center = new Point(Math.round(circle[0]), Math.round(circle[1]));
                int radius = (int) Math.round(circle[2]);

                // Draw circle outline
                Imgproc.circle(visualImage, center, radius, new Scalar(0, 255, 0), 2);
                // Draw center
                Imgproc.circle(visualImage, center, 3, new Scalar(255, 0, 0), -1);

                // Add label
                String label = String.format("R:%d", radius);
                Imgproc.putText(visualImage, label,
                              new Point(center.x - 20, center.y - radius - 10),
                              Imgproc.FONT_HERSHEY_SIMPLEX, 0.5,
                              new Scalar(255, 255, 0), 2);
            }
        }

        logger.info("Circle Detection: Found " + circlesList.size() + " circles");

        circles.release();
        gray.release();

        return new CircleDetectionResult(circlesList, visualImage);
    }

    /**
     * Detects spheres using contour detection
     * Alternative method for sphere detection, works well with contrasting objects
     *
     * @param image Input ultrasound image
     * @param visualize If true, returns visualization
     * @return CircleDetectionResult with detected blobs (as circles)
     */
    public CircleDetectionResult detectSpheresBlob(Mat image, boolean visualize) {
        Mat gray = preprocessImage(image);

        // Apply binary threshold
        Mat binary = new Mat();
        Imgproc.threshold(gray, binary, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);

        // Find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binary, contours, hierarchy,
                            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        List<double[]> blobsList = new ArrayList<>();
        List<Double> circularityList = new ArrayList<>();
        Mat visualImage = null;

        if (visualize) {
            visualImage = image.clone();
            if (visualImage.channels() == 1) {
                Imgproc.cvtColor(visualImage, visualImage, Imgproc.COLOR_GRAY2BGR);
            }
        }

        // Analyze each contour
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);

            // Filter by area (adjust these values for your use case)
            if (area < 100 || area > 10000) {
                continue;
            }

            // Calculate circularity
            double perimeter = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
            double circularity = 4 * Math.PI * area / (perimeter * perimeter);

            // Filter by circularity (sphere-like objects have circularity close to 1)
            if (circularity > 0.6) {
                // Calculate enclosing circle
                MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
                Point center = new Point();
                float[] radius = new float[1];
                Imgproc.minEnclosingCircle(contour2f, center, radius);

                double[] blob = {center.x, center.y, radius[0]};
                blobsList.add(blob);
                circularityList.add(circularity);

                if (visualize) {
                    Imgproc.circle(visualImage, center, (int) radius[0],
                                 new Scalar(255, 0, 255), 2);
                    Imgproc.circle(visualImage, center, 2, new Scalar(0, 255, 0), -1);

                    // Add circularity label
                    String label = String.format("C:%.2f", circularity);
                    Imgproc.putText(visualImage, label,
                                  new Point(center.x - 20, center.y - radius[0] - 10),
                                  Imgproc.FONT_HERSHEY_SIMPLEX, 0.4,
                                  new Scalar(255, 255, 0), 1);
                }

                contour2f.release();
            }
        }

        logger.info("Blob Detection: Found " + blobsList.size() + " blobs");

        binary.release();
        hierarchy.release();
        gray.release();

        CircleDetectionResult result = new CircleDetectionResult(blobsList, visualImage);
        result.circularityScores = circularityList;
        return result;
    }

    /**
     * Detects multiple horizontal lines (e.g., multiple water bath interfaces)
     * Uses histogram projection to find horizontal features
     *
     * @param image Input ultrasound image
     * @param visualize If true, returns visualization
     * @return LineDetectionResult with detected horizontal lines
     */
    public LineDetectionResult detectHorizontalInterfaces(Mat image, boolean visualize) {
        Mat gray = preprocessImage(image);
        Mat edges = new Mat();

        // Apply edge detection
        Imgproc.Canny(gray, edges, cannyThreshold1, cannyThreshold2);

        // Create horizontal projection (sum of edge pixels per row)
        int[] projection = new int[edges.rows()];
        for (int y = 0; y < edges.rows(); y++) {
            int sum = 0;
            for (int x = 0; x < edges.cols(); x++) {
                sum += edges.get(y, x)[0] > 0 ? 1 : 0;
            }
            projection[y] = sum;
        }

        // Find peaks in projection (horizontal lines)
        // Use configurable ratio instead of hardcoded cols()/4
        int minPeakHeight = (int) (edges.cols() * minPeakHeightRatio);
        List<Integer> peaks = findPeaks(projection, minPeakHeight);

        List<double[]> linesList = new ArrayList<>();
        Mat visualImage = null;

        if (visualize) {
            visualImage = image.clone();
            if (visualImage.channels() == 1) {
                Imgproc.cvtColor(visualImage, visualImage, Imgproc.COLOR_GRAY2BGR);
            }
        }

        for (int y : peaks) {
            double[] line = {0, y, image.cols() - 1, y};
            linesList.add(line);

            if (visualize) {
                Imgproc.line(visualImage, new Point(0, y),
                           new Point(image.cols() - 1, y),
                           new Scalar(0, 255, 255), 2);
                Imgproc.putText(visualImage, "Interface", new Point(10, y - 5),
                              Imgproc.FONT_HERSHEY_SIMPLEX, 0.5,
                              new Scalar(0, 255, 255), 1);
            }
        }

        logger.info("Horizontal Interface Detection: Found " + linesList.size() + " interfaces");

        edges.release();
        gray.release();

        return new LineDetectionResult(linesList, visualImage);
    }

    /**
     * Automatically estimates optimal Canny thresholds for an ultrasound image
     * Uses gradient magnitude statistics for robust threshold selection
     *
     * @param image Input ultrasound image
     * @return double[2] array: [lowerThreshold, upperThreshold]
     */
    public double[] estimateCannyThresholds(Mat image) {
        Mat gray = new Mat();

        // Convert to grayscale if needed
        if (image.channels() == 3) {
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        // Apply CLAHE preprocessing (same as in detection)
        Mat enhanced = new Mat();
        org.opencv.imgproc.CLAHE clahe = Imgproc.createCLAHE(2.0, new Size(8, 8));
        clahe.apply(gray, enhanced);

        // Apply bilateral filter
        Mat denoised = new Mat();
        Imgproc.bilateralFilter(enhanced, denoised, 9, 75, 75);

        // Calculate gradient magnitudes using Sobel
        Mat gradX = new Mat();
        Mat gradY = new Mat();
        Imgproc.Sobel(denoised, gradX, org.opencv.core.CvType.CV_32F, 1, 0, 3);
        Imgproc.Sobel(denoised, gradY, org.opencv.core.CvType.CV_32F, 0, 1, 3);

        Mat magnitude = new Mat();
        Core.magnitude(gradX, gradY, magnitude);

        // Convert to 8-bit for percentile calculation
        Mat magnitude8U = new Mat();
        magnitude.convertTo(magnitude8U, org.opencv.core.CvType.CV_8U);

        // Calculate percentiles of gradient magnitudes
        // For ultrasound: use lower percentiles due to speckle noise
        MatOfDouble mean = new MatOfDouble();
        MatOfDouble stddev = new MatOfDouble();
        Core.meanStdDev(magnitude8U, mean, stddev);

        double meanGrad = mean.get(0, 0)[0];
        double stdGrad = stddev.get(0, 0)[0];

        // Threshold estimation based on gradient statistics
        // Lower threshold: mean - 0.5*std (conservative to preserve edges)
        // Upper threshold: mean + 1.5*std
        double lowerThreshold = Math.max(15, meanGrad - 0.5 * stdGrad);
        double upperThreshold = meanGrad + 1.5 * stdGrad;

        // Ensure proper ratio between thresholds (typically 2:1 to 3:1)
        if (upperThreshold < lowerThreshold * 2) {
            upperThreshold = lowerThreshold * 2.5;
        }

        // Clamp to reasonable ranges for ultrasound
        lowerThreshold = Math.max(10, Math.min(80, lowerThreshold));
        upperThreshold = Math.max(30, Math.min(200, upperThreshold));

        logger.info(String.format("Auto-estimated Canny thresholds: %.1f / %.1f (gradient mean=%.1f, std=%.1f)",
                lowerThreshold, upperThreshold, meanGrad, stdGrad));

        // Cleanup
        gray.release();
        enhanced.release();
        denoised.release();
        gradX.release();
        gradY.release();
        magnitude.release();
        magnitude8U.release();

        return new double[]{lowerThreshold, upperThreshold};
    }

    /**
     * Enhanced preprocessing for ultrasound images
     * Applies denoising, contrast enhancement, and normalization
     *
     * @param image Input image
     * @return Preprocessed grayscale image
     */
    private Mat preprocessImage(Mat image) {
        Mat gray = new Mat();

        // Convert to grayscale if needed
        if (image.channels() == 3) {
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }

        // Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        // Better than global equalization - avoids over-boosting speckle noise
        Mat enhanced = new Mat();
        org.opencv.imgproc.CLAHE clahe = Imgproc.createCLAHE(2.0, new Size(8, 8));
        clahe.apply(gray, enhanced);

        // Apply bilateral filter to reduce noise while preserving edges
        Mat denoised = new Mat();
        Imgproc.bilateralFilter(enhanced, denoised, 9, 75, 75);

        enhanced.release();
        gray.release();

        return denoised;
    }

    /**
     * Clips line endpoints to image boundaries
     * Prevents coordinates from going off-image for near-vertical/horizontal lines
     */
    private double[] clipLineToImage(double[] line, int width, int height) {
        double x1 = Math.max(0, Math.min(width - 1, line[0]));
        double y1 = Math.max(0, Math.min(height - 1, line[1]));
        double x2 = Math.max(0, Math.min(width - 1, line[2]));
        double y2 = Math.max(0, Math.min(height - 1, line[3]));
        return new double[]{x1, y1, x2, y2};
    }

    /**
     * Finds peaks in a 1D signal (for horizontal projection analysis)
     */
    private List<Integer> findPeaks(int[] signal, int minPeakHeight) {
        List<Integer> peaks = new ArrayList<>();

        for (int i = 1; i < signal.length - 1; i++) {
            if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1]
                && signal[i] > minPeakHeight) {
                peaks.add(i);
            }
        }

        return peaks;
    }

    // Getter and setter methods for parameters

    public void setCannyThresholds(double threshold1, double threshold2) {
        this.cannyThreshold1 = threshold1;
        this.cannyThreshold2 = threshold2;
    }

    public double getCannyThreshold1() {
        return this.cannyThreshold1;
    }

    public double getCannyThreshold2() {
        return this.cannyThreshold2;
    }

    public void setHoughThreshold(int threshold) {
        this.houghThreshold = threshold;
    }

    public void setLineParameters(double minLength, double maxGap) {
        this.minLineLength = minLength;
        this.maxLineGap = maxGap;
    }

    public void setCircleParameters(double dp, double minDist, int param1, int param2,
                                   int minRadius, int maxRadius) {
        this.dp = dp;
        this.minDist = minDist;
        this.circleParam1 = param1;
        this.circleParam2 = param2;
        this.minRadius = minRadius;
        this.maxRadius = maxRadius;
    }

    public void setMinPeakHeightRatio(double ratio) {
        this.minPeakHeightRatio = Math.max(0.05, Math.min(0.5, ratio));  // Clamp to [0.05, 0.5]
    }

    public double getMinPeakHeightRatio() {
        return this.minPeakHeightRatio;
    }

}

