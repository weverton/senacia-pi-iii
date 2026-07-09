# change_detector.py
import cv2

class ChangeDetector:
    def __init__(self, min_area=2000, diff_threshold=25):
        """
        Initializes the change detector.
        
        :param min_area: Minimum pixel area that must change to trigger a 'True' result.
                         Increase this to ignore small movements (like leaves or bugs).
        :param diff_threshold: Pixel intensity difference required to count as a change (0-255).
                               Decrease this to make it more sensitive to slight lighting changes.
        """
        self.min_area = min_area
        self.diff_threshold = diff_threshold
        self.previous_gray_frame = None

    def compare(self, frame):
        """
        Compares the current BGR frame with the previous one.
        
        :param frame: OpenCV image (BGR format)
        :return: tuple (is_changed: bool, change_percentage: float, delta_frame: np.array)
        """
        # 1. Convert to grayscale (much faster to process than color)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply Gaussian Blur to reduce camera sensor noise and minor lighting flickers
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 3. If it's the very first frame, save it and report no change
        if self.previous_gray_frame is None:
            self.previous_gray_frame = gray
            return True, 0.0, None

        # 4. Calculate the absolute difference between the current and previous frame
        frame_delta = cv2.absdiff(self.previous_gray_frame, gray)
        
        # 5. Threshold the delta image
        # Any pixel difference > diff_threshold becomes white (255), else black (0)
        thresh = cv2.threshold(frame_delta, self.diff_threshold, 255, cv2.THRESH_BINARY)[1]

        # 6. Dilate the thresholded image to fill in holes in the moving objects
        thresh = cv2.dilate(thresh, None, iterations=2)

        # 7. Find the outlines (contours) of the changed areas
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 8. Calculate the total area of the changed regions
        total_changed_area = 0
        for c in contours:
            total_changed_area += cv2.contourArea(c)

        # Calculate what percentage of the screen changed
        total_pixels = frame.shape[0] * frame.shape[1]
        change_percentage = (total_changed_area / total_pixels) * 100

        # 9. Determine if the change is significant enough to trigger an alert
        is_changed = total_changed_area > self.min_area

        # 10. Update the previous frame for the next loop iteration
        self.previous_gray_frame = gray

        return is_changed, change_percentage, frame_delta