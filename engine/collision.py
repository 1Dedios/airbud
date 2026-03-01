# collision.py
import cv2

class MotionCollisionDetector:
    def __init__(self, history=200, threshold=25, motion_ratio=0.10):
        self.bg_region = None
        self.motion_ratio = motion_ratio

    def hand_over_target(self, frame, target):
        x, y, s = target.get_region()

        # Ensure crop is inside frame bounds
        if y + s > frame.shape[0] or x + s > frame.shape[1]:
            return False

        crop = frame[y:y+s, x:x+s]

        # If crop is empty, skip detection
        if crop.size == 0:
            return False

        # Initialize background region
        if self.bg_region is None:
            self.bg_region = crop.copy()
            return False

        # If background size doesn't match crop (target moved), reset it
        if crop.shape != self.bg_region.shape:
            self.bg_region = crop.copy()
            return False

        # Compute difference
        diff = cv2.absdiff(crop, self.bg_region)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

        motion_pixels = cv2.countNonZero(thresh)
        return motion_pixels > (s * s * self.motion_ratio)
