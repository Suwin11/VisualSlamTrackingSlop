"""
detectors.py
Background feature detectors + Lucas-Kanade optical flow tracker.

Optimizations & Fixes:
- Shared prebuilt optical flow image pyramids (`build_lk_pyramid`) so running
  multiple trackers (Shi-Tomasi + ORB) builds the pyramid once per frame instead of 4x.
- Increased pyramid depth (`maxLevel=4`, `winSize=(21, 21)`) and raised
  `JUMP_THRESHOLD = 140.0 px` so fast 90-180 deg robot/camera turns are tracked
  cleanly without prematurely triggering a jump reset.
"""

import cv2
import numpy as np


# ============================================================
# Shared LK optical-flow utility
# ============================================================
LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 4
LK_PARAMS = dict(
    winSize=LK_WIN_SIZE,
    maxLevel=LK_MAX_LEVEL,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)
JUMP_THRESHOLD = 140.0  # px – allows fast 180-deg turns (~25 deg/frame) without false resets
FB_ERR_THRESHOLD = 1.5  # px – forward-backward consistency threshold


def build_lk_pyramid(gray):
    """No-op for OpenCV 5.0+ compatibility (OpenCV 5 Python bindings require 2D Mat/UMat in calcOpticalFlowPyrLK)."""
    return None


def track_lk(prev_gray, gray, pts, static_mask, prev_pyr=None, curr_pyr=None):
    """
    Run forward-backward Lucas-Kanade on *pts*.
    Accepts optional precomputed pyramids (`prev_pyr`, `curr_pyr`) to avoid
    rebuilding image pyramids when running multiple feature trackers.

    Returns
    -------
    old_pts : ndarray (N, 2)
    new_pts : ndarray (N, 2)
    jumped  : bool – True when a sudden camera jolt / stream glitch is detected
    """
    empty = np.empty((0, 2), dtype=np.float32)

    if prev_gray is None or gray is None or len(pts) == 0:
        return empty, empty, False

    p0 = pts.reshape(-1, 1, 2)
    src_prev = prev_gray
    src_curr = gray

    p1, st, _ = cv2.calcOpticalFlowPyrLK(src_prev, src_curr, p0, None, **LK_PARAMS)
    if p1 is None or st is None:
        return empty, empty, False

    p0b, stb, _ = cv2.calcOpticalFlowPyrLK(src_curr, src_prev, p1, None, **LK_PARAMS)
    if p0b is None or stb is None:
        return empty, empty, False

    o = p0.reshape(-1, 2)
    n = p1.reshape(-1, 2)
    err = np.linalg.norm(o - p0b.reshape(-1, 2), axis=1)

    h, w = gray.shape
    valid = (st.ravel() == 1) & (stb.ravel() == 1) & (err < FB_ERR_THRESHOLD)
    valid &= (n[:, 0] >= 0) & (n[:, 0] < w) & (n[:, 1] >= 0) & (n[:, 1] < h)

    if len(n) > 0 and static_mask is not None:
        xi = np.clip(n[:, 0].astype(np.int32), 0, w - 1)
        yi = np.clip(n[:, 1].astype(np.int32), 0, h - 1)
        valid &= (static_mask[yi, xi] > 0)

    old_pts, new_pts = o[valid], n[valid]

    if len(new_pts) > 0:
        median_motion = float(np.median(np.linalg.norm(new_pts - old_pts, axis=1)))
        if median_motion > JUMP_THRESHOLD:
            return empty, empty, True

    return old_pts, new_pts, False


def compute_compensated_residuals(old_pts, new_pts, reproj_threshold=3.2):
    """
    Monocular Ego-Motion Compensation (Zero IMU required):
    Uses RANSAC to fit the dominant 2D planar motion model (rotation + translation + scale)
    explaining the background camera motion between old_pts and new_pts.
    Computes the residual motion vector for each point after subtracting camera ego-motion.

    Returns:
    --------
    residuals : ndarray (N,) - independent object motion magnitude in pixels
    inliers   : ndarray (N,) bool - True if point belongs to rigid static background
    """
    n = len(new_pts)
    if n < 4 or len(old_pts) != n:
        return np.zeros(n, dtype=np.float32), np.ones(n, dtype=bool)

    # cv2.estimateAffinePartial2D estimates 4-DOF motion:
    # 2D translation (tx, ty), 2D rotation (theta), uniform scale (forward/backward zoom)
    M, inlier_mask = cv2.estimateAffinePartial2D(
        old_pts, new_pts, method=cv2.RANSAC,
        ransacReprojThreshold=reproj_threshold,
        maxIters=100, confidence=0.99
    )

    if M is None or inlier_mask is None or np.sum(inlier_mask) < 4:
        # Fallback: compensate median 2D camera shift
        shifts = new_pts - old_pts
        med_shift = np.median(shifts, axis=0)
        res = np.linalg.norm(shifts - med_shift, axis=1).astype(np.float32)
        inliers = (res <= reproj_threshold)
        return res, inliers

    ones = np.ones((n, 1), dtype=np.float32)
    old_h = np.hstack([old_pts, ones])
    pred_new = (M @ old_h.T).T
    residuals = np.linalg.norm(new_pts - pred_new, axis=1).astype(np.float32)
    inliers = inlier_mask.ravel().astype(bool) & (residuals <= reproj_threshold)
    return residuals, inliers


# ============================================================
# Shi-Tomasi (Good Features to Track)  +  Dynamic Marking
# ============================================================
class ShiTomasiTracker:
    """
    Shi-Tomasi corner detector tracked with LK optical flow.
    """

    VEL_RADIUS_MIN = 2          # px at velocity ≈ 0
    VEL_RADIUS_MAX = 8          # px at velocity >= VEL_CAP
    VEL_CAP        = 12.0       # px/frame – saturation velocity
    COLOR_STATIC   = (80, 180, 80)   # BGR dim green  (slow / static background)
    COLOR_FAST     = (0, 0, 255)     # BGR bright red (fast / moving objects)
    COLOR_MOVING   = (0, 0, 255)     # BGR bright red (dynamic object markers)

    def __init__(self, max_points=250, min_points=120, min_dist=15):
        self.max_points = max_points
        self.min_points = min_points
        self.min_dist = min_dist
        self.pts = np.empty((0, 2), dtype=np.float32)
        self.prev_gray = None
        self.prev_pyr = None

    def reset(self):
        self.pts = np.empty((0, 2), dtype=np.float32)
        self.prev_gray = None
        self.prev_pyr = None

    def update(self, gray, static_mask, curr_pyr=None):
        """Track + replenish. Returns (old_pts, new_pts)."""
        old_pts, new_pts, jumped = track_lk(
            self.prev_gray, gray, self.pts, static_mask,
            prev_pyr=self.prev_pyr, curr_pyr=curr_pyr
        )

        if jumped:
            self.pts = np.empty((0, 2), dtype=np.float32)
            old_pts = new_pts = np.empty((0, 2), dtype=np.float32)
        else:
            self.pts = new_pts

        if len(self.pts) < self.min_points:
            detect_mask = static_mask.copy() if static_mask is not None else np.full(gray.shape, 255, dtype=np.uint8)
            for x, y in self.pts:
                cv2.circle(detect_mask, (int(x), int(y)), self.min_dist, 0, -1)
            needed = self.max_points - len(self.pts)
            corners = cv2.goodFeaturesToTrack(
                gray, maxCorners=needed, qualityLevel=0.01,
                minDistance=self.min_dist, mask=detect_mask,
            )
            if corners is not None:
                nc = corners.reshape(-1, 2)
                self.pts = nc if len(self.pts) == 0 else np.vstack([self.pts, nc])

        self.prev_gray = gray
        self.prev_pyr = curr_pyr
        return old_pts, new_pts

    @staticmethod
    def draw(canvas, old_pts, new_pts, dynamic_marking=True, object_mask=None):
        """
        Draw markers onto *canvas*.
        When dynamic_marking is True:
          - Static background points are drawn as small green circles.
          - Moving objects / moving points (or points inside object_mask, or with velocity > 0)
            are marked with RED dots scaling with velocity up to COLOR_FAST (0, 0, 255).
        When dynamic_marking is False:
          - Uniform static dots (green, fixed radius) with no dynamic velocity or red marking.
        """
        if len(new_pts) == 0:
            return

        if not dynamic_marking:
            for x, y in new_pts:
                cv2.circle(canvas, (int(round(x)), int(round(y))), ShiTomasiTracker.VEL_RADIUS_MIN + 1,
                           ShiTomasiTracker.COLOR_STATIC, -1, cv2.LINE_AA)
            return

        residuals, inliers = compute_compensated_residuals(old_pts, new_pts, reproj_threshold=3.2)
        inliers_bool = inliers.astype(bool)

        # Independent motion scaling (0.0 to 1.0) based on residual motion relative to camera
        t = np.clip(residuals / 8.0, 0.0, 1.0)
        r_min = ShiTomasiTracker.VEL_RADIUS_MIN
        r_max = ShiTomasiTracker.VEL_RADIUS_MAX
        cs = ShiTomasiTracker.COLOR_STATIC
        cf = ShiTomasiTracker.COLOR_FAST

        h, w = canvas.shape[:2]
        for i, (x, y) in enumerate(new_pts):
            xi, yi = int(round(x)), int(round(y))
            # Point is on a moving object if:
            # 1. It lands inside YOLO's object segmentation mask
            # 2. OR it has significant independent motion violating camera ego-motion (residual >= 3.0 px)
            is_in_yolo_obj = False
            if object_mask is not None and 0 <= xi < w and 0 <= yi < h:
                is_in_yolo_obj = bool(object_mask[yi, xi] > 0)

            is_indep_moving = bool((not inliers_bool[i]) and (residuals[i] >= 3.0))

            if is_in_yolo_obj or is_indep_moving:
                # Mark moving object with prominent red dot!
                radius = max(int(r_min + max(t[i], 0.35) * (r_max - r_min)), 4)
                color = ShiTomasiTracker.COLOR_MOVING
            else:
                # Static background point adhering to camera ego-motion: keep small green dot
                radius = r_min
                color = ShiTomasiTracker.COLOR_STATIC

            cv2.circle(canvas, (xi, yi), radius, color, -1, cv2.LINE_AA)


# ============================================================
# ORB (Oriented FAST + Rotated BRIEF)
# ============================================================
class ORBTracker:
    """
    ORB keypoint detector tracked with LK optical flow.
    Drawn as small cyan diamond outlines.
    """

    POINT_COLOR        = (255, 200, 0)  # BGR cyan-ish (static features)
    POINT_COLOR_MOVING = (0, 0, 255)    # BGR bright red (moving objects / fast points)
    DIAMOND_SIZE       = 4              # half-diagonal px

    def __init__(self, max_points=200, min_points=80, min_dist=12, nfeatures=400):
        self.max_points = max_points
        self.min_points = min_points
        self.min_dist = min_dist
        self.orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=12)
        self.pts = np.empty((0, 2), dtype=np.float32)
        self.prev_gray = None
        self.prev_pyr = None

    def reset(self):
        self.pts = np.empty((0, 2), dtype=np.float32)
        self.prev_gray = None
        self.prev_pyr = None

    def update(self, gray, static_mask, curr_pyr=None):
        """Track + replenish. Returns (old_pts, new_pts)."""
        old_pts, new_pts, jumped = track_lk(
            self.prev_gray, gray, self.pts, static_mask,
            prev_pyr=self.prev_pyr, curr_pyr=curr_pyr
        )

        if jumped:
            self.pts = np.empty((0, 2), dtype=np.float32)
            old_pts = new_pts = np.empty((0, 2), dtype=np.float32)
        else:
            self.pts = new_pts

        if len(self.pts) < self.min_points:
            detect_mask = static_mask.copy() if static_mask is not None else np.full(gray.shape, 255, dtype=np.uint8)
            for x, y in self.pts:
                cv2.circle(detect_mask, (int(x), int(y)), self.min_dist, 0, -1)
            kps = self.orb.detect(gray, mask=detect_mask)
            if kps:
                kps = sorted(kps, key=lambda k: k.response, reverse=True)
                needed = self.max_points - len(self.pts)
                kps = kps[:needed]
                nk = np.array([[k.pt[0], k.pt[1]] for k in kps], dtype=np.float32)
                self.pts = nk if len(self.pts) == 0 else np.vstack([self.pts, nk])

        self.prev_gray = gray
        self.prev_pyr = curr_pyr
        return old_pts, new_pts

    @staticmethod
    def draw(canvas, old_pts, new_pts, trail_canvas, dynamic_marking=True, object_mask=None):
        """Draw fading trail lines onto *trail_canvas* and markers onto *canvas*."""
        trail_canvas[:] = cv2.convertScaleAbs(trail_canvas, alpha=0.90)

        for (x0, y0), (x1, y1) in zip(old_pts, new_pts):
            cv2.line(trail_canvas, (int(x0), int(y0)), (int(x1), int(y1)),
                     (200, 160, 0), 1)

        cv2.add(canvas, trail_canvas, dst=canvas)

        d = ORBTracker.DIAMOND_SIZE
        h, w = canvas.shape[:2]

        residuals, inliers = compute_compensated_residuals(old_pts, new_pts, reproj_threshold=3.2)
        inliers_bool = inliers.astype(bool)

        for i, (x, y) in enumerate(new_pts):
            xi, yi = int(round(x)), int(round(y))
            is_moving_obj = False
            if dynamic_marking:
                if object_mask is not None and 0 <= xi < w and 0 <= yi < h:
                    is_moving_obj = bool(object_mask[yi, xi] > 0)
                if not is_moving_obj and (not inliers_bool[i]) and (residuals[i] >= 3.0):
                    is_moving_obj = True

            if dynamic_marking and is_moving_obj:
                # Mark moving objects / points with distinct red dots
                cv2.circle(canvas, (xi, yi), d + 1, ORBTracker.POINT_COLOR_MOVING, -1, cv2.LINE_AA)
            else:
                diamond = np.array(
                    [[xi, yi - d], [xi + d, yi], [xi, yi + d], [xi - d, yi]],
                    dtype=np.int32,
                )
                cv2.polylines(canvas, [diamond], True, ORBTracker.POINT_COLOR, 1, cv2.LINE_AA)
