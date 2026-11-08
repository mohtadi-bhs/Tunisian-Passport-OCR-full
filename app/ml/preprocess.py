import numpy as np
from typing import List, Tuple


class Preprocessor:
    def read_image(self, data: bytes) -> np.ndarray:
        import cv2
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img

    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
        import cv2
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img

    def crop(self, img: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]) -> np.ndarray:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return img
        return img[y1:y2, x1:x2]

    def warp_crop(self, img: np.ndarray, poly: List[Tuple[float, float]], margin_x: float = 0.0, margin_y: float = 0.0) -> np.ndarray:
        """Straighten an oriented field into an upright rectangle with the text reading left to right.

        The model may list the four corners in either direction and from any starting corner, so:
          1. the corners are put in clockwise order (on screen), which prevents a vertical flip;
          2. the long pair of edges is the text width;
          3. the width is forced to point right (+x), which prevents a 180-degree rotation;
          4. an optional margin is added (default 0: zero margins gave the best results on the
             passport photo, and wider margins pulled in neighbouring text).
        Falls back to the axis-aligned crop if the polygon is degenerate.
        """
        import cv2
        pts = np.asarray(poly, dtype=np.float32).reshape(4, 2)

        # 1) clockwise on screen (image y axis points down): positive shoelace area
        area2 = sum(pts[i, 0] * pts[(i + 1) % 4, 1] - pts[(i + 1) % 4, 0] * pts[i, 1] for i in range(4))
        if area2 < 0:
            pts = pts[[0, 3, 2, 1]]

        def edge(a, b):
            return float(np.linalg.norm(pts[a] - pts[b]))

        # 2) make edge 0-1 the long one (text width)
        if edge(1, 2) > edge(0, 1):
            pts = np.roll(pts, -1, axis=0)
        # 3) make the width point to the right, so the text is not rotated 180 degrees
        if pts[1, 0] < pts[0, 0]:
            pts = np.roll(pts, -2, axis=0)

        width = int(round(edge(0, 1)))
        height = int(round(edge(1, 2)))
        if width < 2 or height < 2:
            xs, ys = pts[:, 0], pts[:, 1]
            return self.crop(img, (xs.min(), ys.min(), xs.max(), ys.max()))

        # 4) optional margin around the box. The margin shows real image pixels from outside the box.
        px = int(round(margin_x * height))
        py = int(round(margin_y * height))
        dst = np.array([[px, py], [px + width - 1, py],
                        [px + width - 1, py + height - 1], [px, py + height - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        return cv2.warpPerspective(img, M, (width + 2 * px, height + 2 * py), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)