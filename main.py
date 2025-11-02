import sys
import os
import argparse
import glob
import pickle
import hashlib
import tempfile
import subprocess
from datetime import datetime
from multiprocessing import Pool, cpu_count, freeze_support
# --- ▼▼▼【追加】並列処理とプログレスバーのためにライブラリをインポート ▼▼▼ ---
import concurrent.futures
from functools import partial
try:
    from tqdm import tqdm
except ImportError:
    print("警告: tqdmライブラリが見つかりません。進捗バーなしで処理を続行します。")
    print("より良い体験のために 'pip install tqdm' の実行をお勧めします。")
    # tqdmがない場合のダミー関数を定義
    def tqdm(iterable, **kwargs):
        return iterable
# --- ▲▲▲ 追加はここまで ▲▲▲ ---


import cv2
import numpy as np
from PySide6.QtCore import (QObject, Qt, QThread, Signal, Slot, QSize, QPointF,
                            QRectF, QPoint, QMargins)
from PySide6.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QPolygonF,
                           QBrush, QIcon)
from PySide6.QtWidgets import (QApplication, QMainWindow, QGraphicsView,
                               QGraphicsScene, QGraphicsPixmapItem,
                               QVBoxLayout, QHBoxLayout, QWidget, QLabel,
                               QPushButton, QListWidget, QListWidgetItem,
                               QDialog, QDialogButtonBox, QMessageBox,
                               QGraphicsRectItem, QGraphicsPolygonItem)


CACHE_DIR = ".Photogrammetry_Ortho_Image_Refiner_cache"


def _get_image_hash(image_path):
    hasher = hashlib.md5()
    with open(image_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def save_cache(ortho_path, detail_path, data):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    cache_filename = (
        f"{_get_image_hash(ortho_path)}_{_get_image_hash(detail_path)}.pkl"
    )
    with open(os.path.join(CACHE_DIR, cache_filename), 'wb') as f:
        pickle.dump(data, f)


def load_cache(ortho_path, detail_path):
    cache_filename = (
        f"{_get_image_hash(ortho_path)}_{_get_image_hash(detail_path)}.pkl"
    )
    cache_path = os.path.join(CACHE_DIR, cache_filename)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            try:
                return pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                print(f"\n警告: キャッシュファイルの読み込みに失敗しました: {cache_path}")
                return None
    return None


def export_to_psd(original_ortho_path, png_layer_paths, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    try: script_dir = os.path.dirname(os.path.realpath(__file__))
    except NameError: script_dir = os.getcwd()
    img2psd_script_path = os.path.join(script_dir, 'img2psd.py')
    if not os.path.exists(img2psd_script_path): raise FileNotFoundError(f"img2psd.py が見つかりません: {img2psd_script_path}")
    all_image_paths = [os.path.abspath(original_ortho_path)] + [os.path.abspath(p) for p in png_layer_paths]
    command = [sys.executable, img2psd_script_path] + all_image_paths
    try:
        result = subprocess.run(command, cwd=output_dir, check=True, capture_output=True)
        try:
            stdout_text = result.stdout.decode('utf-8')
            stderr_text = result.stderr.decode('utf-8')
        except UnicodeDecodeError:
            stdout_text = result.stdout.decode('cp932', errors='replace')
            stderr_text = result.stderr.decode('cp932', errors='replace')
        output_lines = stdout_text.strip().splitlines()
        if not output_lines:
            if stderr_text: raise RuntimeError(f"img2psd.py の実行に失敗: {stderr_text}")
            raise RuntimeError("img2psd.py が標準出力を返しませんでした。")
        last_line = output_lines[-1]
        prefix = "PSDファイルを保存しました: "
        if last_line.startswith(prefix): return last_line[len(prefix):].strip()
        else: raise RuntimeError(f"img2psd.pyが処理に失敗しました: {stdout_text}")
    except subprocess.CalledProcessError as e:
        try:
            stdout_text = e.stdout.decode('utf-8')
            stderr_text = e.stderr.decode('utf-8')
        except UnicodeDecodeError:
            stdout_text = e.stdout.decode('cp932', errors='replace')
            stderr_text = e.stderr.decode('cp932', errors='replace')
        error_message = stderr_text if stderr_text else stdout_text
        raise RuntimeError(f"PSDエクスポート失敗: {error_message}")
    except Exception as e: raise


RESIZE_WIDTH_COARSE = 2500
SIFT_NFEATURES_COARSE = 10000
TILE_SIZE = 1024
TILE_OVERLAP = 128
SIFT_NFEATURES_PER_TILE = 5000
MAX_FEATURES_FOR_MATCHING = 80000
HULL_MARGIN_PIXELS = 200
DEDUPLICATION_GRID_SIZE = 5.0
RATIO_TEST_THRESHOLD = 0.75
RANSAC_REPROJ_THRESHOLD = 3.0


def detect_features_in_tile(args):
    tile, offset_x, offset_y = args
    gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES_PER_TILE)
    akaze = cv2.AKAZE_create()
    kp_sift = sift.detect(gray_tile, None)
    kp_akaze = akaze.detect(gray_tile, None)
    combined_keypoints = list(kp_sift) + list(kp_akaze)
    return [(kp.pt[0] + offset_x, kp.pt[1] + offset_y, kp.response, kp.size,
             kp.angle, kp.octave, kp.class_id) for kp in combined_keypoints]

def perform_coarse_alignment_headless(ortho_img, detail_img):
    ortho_h, ortho_w = ortho_img.shape[:2]
    detail_h, detail_w = detail_img.shape[:2]
    scale_o = RESIZE_WIDTH_COARSE / ortho_w
    ortho_small = cv2.resize(ortho_img, (RESIZE_WIDTH_COARSE, int(ortho_h * scale_o)), interpolation=cv2.INTER_AREA)
    scale_d = RESIZE_WIDTH_COARSE / detail_w
    detail_small = cv2.resize(detail_img, (RESIZE_WIDTH_COARSE, int(detail_h * scale_d)), interpolation=cv2.INTER_AREA)
    sift_coarse = cv2.SIFT_create(nfeatures=SIFT_NFEATURES_COARSE)
    kp_o_s, des_o_s = sift_coarse.detectAndCompute(ortho_small, None)
    kp_d_s, des_d_s = sift_coarse.detectAndCompute(detail_small, None)
    if des_o_s is None or des_d_s is None: raise ValueError("[Coarse] 特徴量計算失敗")
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_d_s, des_o_s, k=2)
    good_matches = [m for m_n in matches if len(m_n) == 2 and (m := m_n[0]) and (n := m_n[1]) and m.distance < 0.75 * n.distance]
    if len(good_matches) < 10: raise ValueError("[Coarse] 初期対応点が見つかりません。")
    src_pts = np.float32([kp_d_s[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) / scale_d
    dst_pts = np.float32([kp_o_s[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) / scale_o
    M_coarse, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M_coarse is None: raise ValueError("[Coarse] 初期ホモグラフィの計算に失敗。")
    src_corners = np.float32([[0, 0], [detail_w, 0], [detail_w, detail_h], [0, detail_h]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(src_corners, M_coarse).squeeze(axis=1)
    return {"warped_corners": warped_corners}


class ImageProcessor(QObject):
    status_updated = Signal(str)
    processing_finished = Signal(dict)
    processing_error = Signal(str)
    composition_saved = Signal(str)
    def __init__(self): super().__init__()

    @Slot(str, str)
    def run_processing(self, ortho_path, detail_path):
        try:
            self._update_status("画像を読み込んでいます...")
            ortho_img = cv2.imread(ortho_path)
            detail_img = cv2.imread(detail_path)
            if ortho_img is None or detail_img is None: raise FileNotFoundError("画像ファイルの読み込みに失敗しました。")
            data = self._perform_fine_alignment(ortho_img, detail_img)
            polygons = self._generate_extension_polygons(data['hull_points'], data['warped_corners'])
            data['extension_polygons'] = polygons
            data['ortho_image'] = ortho_img
            data['detail_image'] = detail_img
            self.processing_finished.emit(data)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.processing_error.emit(f"高精度計算エラー: {e}")

    def _update_status(self, message):
        self.status_updated.emit(message)
        print(message)

    def _perform_coarse_alignment(self, ortho_img, detail_img):
        result = perform_coarse_alignment_headless(ortho_img, detail_img)
        ortho_h, ortho_w = ortho_img.shape[:2]
        detail_h, detail_w = detail_img.shape[:2]
        scale_o = RESIZE_WIDTH_COARSE / ortho_w
        ortho_small = cv2.resize(ortho_img, (RESIZE_WIDTH_COARSE, int(ortho_h * scale_o)), interpolation=cv2.INTER_AREA)
        scale_d = RESIZE_WIDTH_COARSE / detail_w
        detail_small = cv2.resize(detail_img, (RESIZE_WIDTH_COARSE, int(detail_h * scale_d)), interpolation=cv2.INTER_AREA)
        sift_coarse = cv2.SIFT_create(nfeatures=SIFT_NFEATURES_COARSE)
        kp_o_s, des_o_s = sift_coarse.detectAndCompute(ortho_small, None)
        kp_d_s, des_d_s = sift_coarse.detectAndCompute(detail_small, None)
        if des_o_s is None or des_d_s is None: raise ValueError("[Coarse] 特徴量計算失敗")
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des_d_s, des_o_s, k=2)
        good_matches = [m for m_n in matches if len(m_n) == 2 and (m := m_n[0]) and (n := m_n[1]) and m.distance < 0.75 * n.distance]
        if len(good_matches) < 10: raise ValueError("[Coarse] 初期対応点が見つかりません。")
        src_pts = np.float32([kp_d_s[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) / scale_d
        dst_pts = np.float32([kp_o_s[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) / scale_o
        M_coarse, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M_coarse is None: raise ValueError("[Coarse] 初期ホモグラフィの計算に失敗。")
        src_corners = np.float32([[0, 0], [detail_w, 0], [detail_w, detail_h], [0, detail_h]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(src_corners, M_coarse).squeeze(axis=1)
        return {"warped_corners": warped_corners, "M_coarse": M_coarse}

    def _perform_fine_alignment(self, ortho_img, detail_img):
        ortho_h, ortho_w = ortho_img.shape[:2]
        detail_h, detail_w = detail_img.shape[:2]
        self._update_status("[Coarse] 荒い位置合わせを再計算中...")
        coarse_data = self._perform_coarse_alignment(ortho_img, detail_img)
        M_coarse = coarse_data["M_coarse"]
        self._update_status("[Fine] 関心領域を特定中...")
        coarse_corners = coarse_data["warped_corners"]
        x, y, w, h = cv2.boundingRect(coarse_corners.astype(np.int32))
        ortho_roi = ortho_img[max(0, y - HULL_MARGIN_PIXELS): min(ortho_h, y + h + HULL_MARGIN_PIXELS), max(0, x - HULL_MARGIN_PIXELS): min(ortho_w, x + w + HULL_MARGIN_PIXELS)]
        kp_ortho_tuples_roi = self._parallel_feature_detection(ortho_roi, "オルソ画像ROI")
        kp_detail_tuples = self._parallel_feature_detection(detail_img, "詳細画像(フル)")
        kp_ortho_tuples = [(t[0] + max(0, x - HULL_MARGIN_PIXELS), t[1] + max(0, y - HULL_MARGIN_PIXELS), *t[2:]) for t in kp_ortho_tuples_roi]
        kp_ortho = [cv2.KeyPoint(x=t[0], y=t[1], size=t[3], angle=t[4], response=t[2], octave=t[5], class_id=t[6]) for t in kp_ortho_tuples]
        kp_detail = [cv2.KeyPoint(x=t[0], y=t[1], size=t[3], angle=t[4], response=t[2], octave=t[5], class_id=t[6]) for t in kp_detail_tuples]
        self._update_status("[Fine] SIFTディスクリプタを計算し、最終マッチングを実行中...")
        sift_fine = cv2.SIFT_create()
        _, des_ortho = sift_fine.compute(ortho_img, kp_ortho)
        _, des_detail = sift_fine.compute(detail_img, kp_detail)
        if des_detail is None or des_ortho is None: raise ValueError("[Fine] 特徴量計算失敗")
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des_detail, des_ortho, k=2)
        good_matches = [m for m_n in matches if len(m_n) == 2 and (m := m_n[0]) and (n := m_n[1]) and m.distance < RATIO_TEST_THRESHOLD * n.distance]
        if len(good_matches) < 10: raise ValueError("[Fine] 対応点が少なすぎます。")
        src_pts = np.float32([kp_detail[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ortho[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M_fine, mask_fine = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
        if M_fine is None: raise ValueError("[Fine] 最終ホモグラフィ計算失敗")
        self._update_status(f"最終的に{np.sum(mask_fine)}個の信頼できる対応点を発見。")
        inlier_dst_pts = dst_pts[mask_fine.ravel() == 1]
        if len(inlier_dst_pts) < 3: raise ValueError("凸包計算のための点が不足しています。")
        hull_points = cv2.convexHull(inlier_dst_pts, returnPoints=True).squeeze(axis=1)
        src_corners = np.float32([[0, 0], [detail_w, 0], [detail_w, detail_h], [0, detail_h]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(src_corners, M_fine).squeeze(axis=1)
        return {"homography_matrix": M_fine, "hull_points": hull_points, "warped_corners": warped_corners}

    def _parallel_feature_detection(self, image, image_name="画像"):
        h, w = image.shape[:2]
        tasks = []
        for y in range(0, h, TILE_SIZE):
            for x in range(0, w, TILE_SIZE):
                tile = image[max(0, y - TILE_OVERLAP):min(h, y + TILE_SIZE + TILE_OVERLAP), max(0, x - TILE_OVERLAP):min(w, x + TILE_SIZE + TILE_OVERLAP)]
                tasks.append((tile, max(0, x - TILE_OVERLAP), max(0, y - TILE_OVERLAP)))
        num_cores = max(1, cpu_count() - 1)
        self._update_status(f"[{image_name}] {len(tasks)}個のタイルを{num_cores}コアで並列検出中...")
        with Pool(processes=num_cores) as pool: results = pool.map(detect_features_in_tile, tasks)
        all_kp_tuples = [kp for tile_kps in results for kp in tile_kps]
        self._update_status(f"[{image_name}] 候補点 {len(all_kp_tuples)}個から重複を排除中...")
        all_kp_tuples.sort(key=lambda item: item[2], reverse=True)
        grid, unique_kp_tuples = {}, []
        for kp_tuple in all_kp_tuples:
            grid_key = (int(kp_tuple[0] / DEDUPLICATION_GRID_SIZE), int(kp_tuple[1] / DEDUPLICATION_GRID_SIZE))
            if grid_key not in grid:
                grid[grid_key] = True
                unique_kp_tuples.append(kp_tuple)
        if len(unique_kp_tuples) > MAX_FEATURES_FOR_MATCHING:
            self._update_status(f"[{image_name}] {len(unique_kp_tuples)}個から品質上位{MAX_FEATURES_FOR_MATCHING}個を選抜。")
            unique_kp_tuples = unique_kp_tuples[:MAX_FEATURES_FOR_MATCHING]
        else: self._update_status(f"[{image_name}] {len(unique_kp_tuples)}個のユニークな特徴点を検出。")
        return unique_kp_tuples

    def _get_line_intersection(self, p1, p2, p3, p4):
        p1, p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)
        d = (p2[0] - p1[0]) * (p4[1] - p3[1]) - (p2[1] - p1[1]) * (p4[0] - p3[0])
        if abs(d) < 1e-6: return None
        t = ((p3[0] - p1[0]) * (p4[1] - p3[1]) - (p3[1] - p1[1]) * (p4[0] - p3[0])) / d
        u = -((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])) / d
        if t >= 0 and 0 <= u <= 1: return p1 + t * (p2 - p1)
        return None

    def _get_bisector_intersection(self, p_curr, p_prev, p_next, warped_border_lines):
        v1, v2 = p_prev - p_curr, p_next - p_curr
        v1_norm, v2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
        if v1_norm == 0 or v2_norm == 0: return None
        bisector_vec = -((v1 / v1_norm) + (v2 / v2_norm))
        if np.linalg.norm(bisector_vec) < 1e-6: return None
        p_far = p_curr + bisector_vec * 1e6
        for i, (line_start, line_end) in enumerate(warped_border_lines):
            intersect_pt = self._get_line_intersection(p_curr, p_far, line_start, line_end)
            if intersect_pt is not None: return (QPointF(*intersect_pt), i)
        return None

    def _generate_extension_polygons(self, hull_points, warped_corners):
        self._update_status("輪郭を考慮した拡張領域を生成しています...")
        polygons = []
        if hull_points.ndim == 1 or len(hull_points) < 2: return polygons
        warped_border_lines = [(warped_corners[i], warped_corners[(i + 1) % 4]) for i in range(4)]
        for i in range(len(hull_points)):
            h1, h2 = hull_points[i], hull_points[(i + 1) % len(hull_points)]
            res1 = self._get_bisector_intersection(h1, hull_points[i - 1], h2, warped_border_lines)
            if not res1: continue
            i1, edge_idx1 = res1
            res2 = self._get_bisector_intersection(h2, h1, hull_points[(i + 2) % len(hull_points)], warped_border_lines)
            if not res2: continue
            i2, edge_idx2 = res2
            path_nodes = [i1]
            if edge_idx1 != edge_idx2:
                curr_idx = (edge_idx1 + 1) % 4
                while curr_idx != edge_idx2:
                    path_nodes.append(QPointF(*warped_corners[curr_idx]))
                    curr_idx = (curr_idx + 1) % 4
                path_nodes.append(QPointF(*warped_corners[edge_idx2]))
            path_nodes.append(i2)
            unique_path_nodes = [p for j, p in enumerate(path_nodes) if j == 0 or path_nodes[j - 1].toPoint() != p.toPoint()]
            polygons.append(QPolygonF([QPointF(*h2), QPointF(*h1)] + unique_path_nodes))
        return polygons

    @Slot(dict, list)
    def create_and_save_final_image(self, data, selected_polygons):
        self._update_status("最終画像を生成・保存しています...")
        ortho_img, detail_img, M = data["ortho_image"], data["detail_image"], data["homography_matrix"]
        ortho_h, ortho_w = ortho_img.shape[:2]
        detail_img_bgra = cv2.cvtColor(detail_img, cv2.COLOR_BGR2BGRA) if detail_img.shape[2] == 3 else detail_img.copy()
        warped_detail_bgra = cv2.warpPerspective(detail_img_bgra, M, (ortho_w, ortho_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        final_mask = np.zeros((ortho_h, ortho_w), dtype=np.uint8)
        for poly in selected_polygons:
            pts = np.array([[p.x(), p.y()] for p in poly.toList()], dtype=np.int32)
            cv2.fillPoly(final_mask, [pts], 255)
        warped_detail_bgra[:, :, 3] = cv2.bitwise_and(warped_detail_bgra[:, :, 3], final_mask)
        if not os.path.exists("outputs"): os.makedirs("outputs")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        png_path = os.path.join("outputs", f"blended_{timestamp}.png")
        cv2.imwrite(png_path, warped_detail_bgra)
        self._update_status(f"処理完了！ ファイルを保存しました: {png_path}")
        self.composition_saved.emit(png_path)


class UpdateWorker(QObject):
    finished = Signal(str, str, np.ndarray)
    error = Signal(str)
    @Slot(str, str)
    def run(self, current_ortho_path, overlay_png_path):
        try:
            base_img = cv2.imread(current_ortho_path)
            overlay_img = cv2.imread(overlay_png_path, cv2.IMREAD_UNCHANGED)
            if base_img is None: raise FileNotFoundError(f"ベース画像の読み込みに失敗: {current_ortho_path}")
            if overlay_img is None: raise FileNotFoundError(f"オーバーレイ画像の読み込みに失敗: {overlay_png_path}")
            alpha = overlay_img[:, :, 3] / 255.0
            for c in range(0, 3): base_img[:, :, c] = (overlay_img[:, :, c] * alpha + base_img[:, :, c] * (1 - alpha))
            temp_handle, temp_path = tempfile.mkstemp(suffix=".png")
            os.close(temp_handle)
            cv2.imwrite(temp_path, base_img)
            self.finished.emit(temp_path, overlay_png_path, base_img)
        except Exception as e: self.error.emit(str(e))


class PhotoViewer(QGraphicsView):
    region_selected = Signal(QRectF)
    MODE_SELECTION, MODE_POLYGON_TOGGLE = 1, 2
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self._is_selecting, self._is_panning = False, False
        self._selection_start_pos, self._pan_start_pos = QPointF(), QPoint()
        self.selection_rect_item = QGraphicsRectItem()
        self.selection_rect_item.setPen(QPen(QColor(0, 120, 215, 200), 2, Qt.DashLine))
        self.selection_rect_item.setBrush(QBrush(QColor(0, 120, 215, 60)))
        self.selection_rect_item.setZValue(10000)
        self.current_mode = self.MODE_SELECTION
    def set_mode(self, mode): self.current_mode = mode
    def wheelEvent(self, event): self.scale(1.25 if event.angleDelta().y() > 0 else 0.8, 1.25 if event.angleDelta().y() > 0 else 0.8)
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._is_panning = True
            self._pan_start_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor); return
        if self.current_mode == self.MODE_SELECTION and event.button() == Qt.LeftButton:
            self._is_selecting = True
            if self.selection_rect_item.scene() is None: self.scene().addItem(self.selection_rect_item)
            self._selection_start_pos = self.mapToScene(event.position().toPoint())
            self.selection_rect_item.setRect(QRectF(self._selection_start_pos, QSize(0, 0)))
            self.selection_rect_item.show()
        else: super().mousePressEvent(event)
    def mouseMoveEvent(self, event):
        if self._is_panning:
            delta = event.position().toPoint() - self._pan_start_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan_start_pos = event.position().toPoint()
        elif self._is_selecting: self.selection_rect_item.setRect(QRectF(self._selection_start_pos, self.mapToScene(event.position().toPoint())).normalized())
        else: super().mouseMoveEvent(event)
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
        elif event.button() == Qt.LeftButton and self._is_selecting:
            self._is_selecting = False
            if self.selection_rect_item.rect().width() > 5: self.region_selected.emit(self.selection_rect_item.rect())
        else: super().mouseReleaseEvent(event)
    def clear_selection(self): self.selection_rect_item.hide()


class CandidateDialog(QDialog):
    def __init__(self, candidates, parent=None):
        super().__init__(parent)
        self.setWindowTitle("候補画像の選択")
        self.selected_path = None
        layout, self.list_widget = QVBoxLayout(self), QListWidget()
        self.list_widget.setIconSize(QSize(150, 150))
        self.list_widget.itemDoubleClicked.connect(self.accept)
        for path, thumb in candidates:
            item = QListWidgetItem(os.path.basename(path))
            item.setIcon(QIcon(thumb))
            item.setData(Qt.UserRole, path)
            self.list_widget.addItem(item)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(self.list_widget)
        layout.addWidget(buttons)
    def accept(self):
        if self.list_widget.selectedItems():
            self.selected_path = self.list_widget.selectedItems()[0].data(Qt.UserRole)
            super().accept()


class MainWindow(QMainWindow):
    start_processing_signal = Signal(str, str)
    start_update_signal = Signal(str, str)

    def __init__(self, ortho_path, detail_dir, preloaded_cache):
        super().__init__()
        self.setWindowTitle("Photogrammetry_Ortho_Image_Refiner")
        self.setGeometry(100, 100, 1400, 900)
        self.initial_ortho_path, self.detail_dir = ortho_path, detail_dir
        self.candidate_cache = preloaded_cache
        self.generated_pngs, self.data = [], None
        self.selectable_items, self.selected_states = [], []
        self.current_ortho_path = self.initial_ortho_path
        self.base_pixmap_item, self.temp_files = None, []
        self.status_label = QLabel("初期化中...")
        self.scene = QGraphicsScene()
        self.view = PhotoViewer(self.scene)
        self.find_button = QPushButton("② この領域で候補を検索")
        self.confirm_button = QPushButton("✅ 選択を確定して合成")
        self.export_button = QPushButton("📦 PSDとしてエクスポート")
        central_widget, main_layout = QWidget(), QVBoxLayout()
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.find_button)
        button_layout.addWidget(self.confirm_button)
        button_layout.addStretch()
        button_layout.addWidget(self.export_button)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.view)
        main_layout.addLayout(button_layout)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.proc_thread = QThread()
        self.processor = ImageProcessor()
        self.processor.moveToThread(self.proc_thread)
        self.start_processing_signal.connect(self.processor.run_processing)
        self.processor.processing_finished.connect(self.on_processing_finished)
        self.processor.composition_saved.connect(self.on_composition_saved)
        self.processor.processing_error.connect(self.on_processing_error)
        self.processor.status_updated.connect(self.status_label.setText)
        self.update_thread = QThread()
        self.update_worker = UpdateWorker()
        self.update_worker.moveToThread(self.update_thread)
        self.start_update_signal.connect(self.update_worker.run)
        self.update_worker.finished.connect(self.on_update_finished)
        self.update_worker.error.connect(self.on_processing_error)
        self.view.region_selected.connect(self.on_region_selected)
        self.find_button.clicked.connect(self.on_find_button_clicked)
        self.confirm_button.clicked.connect(self.save_result)
        self.export_button.clicked.connect(self.export_project)
        self.proc_thread.start()
        self.update_thread.start()
        self.setup_scene()
        self.reset_ui_state()

    def setup_scene(self):
        ortho_cv_img = cv2.imread(self.current_ortho_path)
        if ortho_cv_img is None: self.on_processing_error("OpenCVでのオルソ画像の読み込みに失敗しました。"); return
        pixmap = self._cv_to_qpixmap(ortho_cv_img)
        if pixmap.isNull(): self.on_processing_error("オルソ画像の表示に失敗"); return
        self.base_pixmap_item = self.scene.addPixmap(pixmap)
        rect = pixmap.rect()
        margins = QMargins(int(rect.width() * 0.5), int(rect.height() * 0.5), int(rect.width() * 0.5), int(rect.height() * 0.5))
        self.scene.setSceneRect(rect.marginsAdded(margins))
        self.view.fitInView(rect, Qt.KeepAspectRatio)

    def reset_ui_state(self):
        self.find_button.setEnabled(False)
        self.confirm_button.setEnabled(False)
        self.export_button.setEnabled(len(self.generated_pngs) > 0)
        self.view.set_mode(self.view.MODE_SELECTION)
        for item in self.selectable_items: self.scene.removeItem(item)
        self.selectable_items.clear()
        self.selected_states.clear()
        self.data = None
        self.status_label.setText("① 補完したい領域を左ドラッグで選択してください。")

    def on_region_selected(self, rect):
        self.reset_ui_state()
        self.view.clear_selection()
        self.view.selection_rect_item.setRect(rect)
        self.view.selection_rect_item.show()
        self.current_selection_rect = rect
        self.find_button.setEnabled(True)
        self.status_label.setText("領域を選択しました。「② この領域で候補を検索」ボタンを押してください。")

    def on_find_button_clicked(self):
        self.view.clear_selection()
        candidates = self.find_candidate_images(QPolygonF(self.current_selection_rect))
        if not candidates:
            QMessageBox.information(self, "情報", "選択領域にマッチする画像が見つかりませんでした。"); self.reset_ui_state(); return
        dialog = CandidateDialog(candidates, self)
        if dialog.exec():
            self.find_button.setEnabled(False)
            self.status_label.setText("高精度計算をバックグラウンドで開始します...")
            self.start_processing_signal.emit(self.current_ortho_path, dialog.selected_path)
        else: self.reset_ui_state()

    def on_processing_finished(self, data):
        self.data = data
        self.view.set_mode(self.view.MODE_POLYGON_TOGGLE)
        warped_pixmap = self._cv_to_qpixmap(cv2.warpPerspective(data["detail_image"], data["homography_matrix"], (data["ortho_image"].shape[1], data["ortho_image"].shape[0])))
        preview_item = self.scene.addPixmap(warped_pixmap)
        preview_item.setOpacity(0.5)
        self.selectable_items.append(preview_item)
        hull_polygon = QPolygonF([QPointF(*p) for p in data["hull_points"]])
        hull_item = QGraphicsPolygonItem(hull_polygon)
        hull_item.setBrush(QBrush(QColor(0, 255, 0, 100)))
        hull_item.setPen(QPen(Qt.green, 2))
        self.scene.addItem(hull_item)
        self.selectable_items.append(hull_item)
        colors = [QColor(255, 0, 0, 80), QColor(0, 0, 255, 80), QColor(255, 255, 0, 80), QColor(255, 0, 255, 80)]
        for i, poly in enumerate(data['extension_polygons']):
            item = QGraphicsPolygonItem(poly)
            item.setBrush(QBrush(colors[i % len(colors)]))
            item.setPen(QPen(Qt.white, 1, Qt.DashLine))
            item.setAcceptHoverEvents(True)
            item.mousePressEvent = lambda event, idx=i: self.toggle_selection(idx)
            self.scene.addItem(item)
            self.selectable_items.append(item)
            self.selected_states.append(False)
        self.confirm_button.setEnabled(True)
        self.status_label.setText("③ 合成に含めたい拡張領域をクリックし、「選択を確定して合成」を押してください。")

    def toggle_selection(self, index):
        self.selected_states[index] = not self.selected_states[index]
        item = self.selectable_items[index + 2]
        color = item.brush().color()
        if self.selected_states[index]:
            color.setAlpha(180)
            item.setPen(QPen(Qt.yellow, 3, Qt.SolidLine))
        else:
            color.setAlpha(80)
            item.setPen(QPen(Qt.white, 1, Qt.DashLine))
        item.setBrush(QBrush(color))

    def save_result(self):
        final_polygons = [QPolygonF([QPointF(*p) for p in self.data["hull_points"]])]
        final_polygons.extend(self.data['extension_polygons'][i] for i, selected in enumerate(self.selected_states) if selected)
        self.confirm_button.setEnabled(False)
        self.processor.create_and_save_final_image(self.data, final_polygons)

    def on_composition_saved(self, png_path):
        self.status_label.setText("オルソ画像をバックグラウンドで更新しています...")
        self.start_update_signal.emit(self.current_ortho_path, png_path)

    def on_update_finished(self, new_ortho_path, original_png_path, blended_image):
        self.temp_files.append(new_ortho_path)
        self.current_ortho_path = new_ortho_path
        new_pixmap = self._cv_to_qpixmap(blended_image)
        if new_pixmap.isNull(): self.on_processing_error("更新されたオルソ画像の表示に失敗しました。"); return
        self.base_pixmap_item.setPixmap(new_pixmap)
        self.generated_pngs.append(original_png_path)
        self.reset_ui_state()
        print("オルソ画像の更新が完了しました。")

    def on_processing_error(self, message):
        QMessageBox.critical(self, "エラー", message)
        self.reset_ui_state()

    def find_candidate_images(self, selection_polygon):
        candidates_with_thumbs = []
        for candidate in self.candidate_cache:
            if candidate["polygon"].intersects(selection_polygon):
                path = candidate["path"]
                candidates_with_thumbs.append((path, QPixmap(path).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        return candidates_with_thumbs

    def _cv_to_qpixmap(self, cv_img):
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        q_format = QImage.Format_BGR888 if ch == 3 else QImage.Format_BGRA8888
        return QPixmap.fromImage(QImage(cv_img.data, w, h, bytes_per_line, q_format))

    def export_project(self):
        if not self.generated_pngs:
            QMessageBox.warning(self, "警告", "エクスポートするレイヤーがありません。"); return
        try:
            path = export_to_psd(self.initial_ortho_path, self.generated_pngs, "outputs")
            QMessageBox.information(self, "成功", f"PSDファイルを保存しました:\n{path}")
        except Exception as e: self.on_processing_error(f"PSDエクスポート失敗: {e}")
        finally: self.reset_ui_state()

    def closeEvent(self, event):
        self.proc_thread.quit()
        self.update_thread.quit()
        self.proc_thread.wait()
        self.update_thread.wait()
        for f in self.temp_files:
            try: os.remove(f)
            except OSError as e: print(f"一時ファイルの削除に失敗: {e}")
        event.accept()

def run_precomputation_coarse(ortho_path, detail_dir):
    print("--- 事前計算(荒いマッチング)を開始します ---")
    print(f"オルソ画像: {ortho_path}")
    print(f"詳細画像ディレクトリ: {detail_dir}")
    detail_paths = glob.glob(os.path.join(detail_dir, "*.jpg")) + glob.glob(os.path.join(detail_dir, "*.png"))
    if not detail_paths:
        print("警告: 指定されたディレクトリに詳細画像が見つかりませんでした。"); return
    try: ortho_hash = _get_image_hash(ortho_path)
    except FileNotFoundError: print(f"!!! エラー: オルソ画像が見つかりません: {ortho_path}"); return
    existing_detail_hashes = set()
    if os.path.exists(CACHE_DIR):
        prefix = f"{ortho_hash}_"
        for filename in os.listdir(CACHE_DIR):
            if filename.startswith(prefix) and filename.endswith(".pkl"): existing_detail_hashes.add(filename[len(prefix):-4])
    ortho_img = cv2.imread(ortho_path)
    if ortho_img is None: print(f"!!! エラー: オルソ画像が読み込めません: {ortho_path}"); return
    newly_cached_count = 0
    for i, detail_path in enumerate(tqdm(detail_paths, desc="事前計算")):
        try:
            detail_hash = _get_image_hash(detail_path)
            if detail_hash in existing_detail_hashes: continue
            detail_img = cv2.imread(detail_path)
            if detail_img is None: raise FileNotFoundError("詳細画像ファイルの読み込みに失敗しました。")
            coarse_alignment_data = perform_coarse_alignment_headless(ortho_img, detail_img)
            cache_data = {"warped_corners": coarse_alignment_data["warped_corners"]}
            save_cache(ortho_path, detail_path, cache_data)
            existing_detail_hashes.add(detail_hash)
            newly_cached_count += 1
        except Exception as e:
            # tqdmを使っている場合、エラーメッセージがバーで上書きされないようにする
            tqdm.write(f"\n!!! エラーが発生しました: {os.path.basename(detail_path)} - {e}")
    print(f"\n--- 事前計算が完了しました ({newly_cached_count}件のキャッシュを新規作成) ---")

# --- ▼▼▼【修正】並列処理でキャッシュを高速に読み込む関数 ▼▼▼ ---
def _load_single_cache_worker(detail_path, ortho_path):
    """
    単一のキャッシュファイルを読み込むためのワーカー関数。
    並列処理の各スレッドで実行される。
    """
    cache = load_cache(ortho_path, detail_path)
    if cache and 'warped_corners' in cache:
        return {"path": detail_path, "corners_np": cache['warped_corners']}
    return None

def preload_candidate_data_parallel(ortho_path, detail_dir):
    """
    ThreadPoolExecutorを使用して、キャッシュ読み込みを並列化する。
    """
    print("--- キャッシュの並列読み込みを開始します ---")
    detail_paths = glob.glob(os.path.join(detail_dir, "*.jpg")) + glob.glob(os.path.join(detail_dir, "*.png"))
    preloaded_cache = []
    
    # partialを使って、ortho_pathをワーカー関数に固定引数として渡す
    worker = partial(_load_single_cache_worker, ortho_path=ortho_path)
    
    # I/Oバウンドなタスクなので、スレッド数を多めに設定しても良い
    # os.cpu_count() * 5 は一般的な推奨値だが、環境に応じて調整可能
    max_workers = min(32, (os.cpu_count() or 1) * 5)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.mapで全タスクを投入し、tqdmで進捗を表示
        results = list(tqdm(executor.map(worker, detail_paths), total=len(detail_paths), desc="キャッシュ読み込み"))

    # Noneではない結果（正常に読み込めたキャッシュ）だけをリストに追加
    preloaded_cache = [res for res in results if res is not None]
            
    print(f"--- {len(preloaded_cache)} 件のキャッシュ読み込みが完了しました ---")
    return preloaded_cache
# --- ▲▲▲ 修正はここまで ▲▲▲ ---

if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser(description='Photogrammetry_Ortho_Image_Refiner')
    parser.add_argument('ortho_image', help='基準となるオルソ画像のパス')
    parser.add_argument('detail_images_dir', help='高解像度詳細画像が含まれるディレクトリ')
    args = parser.parse_args()

    # --- ワークフロー ①: コマンドラインで事前計算を実行 ---
    run_precomputation_coarse(args.ortho_image, args.detail_images_dir)

    # --- ワークフロー ②: コマンドラインでキャッシュを並列で高速にメモリへロード ---
    preloaded_cache_np = preload_candidate_data_parallel(args.ortho_image, args.detail_images_dir)

    # --- ワークフロー ③: 全ての準備完了後、GUIを起動 ---
    print("GUIを起動します...")
    app = QApplication(sys.argv)
    
    # Qtオブジェクト(QPolygonF)はQApplicationインスタンスの作成後に生成する必要がある
    preloaded_cache_qt = []
    for data in preloaded_cache_np:
        polygon = QPolygonF([QPointF(p[0], p[1]) for p in data['corners_np']])
        preloaded_cache_qt.append({
            "path": data["path"],
            "polygon": polygon
        })

    window = MainWindow(args.ortho_image, args.detail_images_dir, preloaded_cache_qt)
    window.show()
    sys.exit(app.exec())
