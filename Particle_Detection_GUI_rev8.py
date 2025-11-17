import os
import sys
import io
from pathlib import Path
import time
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QGroupBox, QSizePolicy, QLineEdit, QLayout, QCheckBox, QTextEdit, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, QSettings, QRect, QThread, QObject, Signal, Slot, QDateTime, QEvent, QUrl
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QGuiApplication, QDesktopServices
import csv
from collections import deque
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 한글 폰트 설정 (windows 한글깨짐 방지)
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False

# 환경 플래그: 경로 및 저장동작 테스트(True) / 운영(False) 모드 구분
LOAD_TEST_MODE = False
SAVE_TEST_MODE = False
INI_TEST_MODE  = False

# 실행파일 실행 시, "자동 시작 동작 수행" 여부 플래그 : 자동 시작(True) / 수동 시작(False)
AUTO_START_ON_LAUNCH = True

# 이미지/CSV/그래프/이벤트 파일 경로 설정
if LOAD_TEST_MODE:
    IMG_INPUT_DIR = Path('./Images')
else:
    IMG_INPUT_DIR = Path(r'D:/AI Vision/Images')

if SAVE_TEST_MODE:
    CSV_OUTPUT_DIR = Path('./particle_info')
else:
    CSV_OUTPUT_DIR = Path(r'D:/AI Vision/particle_info')

if SAVE_TEST_MODE:
    EVENT_OUTPUT_DIR = CSV_OUTPUT_DIR / 'event_list'
else:
    EVENT_OUTPUT_DIR = Path(r'D:/AI Vision/particle_info/event_list')

if SAVE_TEST_MODE:
    GRAPH_OUTPUT_DIR = CSV_OUTPUT_DIR / 'graph_list'
else:
    GRAPH_OUTPUT_DIR = Path(r'D:/AI Vision/particle_info/graph_list')

if INI_TEST_MODE:
    try:
        ini_base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        ini_base_dir = Path(os.getcwd())
    INI_SETTINGS_PATH = ini_base_dir / "ParticleDetectionApp.ini"
else:
    INI_SETTINGS_PATH = Path(r'D:/AI Vision/ParticleDetectionApp.ini')


# 폴더 미리 생성
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
os.makedirs(EVENT_OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)

# 파티클 탐지 파라미터
WARMUP_FRAMES = 15             # 워밍업 프레임 수
MIN_NOISE_REPEAT = 2           # 워밍업 구간 내 몇 회 이상 등장 좌표를 노이즈로 설정
MANUAL_THRESHOLD   = 7         # 이진화 임계값
KERNEL_SIZE_TOPHAT = (3, 3)    # tophat 모폴로지 연산 커널 크기
KERNEL_SIZE_MORPH  = (3, 3)    # 모폴로지 클로즈 연산 커널 크기
PARTICLE_AREA_MIN  = 8         # 파티클 최소 면적
PARTICLE_AREA_MAX  = 50        # 파티클 최대 면적
WARMUP_NOISE_AREA_MIN = 2      # 워밍업 노이즈 판별 전용 최소 면적

# 파티클 표기 원 파라미터 (유효/노이즈)
VALID_CIRCLE_RADIUS    = 25    # 유효(빨강) 원 반지름
VALID_CIRCLE_THICKNESS = 2     # 유효(빨강) 원 테두리 두께
NOISE_CIRCLE_RADIUS    = 10    # 노이즈(흰색) 원 반지름
NOISE_CIRCLE_THICKNESS = 1     # 노이즈(흰색) 원 테두리 두께

# 밝기 참조 영역 (ROI)
REF_ROI = (70, 90, 20, 20)

# 앵커 밝기 기반 동적 임계값 파라미터
ANCHOR_BRIGHTNESS_REF = 48          # 기준 앵커 밝기
ADAPTIVE_THRESHOLD_GAIN = 0.2       # 적응 임계 보정계수

# 메인 윈도우 기본 위치 (모니터 좌측 하단 기준 offset)
WINDOW_OFFSET_X = 100
WINDOW_OFFSET_Y = 200

# 신규 이미지 부재 시 미리보기 플레이스홀더 텍스트
NO_IMAGE_PLACEHOLDER_TEXT = "신규 이미지 없음"

# 감시 대상 이미지 확장자 목록
ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def safe_image_load(image_path, as_gray=False, max_retries=5, delay=0.2):
    """안전 이미지 로딩 (재시도 최대 5회)"""
    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_UNCHANGED
    for _ in range(max_retries):
        try:
            img_array = np.fromfile(str(image_path), dtype=np.uint8)
            img = cv2.imdecode(img_array, flag)
            if img is not None:
                return img
        except Exception:
            pass
        time.sleep(delay)
    return None


def compute_anchor_brightness(gray_img):
    """70% 최빈 구간 앵커 밝기 계산"""
    if gray_img is None or gray_img.ndim != 2:
        return 0
    x, y, w, h = REF_ROI
    h_img, w_img = gray_img.shape[:2]

    x0 = max(0, min(x, w_img-1))
    y0 = max(0, min(y, h_img-1))
    w0 = max(1, min(w, w_img - x0))
    h0 = max(1, min(h, h_img - y0))

    roi = gray_img[y0:y0+h0, x0:x0+w0]
    if roi.size == 0:
        return 0

    hist = cv2.calcHist([roi], [0], None, [256], [0, 256]).flatten()
    total = int(roi.size)
    if total <= 0:
        return 0

    mode_idx = int(np.argmax(hist))
    left = right = mode_idx
    cum = hist[mode_idx]
    target = 0.70 * total

    while cum < target and (left > 0 or right < 255):
        left_next  = hist[left-1]  if left  > 0   else -1
        right_next = hist[right+1] if right < 255 else -1
        if right_next >= left_next and right < 255:
            right += 1
            cum += hist[right]
        elif left > 0:
            left -= 1
            cum += hist[left]
        else:
            break

    bins = np.arange(left, right+1, dtype=np.float32)
    weights = hist[left:right+1]
    denom = weights.sum()
    if denom <= 0:
        return int(mode_idx)
    anchor = int(round(float((bins * weights).sum() / denom)))
    return max(0, min(anchor, 255))


def compute_adaptive_threshold(anchor_current: int | float | None) -> int:
    """앵커 밝기에 따라 동적으로 조정된 이진화 임계값 계산"""
    if anchor_current is None:
        return int(MANUAL_THRESHOLD)

    adaptive_value = MANUAL_THRESHOLD + ADAPTIVE_THRESHOLD_GAIN * (float(anchor_current) - ANCHOR_BRIGHTNESS_REF)
    adaptive_value = max(0.0, min(255.0, adaptive_value))
    return int(round(adaptive_value))


def particle_detection(image_source, exclude_boxes, threshold=None,
                       area_min=PARTICLE_AREA_MIN, area_max=PARTICLE_AREA_MAX):
    """파티클 탐지 (OpenCV 파이프라인)"""
    if isinstance(image_source, np.ndarray):
        img = image_source
    else:
        img = safe_image_load(image_source, as_gray=True)

    if img is None:
        print(f"❌ 오류: {image_source} 이미지 로딩 실패")
        return None, None, None

    try:
        if img.ndim == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE_TOPHAT)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        thr_value = MANUAL_THRESHOLD if threshold is None else int(threshold)
        thr_value = max(0, min(255, thr_value))
        _, binary_mask = cv2.threshold(tophat, thr_value, 255, cv2.THRESH_BINARY)
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_SIZE_MORPH))

        # 제외영역(2개) 마스킹
        mask = np.ones_like(cleaned_mask, dtype=np.uint8) * 255
        for (x, y, w, h) in exclude_boxes:
            if w > 0 and h > 0:
                cv2.rectangle(mask, (x, y), (x+w, y+h), 0, -1)

        final_mask = np.bitwise_and(cleaned_mask, mask)

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        particle_info = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            if area_min <= area <= area_max:
                particle_info.append((cx, cy, area))

        vis_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in exclude_boxes:
            if w > 0 and h > 0:
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 1)

        return gray, vis_img, particle_info

    except Exception as e:
        print(f"❌ 오류: {e}")
        return None, None, None


def find_new_images(img_dir, processed_set):
    """디렉토리 내 신규 이미지 탐색 (파일명 기준)"""
    all_imgs = sorted([p for p in img_dir.glob('*') if p.suffix.lower() in ALLOWED_IMAGE_EXTS])
    return [p for p in all_imgs if p.name not in processed_set]


class _ImageFileEventHandler(FileSystemEventHandler):
    """watchdog 이벤트 핸들러 (신규 이미지 감지)"""

    def __init__(self, callback, allow_ext):
        super().__init__()
        self._callback = callback
        self._allow_ext = tuple(allow_ext)

    def _handle_path(self, src_path):
        if not src_path:
            return
        path = Path(src_path)
        if path.is_dir():
            return
        if path.suffix.lower() not in self._allow_ext:
            return
        self._callback(path)

    def on_created(self, event):
        self._handle_path(getattr(event, 'src_path', None))

    def on_moved(self, event):
        self._handle_path(getattr(event, 'dest_path', None))


class ImageDirectoryWatcher(QObject):
    """watchdog 기반 이미지 디렉토리 감시기"""

    file_created = Signal(object)

    def __init__(self, directory: Path, allow_ext=None, parent=None):
        super().__init__(parent)
        self._directory = Path(directory)
        self._observer: Observer | None = None # pyright: ignore[reportInvalidTypeForm]
        self._handler: _ImageFileEventHandler | None = None
        if allow_ext is None:
            allow_ext = ALLOWED_IMAGE_EXTS
        self._allow_ext = {ext.lower() for ext in allow_ext}

    def start(self):
        if self._observer is not None:
            return
        if not self._directory.exists():
            self._directory.mkdir(parents=True, exist_ok=True)
        self._handler = _ImageFileEventHandler(self.file_created.emit, self._allow_ext)
        observer = Observer()
        observer.schedule(self._handler, str(self._directory), recursive=False)
        observer.start()
        self._observer = observer

    def stop(self):
        if self._observer is None:
            return
        observer = self._observer
        self._observer = None
        self._handler = None
        try:
            observer.stop()
            observer.join(timeout=2.0)
        except Exception:
            pass


def parse_filename(fname: str):
    """파일명 파서 (형식: EQUIP_Lot#_Process_Attempt_YYYYMMDD_HHMM(SS).ext)"""
    stem = Path(fname).stem
    f = stem.split('_')
    return {
        'equip':   f[0],
        'lot':     f[1],
        'process': f[2],
        'attempt': int(f[3]),
        'date':    f[4],
        'time':    f[5],
        'stem':    stem,
        'name':    Path(fname).name,
    }


class ParticlePlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=None, height=None, dpi=96):
        """그래프 생성 및 크기 조정"""
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)
        self.tick_count = 10   # X축 틱 고정 개수
        W_PX, H_PX = 442, 238  # 그래프 크기 조정
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedSize(W_PX, H_PX)
        self.fig.set_dpi(dpi)
        self.fig.set_size_inches(W_PX / dpi, H_PX / dpi)
        self.fig.subplots_adjust(left=0.15, right=0.98, bottom=0.32, top=0.98)  # 그래프 여백 설정

    def update_plot(self, x_list, y_list, latest_index=None, attempt_list=None):
        """그래프 업데이트 (x: 이미지명, y: 파티클 크기, attempt_list: 각 포인트의 Attempt 정수)"""
        self.ax.cla()

        def shorten(label):
            """그래프 X축 날짜 레이블 단순화 (MM/DD_HH:MM)"""
            info = parse_filename(label)
            return f"{info['date'][4:6]}/{info['date'][6:8]}_{info['time'][0:2]}:{info['time'][2:4]}"

        display_labels = [shorten(str(x)) for x in x_list]
        n = len(display_labels)
        if n == 0:
            self.ax.plot([], [])
            self.draw()
            return

        # X 좌표 및 틱 계산
        if n <= self.tick_count:
            ticks = np.arange(n)
        else:
            ticks = np.linspace(0, n - 1, self.tick_count, dtype=int)

        x_coords = np.arange(n)

        # y 길이 보정
        if len(y_list) < n:
            y_list = np.pad(y_list, (0, n - len(y_list)), mode='constant', constant_values=0)

        # 본선 플롯
        self.ax.plot(x_coords, y_list, linestyle='-', color='black', linewidth=1.5)

        # 빨간 마커(>0)
        x_markers = x_coords[y_list > 0]
        y_markers = y_list[y_list > 0]
        self.ax.scatter(x_markers, y_markers, marker='o', s=80, facecolors='red', edgecolors='black', linewidths=1.5)

        # 현재 포인트(연녹)
        if latest_index is not None and 0 <= latest_index < n:
            y_val = y_list[latest_index]
            self.ax.scatter(latest_index, y_val, marker='o', s=80, facecolors='lightgreen', edgecolors='black', linewidths=1.5)

        # Attempt 그룹 밴딩/경계선/레이블
        if attempt_list is not None and len(attempt_list) == n:
            groups = []
            start = 0
            for i in range(1, n):
                if attempt_list[i] != attempt_list[i - 1]:
                    groups.append((start, i - 1, attempt_list[i - 1]))
                    start = i
            groups.append((start, n - 1, attempt_list[-1]))

            y_min, y_max = self.ax.get_ylim()
            label_y = y_max - (y_max - y_min) * 0.05

            # 교차 배경 밴딩, 경계선, 그룹 라벨
            for idx, (s, e, att) in enumerate(groups):
                if s > e:
                    continue
                # 배경 밴딩
                span_color = "#D8D8D8" if (idx % 2 == 0) else "#AACFFF"
                self.ax.axvspan(s - 0.5, e + 0.5, facecolor=span_color, alpha=0.5, zorder=0)

                # 경계선 (왼쪽 경계만 표시)
                if s > 0:
                    self.ax.axvline(s - 0.5, color='#444444', linestyle='-', linewidth=0.8, alpha=0.8, zorder=1)

                # att 레이블
                mid = (s + e) / 2.0
                self.ax.text(mid, label_y, f"Att-{int(att)}", ha='center', va='top', fontsize=9, fontweight='bold', color='#334455', zorder=2)

        # X축 눈금/레이블
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels([display_labels[i] for i in ticks], rotation=90, fontsize=9)
        self.ax.tick_params(axis='x', which='both', length=0)
        self.ax.xaxis.labelpad = 10
        self.ax.set_ylabel('파티클 크기', fontsize=12)
        self.ax.tick_params(axis='y', labelsize=9)
        self.ax.grid(False, axis='x')

        # 수직/수평 그리드
        for t in ticks:
            self.ax.axvline(x=t, color='#444444', linestyle='-', linewidth=0.5, zorder=0)
        self.ax.grid(True, axis='y', linestyle='--', linewidth=1.0, alpha=0.6, color='#bbbbbb')

        self.draw()


class ImagePreviewLabel(QLabel):
    """이미지 미리보기 및 crop 박스 표시 (박스 2개 + Noise 정보 텍스트)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.box1 = (0, 0, 0, 0)
        self.box2 = (0, 0, 0, 0)
        self.src_shape = None
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(170, 360)
        self.filename = ''
        self.noise_text = ""
        self.anchor_value = None
        self.noise_points = []
        self.placeholder_text = ""

    def set_noise_text(self, text: str):
        """미리보기 하단에 표시할 Noise 정보 텍스트 설정"""
        self.noise_text = text or ""
        self.update()

    def set_anchor_value(self, value: int | None):
        """미리보기에 표시할 앵커 밝기 값 저장"""
        self.anchor_value = int(value) if value is not None else None
        self.update()

    def set_noise_points(self, points):
        """미리보기에 표기할 Noise 좌표 리스트 저장"""
        self.noise_points = list(points) if points else []
        self.update()

    def show_placeholder(self, text: str):
        """이미지 대신 표시할 플레이스홀더 텍스트 설정"""
        self.pixmap = None
        self.src_shape = None
        self.placeholder_text = text or ""
        self.filename = ''
        self.noise_text = ""
        self.noise_points = []
        self.anchor_value = None
        self.update()

    def clear(self):
        """레이블 초기화 (플레이스홀더/이미지/텍스트 모두 삭제)"""
        super().clear()
        self.pixmap = None
        self.src_shape = None
        self.placeholder_text = ""
        self.filename = ''
        self.noise_text = ""
        self.noise_points = []
        self.anchor_value = None
        self.update()

    def show_image(self, npimg, box=None, filename=None):
        if box is not None:
            if isinstance(box, (list, tuple)) and len(box) == 2 and isinstance(box[0], (list, tuple)):
                self.box1 = tuple(box[0])
                self.box2 = tuple(box[1])
            else:
                self.box1 = tuple(box)

        if filename is not None:
            self.filename = filename
        self.src_shape = npimg.shape[:2]
        h, w = self.src_shape
        self.placeholder_text = ""

        if npimg.ndim == 3:
            rgb_img = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb_img.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            qimg = QImage(npimg.data, w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qimg).scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.pixmap = pixmap
        self.repaint()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        if self.pixmap is not None and self.src_shape is not None:
            x0 = (self.width() - self.pixmap.width()) // 2
            y0 = (self.height() - self.pixmap.height()) // 2
            painter.drawPixmap(x0, y0, self.pixmap)

            img_h, img_w = self.src_shape
            scale = min(self.pixmap.width() / img_w, self.pixmap.height() / img_h)

            # 첫 번째 박스 (box1)
            if self.box1 and self.box1[2] > 0 and self.box1[3] > 0:
                rect_x = int(x0 + self.box1[0] * scale)
                rect_y = int(y0 + self.box1[1] * scale)
                rect_w = int(self.box1[2] * scale)
                rect_h = int(self.box1[3] * scale)
                pen = QPen(QColor(0, 255, 0), 1)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(rect_x, rect_y, rect_w, rect_h)

            # 두 번째 박스 (box2)
            if self.box2 and self.box2[2] > 0 and self.box2[3] > 0:
                rect2_x = int(x0 + self.box2[0] * scale)
                rect2_y = int(y0 + self.box2[1] * scale)
                rect2_w = int(self.box2[2] * scale)
                rect2_h = int(self.box2[3] * scale)
                pen2 = QPen(QColor(0, 255, 0), 1)
                painter.setPen(pen2)
                painter.drawRect(rect2_x, rect2_y, rect2_w, rect2_h)

            # 밝기 참조 ROI 박스 + 밝기 값 표기
            rx, ry, rw, rh = REF_ROI
            rect_ref_x = int(x0 + rx * scale)
            rect_ref_y = int(y0 + ry * scale)
            rect_ref_w = int(rw * scale)
            rect_ref_h = int(rh * scale)
            pen_ref = QPen(QColor(255, 255, 255), 1)
            painter.setPen(pen_ref)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect_ref_x, rect_ref_y, rect_ref_w, rect_ref_h)

            # Noise 좌표 흰색 원 표기
            if self.noise_points:
                painter.setBrush(Qt.NoBrush)
                for point in self.noise_points:
                    if not point or len(point) < 2:
                        continue
                    px, py = int(point[0]), int(point[1])
                    cx = int(x0 + px * scale)
                    cy = int(y0 + py * scale)
                    radius = max(1, int(round(NOISE_CIRCLE_RADIUS * scale)))
                    thickness = max(1, int(round(NOISE_CIRCLE_THICKNESS * scale)))
                    pen_noise = QPen(QColor(255, 255, 255), thickness)
                    painter.setPen(pen_noise)
                    painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

            if self.anchor_value is not None:
                font = painter.font()
                font.setPointSize(8)
                painter.setFont(font)
                painter.setPen(QColor(255, 255, 255))
                text = str(int(self.anchor_value))
                metrics = painter.fontMetrics()
                tw = metrics.horizontalAdvance(text)
                th = metrics.height()
                tx = rect_ref_x + (rect_ref_w - tw) // 2
                ty = rect_ref_y - 2
                painter.drawText(QRect(tx, ty - th, tw, th), Qt.AlignCenter, text)

            # 미리보기 하단에 Noise 정보 텍스트 표시
            text = self.noise_text
            if text:
                metrics = painter.fontMetrics()
                tw = metrics.horizontalAdvance(text)
                th = metrics.height()
                rx0 = x0 + self.pixmap.width() - tw - 10
                ry0 = y0 + self.pixmap.height() - th - 6
                bg_rect = QRect(rx0 - 4, ry0 - 2, tw + 8, th + 6)

                painter.fillRect(bg_rect, QColor(0, 0, 0, 120))
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(bg_rect.adjusted(4, 0, -2, 0), Qt.AlignLeft | Qt.AlignVCenter, text)

        else:
            painter.fillRect(self.rect(), QColor(0, 0, 0))
            if self.placeholder_text:
                font = painter.font()
                font.setPointSize(11)
                painter.setFont(font)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(self.rect().adjusted(8, 8, -8, -8), Qt.AlignCenter | Qt.TextWordWrap, self.placeholder_text)

        painter.end()


class BacklogFeeder(QObject):
    """백로그 전용 워커(QThread) 클래스"""
    next_image = Signal(object)
    finished = Signal()

    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.image_paths = list(image_paths)
        self._stop = False

    def stop(self):
        self._stop = True

    @Slot()
    def run(self):
        for p in self.image_paths:
            if self._stop:
                break
            self.next_image.emit(p)
        self.finished.emit()


class AlertPopup(QWidget):
    """탐지 팝업(AlertPopup) 클래스"""
    closedWithAction = Signal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Particle Alert')
        self.resize(1000, 720)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        # 상단 경고 타이틀
        self.title_label = QLabel('Particle이 탐지 되었습니다 !!')
        f = self.title_label.font()
        f.setPointSize(24)
        f.setBold(True)
        self.title_label.setFont(f)
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # 우측 상단: 팝업 무시 옵션
        self.ignore_cb = QCheckBox()
        self.ignore_min_spin = QSpinBox()
        self.ignore_min_spin.setRange(1, 120)
        self.ignore_min_spin.setValue(5)  # Default 5분
        self.ignore_min_spin.setFixedWidth(60)
        small_font = self.ignore_cb.font()
        small_font.setPointSize(9)
        self.ignore_cb.setFont(small_font)
        self.ignore_min_spin.setFont(small_font)
        self.ignore_cb.setChecked(False)
        ignore_label = QLabel('분간 팝업 무시')
        ignore_label.setFont(small_font)

        # 확인 버튼 (우측 상단 배치)
        self.btn_ok = QPushButton('확인')

        # 탐지 이미지 저장 폴더 Link 버튼 생성
        self.btn_open_folder = QPushButton('Particle 탐지 이미지 저장 폴더 Link')
        self.btn_open_folder.setCursor(Qt.PointingHandCursor)
        self.btn_open_folder.setFlat(True)
        self.btn_open_folder.setStyleSheet(
            'QPushButton { color: #1a73e8; text-decoration: underline; } '
            'QPushButton:hover { color: #0b59d4; }'
        )
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.clicked.connect(self._on_open_folder)

        # 상단 텍스트 영역 구성
        topbar = QHBoxLayout()
        topbar.setContentsMargins(0, 0, 0, 0)
        topbar.addWidget(self.title_label, 1)

        top_right = QVBoxLayout()
        top_right.setContentsMargins(0, 0, 0, 0)
        top_right.setSpacing(6)

        # ── 1줄: 무시 옵션 + 확인 버튼 ─────────────────────────────
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)

        top_row.addWidget(self.ignore_cb)
        top_row.addWidget(self.ignore_min_spin)
        top_row.addWidget(ignore_label)

        top_row.addStretch()

        top_row.addWidget(self.btn_ok)

        # ── 2줄: "저장 폴더로 이동" 버튼 ─────────────────────────────
        top_right.addLayout(top_row)
        top_right.addWidget(self.btn_open_folder, alignment=Qt.AlignRight)

        topbar.addLayout(top_right)

        self.info_label = QLabel('')
        self.time_label = QLabel('')
        self.elapsed_label = QLabel('')
        self.count_label = QLabel('')
        info_font = self.info_label.font()
        info_font.setPointSize(12)
        self.info_label.setFont(info_font)
        self.time_label.setFont(info_font)
        self.elapsed_label.setFont(info_font)
        self.count_label.setFont(info_font)
        for lab in (self.info_label, self.time_label, self.elapsed_label, self.count_label):
            lab.setAlignment(Qt.AlignLeft)
        self.elapsed_label.setText('- [경과 시간] -')

        info_inner = QVBoxLayout()
        info_inner.setContentsMargins(0, 0, 0, 0)
        info_inner.setSpacing(4)
        info_inner.addWidget(self.info_label)
        info_inner.addWidget(self.time_label)
        info_inner.addWidget(self.elapsed_label)
        info_inner.addWidget(self.count_label)

        self.info_group_container = QWidget()
        self.info_group_container.setObjectName('infoGroupBody')
        container_layout = QVBoxLayout(self.info_group_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(10)
        container_layout.addLayout(topbar)
        container_layout.addLayout(info_inner)

        self.info_group = QGroupBox(' 발생 정보 ')
        info_group_layout = QVBoxLayout()
        info_group_layout.setContentsMargins(12, 16, 12, 12)
        info_group_layout.setSpacing(8)
        info_group_layout.addWidget(self.info_group_container)
        self.info_group.setLayout(info_group_layout)

        # 그래프 스냅샷
        self.graph_view = QLabel()
        self.graph_view.setFixedHeight(260)
        self.graph_view.setAlignment(Qt.AlignCenter)
        self.graph_view.setStyleSheet("background-color: white;")
        self.graph_view.installEventFilter(self)
        self._graph_pixmap = QPixmap()

        graph_section = QGroupBox(' 그래프 ')
        graph_layout = QVBoxLayout(graph_section)
        graph_layout.setContentsMargins(12, 12, 12, 12)
        graph_layout.addWidget(self.graph_view)

        # 이미지 영역(가로 스크롤)
        self.img_container = QHBoxLayout()
        self.img_container.setSpacing(4)
        self.img_container.setContentsMargins(6, 6, 6, 6)
        self.img_container.addStretch()
        self.img_widget = QWidget()
        self.img_widget.setLayout(self.img_container)
        self.img_scroll = QScrollArea()
        self.img_scroll.setWidgetResizable(True)
        self.img_scroll.setWidget(self.img_widget)
        self.img_scroll.setFixedHeight(260)
        self.img_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.img_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        image_section = QGroupBox(' 탐지 이미지 ')
        image_layout = QVBoxLayout(image_section)
        image_layout.setContentsMargins(12, 12, 12, 12)

        image_layout.addWidget(self.img_scroll)

        # 로그 영역
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("QTextEdit{font-size:10pt}")
        self.log_edit.setMinimumHeight(260)
        self.log_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        log_section = QGroupBox(' 로그 히스토리 ')
        log_layout = QVBoxLayout(log_section)
        log_layout.setContentsMargins(12, 12, 12, 12)
        log_layout.addWidget(self.log_edit)

        graph_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        graph_log_layout = QHBoxLayout()
        graph_log_layout.setContentsMargins(0, 0, 0, 0)
        graph_log_layout.setSpacing(12)
        graph_log_layout.addWidget(graph_section, 1)
        graph_log_layout.addWidget(log_section, 1)

        # 메인 레이아웃
        main = QVBoxLayout(self)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(12)
        main.addWidget(self.info_group)
        main.addLayout(graph_log_layout)
        main.addWidget(image_section)

        # 시그널 연결
        self.btn_ok.clicked.connect(lambda: self._close_with('ok'))

        # 배경 깜빡임
        self._blink_red = False
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._toggle_blink)

        # 경과 시간
        self._first_dt = None
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)

        # 이미지/이벤트 폴더 관리
        self._pix_labels: list[QWidget] = []
        self._event_folder_path: Path | None = None

    def start_blink(self):
        self._blink_timer.start(500)  # 0.5초 주기
        self._blink_red = False
        self._toggle_blink()

    def stop_blink(self):
        self._blink_timer.stop()
        self._blink_red = False
        self.info_group_container.setStyleSheet("")

    def _toggle_blink(self):
        self._blink_red = not self._blink_red
        if self._blink_red:
            self.info_group_container.setStyleSheet("#infoGroupBody {background-color: rgba(255,30,30,0.25); border-radius:8px;}")
        else:
            self.info_group_container.setStyleSheet("")

    def set_first_detect_time(self, qdt: QDateTime):
        self._first_dt = qdt
        self._elapsed_timer.stop()
        self._elapsed_timer.start(1000)
        self._update_elapsed()

    def _update_elapsed(self):
        if not self._first_dt:
            self.elapsed_label.setText('- [경과 시간] -')
            return
        secs = self._first_dt.secsTo(QDateTime.currentDateTime())
        if secs < 0:
            secs = 0
        hh = secs // 3600
        mm = (secs % 3600) // 60
        ss = secs % 60
        self.elapsed_label.setText(f"- [경과 시간] {hh:02d}시간 {mm:02d}분 {ss:02d}초")

    def set_info(self, base_info_text: str, when_text: str, count_text: str):
        display_text = self._format_base_info(base_info_text)
        self.info_label.setText(f"- [기본 정보] {display_text}")
        self.time_label.setText(f"- [발생 시각] {when_text}")
        self.count_label.setText(f"- [누적 횟수] {count_text}")

    def _format_base_info(self, text: str) -> str:
        parts = text.split('_')
        if len(parts) >= 4:
            return '_'.join(parts[:4])
        return text

    def set_graph_pixmap(self, pm: QPixmap):
        if pm is None or pm.isNull():
            self._graph_pixmap = QPixmap()
            self.graph_view.clear()
            return
        self._graph_pixmap = QPixmap(pm)
        self._refresh_graph_view()

    def append_log(self, text: str, when: QDateTime | None = None):
        if when is None:
            when = QDateTime.currentDateTime()
        ts = when.toString('yy/MM/dd_HH:mm:ss')
        self.log_edit.append(f"[{ts}] {text}")
        if self.log_edit.verticalScrollBar() is not None:
            self.log_edit.verticalScrollBar().setValue(self.log_edit.verticalScrollBar().maximum())

    def set_images(self, image_infos):
        # 기존 라벨 제거
        for lab in self._pix_labels:
            self.img_container.removeWidget(lab)
            lab.deleteLater()
        self._pix_labels.clear()

        infos = list(image_infos)

        for entry in infos[:5]:  # 최대 5장, 좌->우
            if isinstance(entry, tuple):
                p, count = entry
            else:
                p, count = entry, 0
            pix = self._load_pix(p)
            image_label = QLabel()
            image_label.setPixmap(pix)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setFixedSize(180, 240)
            image_label.setToolTip(p.name)

            ts_label = QLabel(f"[{self._format_image_timestamp(p)}] {count:02d}회")
            ts_label.setAlignment(Qt.AlignCenter)
            ts_label.setStyleSheet("color: #333; font-size: 10pt;")

            wrapper = QWidget()
            wrapper_layout = QVBoxLayout(wrapper)
            wrapper_layout.setContentsMargins(2, 2, 2, 2)
            wrapper_layout.setSpacing(4)
            wrapper_layout.addWidget(ts_label)
            wrapper_layout.addWidget(image_label)
            wrapper_layout.addStretch()

            self.img_container.insertWidget(self.img_container.count()-1, wrapper)
            self._pix_labels.append(wrapper)


    def _load_pix(self, p: Path) -> QPixmap:
        img = safe_image_load(str(p), as_gray=False)
        if img is None:
            return QPixmap()

        if img.ndim == 2:
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            if img.shape[2] == 3:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb.shape
                qimg = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
            elif img.shape[2] == 4:
                rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                h, w, _ = rgba.shape
                qimg = QImage(rgba.data, w, h, 4*w, QImage.Format_RGBA8888)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape
                qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)

        return QPixmap.fromImage(qimg).scaled(180, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)


    def _format_image_timestamp(self, path: Path) -> str:
        qdt = self._extract_image_datetime(path)
        return qdt.toString('MM/dd_HH:mm:ss')

    def _extract_image_datetime(self, path: Path) -> QDateTime:
        stem = path.stem
        m = re.search(r'(\d{4})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{0,2})', stem)
        if m:
            yyyy = int(m.group(1))
            mm = int(m.group(2))
            dd = int(m.group(3))
            hh = int(m.group(4))
            mi = int(m.group(5))
            ss = int(m.group(6)) if m.group(6) else 0
            qdt = QDateTime(yyyy, mm, dd, hh, mi, ss)
            if qdt.isValid():
                return qdt
        try:
            stat = path.stat()
            qdt = QDateTime.fromSecsSinceEpoch(int(stat.st_mtime), Qt.LocalTime)
            if qdt.isValid():
                return qdt
        except Exception:
            pass
        return QDateTime.currentDateTime()

    def _refresh_graph_view(self):
        if self._graph_pixmap.isNull():
            self.graph_view.clear()
            return
        size = self.graph_view.size()
        w = max(1, size.width())
        h = max(1, size.height())
        scaled = self._graph_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.graph_view.setPixmap(scaled)

    def eventFilter(self, obj, event):
        if obj is self.graph_view and event.type() == QEvent.Resize:
            self._refresh_graph_view()
        return super().eventFilter(obj, event)

    def set_event_folder(self, path: Path | str | None):
        self._event_folder_path = Path(path) if path else None
        self.btn_open_folder.setEnabled(self._event_folder_path is not None)

    def _on_open_folder(self):
        if not self._event_folder_path:
            return
        try:
            folder_str = str(self._event_folder_path)
            if os.name == 'nt':
                os.startfile(folder_str)
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(folder_str))
        except Exception:
            pass

    def closeEvent(self, event):
        self.stop_blink()
        self._elapsed_timer.stop()
        self._first_dt = None
        super().closeEvent(event)

    def _close_with(self, reason: str):
        self.stop_blink()
        self._elapsed_timer.stop()
        self._first_dt = None
        snooze_min = self.ignore_min_spin.value() if self.ignore_cb.isChecked() else 0
        now = QDateTime.currentDateTime()
        if snooze_min > 0:
            self.append_log(f"{snooze_min:02d}분 팝업 무시 체크", now)
        if reason == 'ok':
            self.append_log('확인 종료', now)
        else:
            self.append_log('창 종료', now)
        self.closedWithAction.emit(reason, snooze_min)
        self.close()


class ParticleDetectionGUI(QWidget):
    """메인 GUI 위젯 클래스"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Particle Detector GUI')
        self.resize(500, 400)
        self.settings = QSettings(str(INI_SETTINGS_PATH), QSettings.IniFormat)
        self._window_position_restored = False

        # 텍스트 박스 ('장비명_Lot#_Process' 표기)
        self.device_info_box = QLineEdit()
        self.device_info_box.setReadOnly(True)
        self.device_info_box.setFixedWidth(180)
        self.device_info_box.setAlignment(Qt.AlignCenter)

        # CSV, Image, Graph(스냅샷) 관리 형식
        self.current_lot = None
        self.current_attempt = None
        self.csv_path = None
        self.csv_header = ['이미지 파일명', 'Attempt 횟수', '밝기', 'adaptive_value', '파티클 발생 여부', '파티클_크기', '파티클_x', '파티클_y']
        self.processed_images: set[str] = set()
        self.session_processed_images: set[str] = set()
        self.saved_graph_lots: set[str] = set()
        self.graph_records: list[tuple[str, int, float]] = []
        self.pending_images: deque[Path] = deque()
        self._is_processing_queue = False

        # Noise Particle 판정 파라미터
        self.prev_particles = []         # 워밍 업 시, 반복 좌표 및 크기 기억
        self.frame_buffer = []           # 워밍 업 시, 파티클 리스트 버퍼
        self.distance_threshold = 20     # 거리 임계값 (반경 내 같은 Noise로 인식)
        self.area_threshold = 20         # 면적 임계값 (차이 내 같은 Noise로 인식)
        self.collecting_initial = True   # 최초 10장 워밍업 플래그

        # 허용 모드 집합 & 래치/홀드 상태
        self.ALLOWED_PULLER_MODES = {"NECK"}  # 허용 모드
        self.disallowed_mode_latched = False  # 비허용 모드 래치
        self.status_hold_until = 0.0          # 상태 메시지 유지 만료 시각
        self.last_image_seen_at = 0.0         # 마지막 새 이미지 관측 시각
        self.no_new_image_grace_s = 9.0       # 신규 이미지 없음 표시 유예 초

        # 팝업/무시/누적
        self.alert_popup: AlertPopup | None = None
        self.popup_snooze_until = 0.0
        self.lot_event_count = 0
        self._popup_images = deque(maxlen=5)
        self._popup_detect_dt: QDateTime | None = None

        # crop1 영역 Spinbox 설정 및 저장값 로딩
        self.crop1_x = QSpinBox(maximum=9999)
        self.crop1_x.setValue(int(self.settings.value('crop1_x', self.settings.value('crop_x', 95))))
        self.crop1_y = QSpinBox(maximum=9999)
        self.crop1_y.setValue(int(self.settings.value('crop1_y', self.settings.value('crop_y', 0))))
        self.crop1_w = QSpinBox(maximum=9999)
        self.crop1_w.setValue(int(self.settings.value('crop1_w', self.settings.value('crop_w', 85))))
        self.crop1_h = QSpinBox(maximum=9999)
        self.crop1_h.setValue(int(self.settings.value('crop1_h', self.settings.value('crop_h', 110))))

        # crop2 영역 Spinbox 설정 및 저장값 로딩
        self.crop2_x = QSpinBox(maximum=9999)
        self.crop2_x.setValue(int(self.settings.value('crop2_x', 113)))
        self.crop2_y = QSpinBox(maximum=9999)
        self.crop2_y.setValue(int(self.settings.value('crop2_y', 110)))
        self.crop2_w = QSpinBox(maximum=9999)
        self.crop2_w.setValue(int(self.settings.value('crop2_w', 50)))
        self.crop2_h = QSpinBox(maximum=9999)
        self.crop2_h.setValue(int(self.settings.value('crop2_h', 120)))

        # CROP 영역 스핀박스 설정
        SPIN_W = 77  # 스핀박스 너비
        for s in [self.crop1_x, self.crop1_y, self.crop1_w, self.crop1_h,
                  self.crop2_x, self.crop2_y, self.crop2_w, self.crop2_h]:
            s.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            s.setFixedWidth(SPIN_W)

        # 체크박스 생성 및 저장값 로딩
        def _load_bool(key, default=True):
            v = self.settings.value(key, default)
            if isinstance(v, bool):
                return v
            return str(v).lower() in ('1', 'true', 'yes', 'y', 'on')

        # Crop 박스 On/Off Default 설정
        self.crop1_enabled_cb = QCheckBox()
        self.crop1_enabled_cb.setChecked(_load_bool('crop1_enabled', True))

        self.crop2_enabled_cb = QCheckBox()
        self.crop2_enabled_cb.setChecked(_load_bool('crop2_enabled', True))

        # 위젯 초기화
        self.start_button = QPushButton('시작')
        self.stop_button = QPushButton('종료')
        self.stop_button.setEnabled(False)
        self.save_opt_button = QPushButton('설정 저장')
        self.status_label = QLabel('- 대기 중 -')
        self.preview_label = ImagePreviewLabel()
        self.plot_canvas = ParticlePlotCanvas()
        self.auto_start_on_launch = AUTO_START_ON_LAUNCH

        # 레이아웃 구성
        main_hbox = QHBoxLayout(self)
        main_hbox.setContentsMargins(10, 15, 10, 10)  # 좌, 상, 우, 하 여백
        main_hbox.setSizeConstraint(QLayout.SetMinimumSize)

        left_panel = QVBoxLayout()
        left_panel.addWidget(self.status_label)

        btns = QHBoxLayout()
        btns.addWidget(self.device_info_box)
        btns.addWidget(self.start_button)
        btns.addWidget(self.stop_button)
        btns.addWidget(self.save_opt_button)
        left_panel.addLayout(btns)

        crop_box = QGroupBox(' 제외 영역 설정 ')
        crop_vbox = QVBoxLayout()

        crop_hbox1 = QHBoxLayout()
        crop_hbox1.addWidget(self.crop1_enabled_cb)
        crop_hbox1.addWidget(QLabel("X:")); crop_hbox1.addWidget(self.crop1_x)
        crop_hbox1.addWidget(QLabel("Y:")); crop_hbox1.addWidget(self.crop1_y)
        crop_hbox1.addWidget(QLabel("W:")); crop_hbox1.addWidget(self.crop1_w)
        crop_hbox1.addWidget(QLabel("H:")); crop_hbox1.addWidget(self.crop1_h)
        crop_vbox.addLayout(crop_hbox1)

        crop_hbox2 = QHBoxLayout()
        crop_hbox2.addWidget(self.crop2_enabled_cb)
        crop_hbox2.addWidget(QLabel("X:")); crop_hbox2.addWidget(self.crop2_x)
        crop_hbox2.addWidget(QLabel("Y:")); crop_hbox2.addWidget(self.crop2_y)
        crop_hbox2.addWidget(QLabel("W:")); crop_hbox2.addWidget(self.crop2_w)
        crop_hbox2.addWidget(QLabel("H:")); crop_hbox2.addWidget(self.crop2_h)
        crop_vbox.addLayout(crop_hbox2)

        crop_box.setLayout(crop_vbox)
        left_panel.addWidget(crop_box)

        left_panel.addWidget(self.plot_canvas)
        left_panel.addStretch()

        right_panel = QVBoxLayout()
        right_panel.addSpacing(24)  # 이미지 미리보기 상단부 여백
        right_panel.addWidget(self.preview_label, alignment=Qt.AlignTop | Qt.AlignLeft)
        right_panel.addStretch()

        main_hbox.addLayout(left_panel)
        main_hbox.addLayout(right_panel)

        # 시그널 연결 (미리보기 즉시 반영)
        for s in [self.crop1_x, self.crop1_y, self.crop1_w, self.crop1_h,
                  self.crop2_x, self.crop2_y, self.crop2_w, self.crop2_h]:
            s.valueChanged.connect(self.update_preview)
        self.start_button.clicked.connect(self.start_realtime)
        self.stop_button.clicked.connect(self.stop_realtime)
        self.save_opt_button.clicked.connect(self.save_options)

        # Crop 영역 체크박스 (활성/비활성)
        self.crop1_enabled_cb.toggled.connect(self._on_crop1_toggled)
        self.crop2_enabled_cb.toggled.connect(self._on_crop2_toggled)

        # 라이브 감시기 (watchdog) 및 대기 타이머
        self.directory_watcher = ImageDirectoryWatcher(IMG_INPUT_DIR, parent=self)
        self.directory_watcher.file_created.connect(self._enqueue_new_image)
        self._allowed_extensions = set(ALLOWED_IMAGE_EXTS)

        self.tail_timer = QTimer(self)
        self.tail_timer.setInterval(500)
        self.tail_timer.timeout.connect(self.process_new_images_tick)

        # 백로그 스레드 구성 요소
        self._backlog_thread: QThread | None = None
        self._backlog_feeder: BacklogFeeder | None = None

        # 표시 제어 플래그
        self.show_noise_text = False
        self.preview_label.set_noise_text("")
        self._apply_crop_enable_states()

        self.update_preview()

        if hasattr(self, "update_graph"):
            self.update_graph()
        else:
            self.plot_canvas.update_plot([], [], latest_index=None)

    def showEvent(self, event):
        """"메인 창 최초 표시 시 INI 저장 좌표 복원 또는 좌하단 오프셋 기본 위치 적용"""
        super().showEvent(event)
        if not self._window_position_restored:
            self._window_position_restored = True
            self._restore_window_position()

    def _restore_window_position(self):
        """INI(QSettings)에 저장된 메안 창 좌표를 복원"""
        saved_x = self.settings.value('main_window_pos_x', None)
        saved_y = self.settings.value('main_window_pos_y', None)
        try:
            x_val = None if saved_x is None else int(float(saved_x))
            y_val = None if saved_y is None else int(float(saved_y))
        except (TypeError, ValueError):
            x_val = y_val = None

        if x_val is not None and y_val is not None:
            x_val, y_val = self._adjust_position_to_screen(x_val, y_val)
            self.move(x_val, y_val)
            return

        self._move_to_default_position()

    def _move_to_default_position(self):
        """저장 값이 없을 때, 기본 위치로 메인 창을 이동"""
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return

        available = screen.availableGeometry()
        frame_geo = self.frameGeometry()
        window_width = frame_geo.width()
        window_height = frame_geo.height()
        target_x = available.x() + WINDOW_OFFSET_X
        target_y = available.y() + available.height() - WINDOW_OFFSET_Y - window_height
        target_x, target_y = self._adjust_position_to_screen(target_x, target_y, window_width, window_height)
        self.move(target_x, target_y)

    def _adjust_position_to_screen(self, x, y, window_width=None, window_height=None):
        """메인 창의 좌표가 현재 모니터의 표시 가능한 영역을 벗어나지 않도록 보정"""
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return int(round(x)), int(round(y))

        available = screen.availableGeometry()
        if window_width is None or window_height is None:
            frame_geo = self.frameGeometry()
            if window_width is None:
                window_width = frame_geo.width()
            if window_height is None:
                window_height = frame_geo.height()

        min_x = available.x()
        min_y = available.y()
        max_x = max(min_x, available.x() + available.width() - window_width)
        max_y = max(min_y, available.y() + available.height() - window_height)

        clamped_x = max(min_x, min(int(round(x)), max_x))
        clamped_y = max(min_y, min(int(round(y)), max_y))
        return clamped_x, clamped_y

    def _set_status(self, text: str, hold_s: float = 0.0):
        """상태 텍스트 세터 (동일 텍스트 중복 세팅 방지 + 홀드 지원)"""
        if hold_s <= 0.0 and self.status_label.text() == text:
            return
        self.status_label.setText(text)
        if hold_s > 0.0:
            self.status_hold_until = time.monotonic() + hold_s
        else:
            self.status_hold_until = 0.0

    def _reset_noise_warmup(self, status_text: str):
        """노이즈 정의 및 워밍업 상태 초기화"""
        self.collecting_initial = True
        self.prev_particles.clear()
        self.frame_buffer.clear()
        self.show_noise_text = False
        self._set_status(status_text)
        self._update_noise_text()

    def _apply_crop_enable_states(self):
        """Crop 체크박스 설정에 따른 스핀박스 활성화/비활성화"""
        enabled1 = self.crop1_enabled_cb.isChecked()
        for s in [self.crop1_x, self.crop1_y, self.crop1_w, self.crop1_h]:
            s.setEnabled(enabled1)
        enabled2 = self.crop2_enabled_cb.isChecked()
        for s in [self.crop2_x, self.crop2_y, self.crop2_w, self.crop2_h]:
            s.setEnabled(enabled2)

    def _mark_image_processed(self, filename: str):
        """현재 Lot/세션 처리 목록에 이미지명 기록"""
        if not filename:
            return
        self.processed_images.add(filename)
        self.session_processed_images.add(filename)

    @Slot(object)
    def _enqueue_new_image(self, path_obj):
        """watchdog로 감지된 신규 이미지를 큐에 적재"""
        if path_obj is None:
            return
        path = Path(path_obj)
        if path.suffix.lower() not in self._allowed_extensions:
            return
        if path.name in self.session_processed_images:
            return
        self.pending_images.append(path)
        if not (self._backlog_thread and self._backlog_thread.isRunning()):
            self._drain_pending_queue()

    def _drain_pending_queue(self):
        """큐에 쌓인 신규 이미지를 순차 처리"""
        if self._is_processing_queue:
            return
        if self._backlog_thread and self._backlog_thread.isRunning():
            return
        self._is_processing_queue = True
        try:
            while self.pending_images:
                path = self.pending_images.popleft()
                if not isinstance(path, Path):
                    path = Path(path)
                if path.suffix.lower() not in self._allowed_extensions:
                    continue
                if path.name in self.session_processed_images:
                    continue
                self.process_single_image(path)
        finally:
            self._is_processing_queue = False

    @Slot(bool)
    def _on_crop1_toggled(self, checked: bool):
        "체크박스1 토글 핸들러"
        self._apply_crop_enable_states()
        self.update_preview()

    @Slot(bool)
    def _on_crop2_toggled(self, checked: bool):
        "체크박스2 토글 핸들러"
        self._apply_crop_enable_states()
        self.update_preview()

    def get_box1(self):
        """Crop 박스1 좌표 반환"""
        if not self.crop1_enabled_cb.isChecked():
            return (0, 0, 0, 0)
        return (self.crop1_x.value(), self.crop1_y.value(),
                self.crop1_w.value(), self.crop1_h.value())

    def get_box2(self):
        """Crop 박스2 좌표 반환"""
        if not self.crop2_enabled_cb.isChecked():
            return (0, 0, 0, 0)
        return (self.crop2_x.value(), self.crop2_y.value(),
                self.crop2_w.value(), self.crop2_h.value())

    def _update_noise_text(self):
        """Noise 정보 텍스트 생성/반영"""
        if not self.show_noise_text or self.collecting_initial:
            self.preview_label.set_noise_text("")
            self.preview_label.set_noise_points([])
            return
        if self.prev_particles:
            text = "; ".join([f"[Noise] X:{x}, Y:{y}, Size:{a}" for (x, y, a) in self.prev_particles])
        else:
            text = "[Noise] None"
        self.preview_label.set_noise_text(text)
        self.preview_label.set_noise_points(self.prev_particles)

    def update_preview(self):
        """이미지 미리보기 / 상부 텍스트 갱신 (최신파일 기준)"""
        try:
            cands = [p for p in IMG_INPUT_DIR.iterdir() if p.suffix.lower() in ALLOWED_IMAGE_EXTS]
            if not cands:
                self.preview_label.show_placeholder(NO_IMAGE_PLACEHOLDER_TEXT)
                return

            latest = max(cands, key=lambda p: p.stat().st_mtime)
            img = safe_image_load(str(latest), as_gray=True)
            if img is None:
                self.preview_label.show_placeholder(NO_IMAGE_PLACEHOLDER_TEXT)
                return

            boxes = [self.get_box1(), self.get_box2()]
            self.preview_label.show_image(img, boxes, filename=latest.name)

            # 파일명 형식: EQUIP_Lot#_Process_Attempt_YYYYMMDD_HHMM(SS)
            fields = latest.stem.split('_')  # [0]=장비, [1]=Lot, [2]=Process, [3]=Attempt, [4]=YYYYMMDD, [5]=HHMM(SS)

            # 허용 모드일 때만 앵커 밝기 표시
            process_ok = (fields[2] in self.ALLOWED_PULLER_MODES)
            self.preview_label.set_anchor_value(compute_anchor_brightness(img) if process_ok else None)

            # 상단 텍스트 박스: 장비명_Lot#_Process_Att
            self.device_info_box.setText('_'.join(fields[:4]))

        except Exception:
            self.preview_label.show_placeholder(NO_IMAGE_PLACEHOLDER_TEXT)


    def start_realtime(self):
        """실시간 파티클 판정 시작 (백로그 → 라이브 테일 순)"""
        # 워밍업 초기화
        self.collecting_initial = True
        self.prev_particles = []
        self.frame_buffer = []
        self.saved_graph_lots.clear()

        self.show_noise_text = False
        self._update_noise_text()

        self.status_label.setText(f"초기 워밍업 중... (최초 {WARMUP_FRAMES}장 데이터 수집)")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.pending_images.clear()
        self._is_processing_queue = False
        self.directory_watcher.start()
        self.tail_timer.start()
        self.last_image_seen_at = time.monotonic()

        # 백로그 목록 구성 (processed_images 기준 미처리 파일 전체)
        backlog = find_new_images(IMG_INPUT_DIR, self.session_processed_images)
        if backlog:
            # 백로그를 메인스레드에 차례로 공급
            self._backlog_thread = QThread()
            self._backlog_feeder = BacklogFeeder(backlog)
            self._backlog_feeder.moveToThread(self._backlog_thread)
            self._backlog_thread.started.connect(self._backlog_feeder.run)
            self._backlog_feeder.next_image.connect(self.process_single_image)
            self._backlog_feeder.finished.connect(self._on_backlog_finished)
            # 안전 종료 연결
            self._backlog_feeder.finished.connect(self._backlog_thread.quit)
            self._backlog_thread.finished.connect(self._backlog_thread.deleteLater)
            self._backlog_thread.finished.connect(lambda: setattr(self, "_backlog_thread", None))
            self._backlog_thread.destroyed.connect(lambda: setattr(self, "_backlog_thread", None))
            self._backlog_feeder.finished.connect(lambda: setattr(self, "_backlog_feeder", None))
            self._backlog_thread.finished.connect(self._drain_pending_queue)
            self._backlog_thread.start()
            self.status_label.setText(f"백로그 처리 시작: {len(backlog)}장")
        else:
            self.status_label.setText("백로그 없음. 라이브 감시 시작.")
            self._drain_pending_queue()
        # 초기 Noise 텍스트
        self._update_noise_text()

    def stop_realtime(self):
        """실시간 파티클 판정 중지"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.tail_timer.isActive():
            self.tail_timer.stop()
        self.directory_watcher.stop()
        self.pending_images.clear()
        self._is_processing_queue = False
        # 백로그 중지
        try:
            if self._backlog_feeder:
                self._backlog_feeder.stop()
            if self._backlog_thread and self._backlog_thread.isRunning():
                self._backlog_thread.quit()
                self._backlog_thread.wait(2000)
        except Exception:
            pass
        self.status_label.setText("- 중지 -")

    def save_options(self):
        """crop 영역 및 메인 창 위치 설정값 저장"""
        # crop1
        self.settings.setValue('crop1_x', self.crop1_x.value())
        self.settings.setValue('crop1_y', self.crop1_y.value())
        self.settings.setValue('crop1_w', self.crop1_w.value())
        self.settings.setValue('crop1_h', self.crop1_h.value())
        self.settings.setValue('crop1_enabled', bool(self.crop1_enabled_cb.isChecked()))
        # crop2
        self.settings.setValue('crop2_x', self.crop2_x.value())
        self.settings.setValue('crop2_y', self.crop2_y.value())
        self.settings.setValue('crop2_w', self.crop2_w.value())
        self.settings.setValue('crop2_h', self.crop2_h.value())
        self.settings.setValue('crop2_enabled', bool(self.crop2_enabled_cb.isChecked()))

        # 메인 창 위치 저장
        top_left = self.frameGeometry().topLeft()
        self.settings.setValue('main_window_pos_x', int(top_left.x()))
        self.settings.setValue('main_window_pos_y', int(top_left.y()))

        self.settings.sync()
        self.status_label.setText("설정 저장됨.")
    
    def closeEvent(self, event):
        """윈도우 종료 시 현재 설정을 .ini에 자동 저장"""
        try:
            self.stop_realtime()
            self.save_options()
        except Exception:
            pass
        finally:
            super().closeEvent(event)

    def process_new_images_tick(self):
        """watchdog 큐 처리 및 신규 이미지 부재 상태 갱신"""
        self._drain_pending_queue()
        now = time.monotonic()
        if self.pending_images:
            return
        if now < self.status_hold_until:
            return
        if (now - self.last_image_seen_at) <= self.no_new_image_grace_s:
            return
        self.preview_label.show_placeholder(NO_IMAGE_PLACEHOLDER_TEXT)
        self._set_status("신규 이미지 없음 (폴더 감시 중)")

    @Slot()
    def _on_backlog_finished(self):
        """백로그 완료 콜백 → 라이브 감시 시작"""
        self.status_label.setText("백로그 완료. 라이브 감시 시작.")
        # 최신 샘플 1장으로 미리보기 갱신
        self.update_preview()
        self._drain_pending_queue()


    @Slot(object)
    def process_single_image(self, img_path):
        """단일 이미지 처리 루틴 (메인 스레드에서 실행 → 모든 UI 요소 매 이미지 갱신)"""
        try:
            img_path = Path(img_path)
            img_name = img_path.name

            # 파일명 형식 기반 파싱: EQUIP_Lot#_Process_Attempt_YYYYMMDD_HHMM(SS).ext
            info = parse_filename(img_name)
            lot_number    = info['lot']
            process_name  = info['process']
            attempt_value = info['attempt']

            # Lot 변경 시 CSV 생성 + 그래프 초기화 + 워밍업 리셋 + 팝업 초기화
            if self.current_lot != lot_number:
                # 직전 Lot 그래프 스냅샷 저장
                try:
                    prev_lot = self.current_lot
                    prev_csv = self.csv_path
                    if (
                        prev_lot
                        and prev_lot != "LotUnknown"
                        and prev_csv
                        and prev_csv.exists()
                        and prev_lot not in self.saved_graph_lots
                    ):
                        out_dir = GRAPH_OUTPUT_DIR
                        ts = time.strftime('%Y%m%d_%H%M%S')
                        out_path = out_dir / f"{prev_lot}_{ts}.jpg"
                        self.plot_canvas.fig.savefig(
                            str(out_path),
                            dpi=self.plot_canvas.fig.dpi,
                            bbox_inches='tight'
                        )
                        self.saved_graph_lots.add(prev_lot)
                except Exception as e:
                    self._set_status(f"그래프 이미지 저장 오류: {e}", hold_s=3.0)

                # 새 Lot로 전환
                self.current_lot = lot_number

                new_lot_valid = bool(lot_number) and (lot_number != "LotUnknown")
                self.csv_path = (CSV_OUTPUT_DIR / f"{lot_number}.csv") if new_lot_valid else None

                self.current_attempt = attempt_value
                self.processed_images.clear()
                self.lot_event_count = 0
                self._popup_images.clear()
                self._popup_detect_dt = None
                if self.alert_popup:
                    self.alert_popup.close()
                    self.alert_popup = None

                # 새 CSV 생성(헤더만)
                try:
                    if self.csv_path and ((not self.csv_path.exists()) or (self.csv_path.stat().st_size == 0)):
                        with open(self.csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                            writer = csv.writer(f)
                            writer.writerow(self.csv_header)
                except Exception as e:
                    self._set_status(f"CSV 초기화 오류: {e}", hold_s=3.0)

                # 기존 CSV가 있으면 중복 방지 세트 재로딩
                if self.csv_path and self.csv_path.exists():
                    try:
                        with open(self.csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                            reader = csv.reader(f)
                            header = next(reader, None)
                            for row in reader:
                                if row:
                                    self._mark_image_processed(row[0])
                    except Exception:
                        pass

                # 워밍업/Noise/상태 리셋
                self._reset_noise_warmup(f"LOT 변경 감지 → 워밍업 재시작 ({WARMUP_FRAMES}장)")

                # 빈 CSV 기반 그래프 즉시 초기화
                self.graph_records.clear()
                self._load_graph_buffer_from_csv()
                self.update_graph()

            else:
                if self.current_attempt is None:
                    self.current_attempt = attempt_value
                elif self.current_attempt != attempt_value:
                    self.current_attempt = attempt_value
                    self._reset_noise_warmup(f"Attempt 변경 감지 → 워밍업 재시작 ({WARMUP_FRAMES}장)")

            # Lot 전환 이후 CSV 기반 중복 세트 로딩이 완료된 상태에서 중복 여부 재확인
            if img_name in self.processed_images:
                self.status_label.setText(f"이미 처리됨: {img_path.name}")
                return

            gray_prev = safe_image_load(str(img_path), as_gray=True)
            if gray_prev is None:
                self.status_label.setText(f"오류: {img_path.name} 이미지 로딩 실패")
                self._mark_image_processed(img_path.name)
                return

            self.last_image_seen_at = time.monotonic()

            # 허용 모드 체크
            if process_name not in self.ALLOWED_PULLER_MODES:
                # 비허용 모드 → 래치 온 (상태 고정)
                self.disallowed_mode_latched = True
                self.show_noise_text = False
                self._update_noise_text()
                self._set_status("지정된 Puller Mode Status가 아님.")

                # 미리보기는 최신 이미지로 갱신하되 앵커값은 숨김
                if gray_prev is not None:
                    self.preview_label.show_image(
                        gray_prev, [self.get_box1(), self.get_box2()],
                        filename=img_path.name
                    )
                else:
                    self.preview_label.show_placeholder(NO_IMAGE_PLACEHOLDER_TEXT)
                self.preview_label.set_anchor_value(None)

                self._mark_image_processed(img_path.name)
                return
            else:
                # 허용 모드 → 래치 해제
                if self.disallowed_mode_latched:
                    self.disallowed_mode_latched = False

            # 현재 처리 대상 파일 기준으로 이미지 미리보기 표기 + 앵커값
            anchor_current = None
            if gray_prev is not None:
                anchor_current = compute_anchor_brightness(gray_prev)
                self.preview_label.show_image(gray_prev, [self.get_box1(), self.get_box2()], filename=img_path.name)
                self.preview_label.set_anchor_value(anchor_current)
            else:
                self.preview_label.show_placeholder(NO_IMAGE_PLACEHOLDER_TEXT)
                self.preview_label.set_anchor_value(None)

            # 현재 Noise 정보 텍스트 갱신 및 임계값 계산
            adaptive_value = compute_adaptive_threshold(anchor_current)
            self._update_noise_text()

            # 워밍업 구간 (최초 10장 동안은 '판정/CSV/그래프' 수행하지 않음)
            if self.collecting_initial:
                self.status_label.setText(f"초기 워밍업 수집 중... ({len(self.frame_buffer) + 1} / {WARMUP_FRAMES})")
                self.show_noise_text = False
                self._update_noise_text()

                # 후보 파티클만 수집(판정/CSV/그래프/이벤트 저장 없음)
                _, _, particle_info = particle_detection(gray_prev, [self.get_box1(), self.get_box2()], adaptive_value, area_min=WARMUP_NOISE_AREA_MIN, area_max=PARTICLE_AREA_MAX)
                if particle_info is None:
                    # 로딩/후보 추출 실패 시 스킵
                    self.status_label.setText(f"오류: {img_path.name} 로딩/후보 추출 실패 (워밍업)")
                    self._mark_image_processed(img_path.name)
                    return

                self.frame_buffer.append(particle_info)

                if len(self.frame_buffer) >= WARMUP_FRAMES:
                    # MIN_NOISE_REPEAT회 이상 등장 좌표를 노이즈 기준으로 채택
                    freq = {}
                    for frame_particles in self.frame_buffer:
                        for (x, y, area) in frame_particles:
                            key = (x, y, area)
                            freq[key] = freq.get(key, 0) + 1
                    self.prev_particles = [k for k, v in freq.items() if v >= MIN_NOISE_REPEAT] or (self.frame_buffer[-1] if self.frame_buffer else [])

                    # 워밍업 종료 이후부터 '허용 모드의 판정 이미지'에서 [Noise] 표기/적용
                    self.collecting_initial = False
                    self.frame_buffer.clear()
                    self.show_noise_text = True
                    self.status_label.setText("▶ 워밍업 완료 → 실시간 판정 시작")
                    self._update_noise_text()

                self._mark_image_processed(img_path.name)
                return

            # 실시간 판정 구간 (워밍업 이후 이미지부터)
            self.status_label.setText(f"처리 중: {img_path.name}")

            self.show_noise_text = True
            self._update_noise_text()

            # 실제 판정 처리 (OpenCV 파이프라인)
            gray, overlay_img, particle_info = particle_detection(gray_prev, [self.get_box1(), self.get_box2()], adaptive_value)
            if overlay_img is None or particle_info is None:
                self.status_label.setText(f"오류: {img_path.name} 처리 실패")
                self._mark_image_processed(img_path.name)
                return

            # 이미지의 앵커 밝기 계산 (CSV 기록용)
            anchor_brightness = anchor_current if anchor_current is not None else compute_anchor_brightness(gray)

            # 유효 파티클 판별 (prev_particles와 매칭되지 않은 것만)
            valid_particles = []
            for (cx, cy, area) in particle_info:
                matched = False
                for (px, py, parea) in self.prev_particles:
                    dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    area_diff = abs(area - parea)
                    if dist < self.distance_threshold and area_diff < self.area_threshold:
                        matched = True
                        break
                if not matched:
                    valid_particles.append((cx, cy, area))

            has_particle = 'O' if valid_particles else 'X'

            # 오버레이: prev(흰) + valid(빨강) + [Noise] 텍스트
            if overlay_img is not None:
                # prev (white)
                for (px, py, parea) in self.prev_particles:
                    cv2.circle(overlay_img, (px, py), NOISE_CIRCLE_RADIUS, (255, 255, 255), NOISE_CIRCLE_THICKNESS, cv2.LINE_AA)
                # valid (red)
                for (vx, vy, varea) in valid_particles:
                    cv2.circle(overlay_img, (vx, vy), VALID_CIRCLE_RADIUS, (0, 0, 255), VALID_CIRCLE_THICKNESS, cv2.LINE_AA)
                # [Noise] 텍스트
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness_text = 1
                noise_text = "[Noise] None" if not self.prev_particles else "; ".join([f"[Noise] X:{x}, Y:{y}, Size:{a}" for (x, y, a) in self.prev_particles])
                img_h, img_w = overlay_img.shape[:2]
                (text_w, text_h), _ = cv2.getTextSize(noise_text, font, font_scale, thickness_text)
                text_x = img_w - text_w - 10
                text_y = img_h - 10
                cv2.putText(overlay_img, noise_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness_text, cv2.LINE_AA)

            # CSV 저장
            if has_particle == 'O':
                for (x, y, a) in valid_particles:
                    row = [img_path.name, attempt_value, anchor_brightness, adaptive_value, has_particle, a, x, y]
                    self.save_csv(row)
            else:
                row = [img_path.name, attempt_value, anchor_brightness, adaptive_value, has_particle, 0, 0, 0]
                self.save_csv(row)

            # 이벤트 이미지 저장 (Lot# 폴더 내) + 팝업 호출/갱신
            if has_particle == 'O' and overlay_img is not None:
                lot_dir = EVENT_OUTPUT_DIR / lot_number
                os.makedirs(lot_dir, exist_ok=True)
                event_path = lot_dir / img_path.name
                cv2.imwrite(str(event_path), overlay_img)

                # 누적 횟수 증가
                self.lot_event_count += 1
                # 팝업 무시 중 여부 확인
                if time.monotonic() < self.popup_snooze_until:
                    # 무시 중이면 이미지 배열만 유지(최대 5장)
                    self._push_popup_image(event_path, self.lot_event_count)
                else:
                    self._maybe_open_or_update_popup(info, event_path)

            # 그래프 갱신
            self.update_graph()

            # 처리 완료 → 3초간 상태 홀드로 깜빡임 억제
            self._set_status(f"처리됨: {img_path.name}  - 파티클:{has_particle}", hold_s=3.0)
            self._mark_image_processed(img_path.name)

        except Exception as e:
            self.status_label.setText(f"오류: {e}")


    def _record_graph_row(self, row):
        """그래프 버퍼에 행 추가"""
        if isinstance(row, dict):
            image_name = row.get('이미지 파일명', '')
            attempt_val = row.get('Attempt 횟수', 0)
            size_val = row.get('파티클_크기', 0)
        else:
            image_name = row[0] if len(row) > 0 else ''
            attempt_val = row[1] if len(row) > 1 else 0
            size_val = row[5] if len(row) > 5 else 0
        if not image_name:
            return
        try:
            attempt_int = int(float(attempt_val))
        except (TypeError, ValueError):
            attempt_int = 0
        try:
            size_float = float(size_val)
        except (TypeError, ValueError):
            size_float = 0.0
        self.graph_records.append((image_name, attempt_int, size_float))

    def _load_graph_buffer_from_csv(self):
        """기존 CSV 데이터를 그래프 버퍼로 로딩"""
        if not self.csv_path or not self.csv_path.exists():
            return
        try:
            with open(self.csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row:
                        self._record_graph_row(row)
        except Exception:
            pass

    def save_csv(self, row, max_retries=5, delay=0.2):
        """결과 CSV 저장 (성공 시 True 반환)"""
        if not self.csv_path:
            return False
        attempt = 0
        while attempt < max_retries:
            try:
                header_exists = self.csv_path.exists()
                with open(self.csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    if not header_exists:
                        writer.writerow(self.csv_header)
                    writer.writerow(row)
                self._record_graph_row(row)
                return True
            except Exception:
                attempt += 1
                time.sleep(delay)
        print("❌ CSV 저장 최종 실패 - 데이터가 저장되지 않았습니다.")
        return False

    def update_graph(self):
        """파티클 그래프 갱신 (Attempt 밴딩 지원)"""
        if not self.graph_records:
            self.plot_canvas.update_plot([], [], latest_index=None)
            return
        image_names = [rec[0] for rec in self.graph_records]
        particle_sizes = np.asarray([rec[2] for rec in self.graph_records], dtype=float)
        attempt_list = [rec[1] for rec in self.graph_records]
        latest_index = len(image_names) - 1
        self.plot_canvas.update_plot(
            image_names,
            particle_sizes,
            latest_index,
            attempt_list=attempt_list
        )


    def _push_popup_image(self, path: Path, event_count: int):
        """팝업창 이미지 출력"""
        self._popup_images.append((path, event_count))

    def _parse_info_datetime(self, info: dict) -> QDateTime:
        # info['date'] = YYYYMMDD, info['time'] = HHMM or HHMMSS
        yyyy = int(info['date'][0:4])
        mm = int(info['date'][4:6])
        dd = int(info['date'][6:8])
        hh = int(info['time'][0:2])
        mi = int(info['time'][2:4])
        ss = int(info['time'][4:6]) if len(info['time']) >= 6 else 0
        qdt = QDateTime(yyyy, mm, dd, hh, mi, ss)
        if not qdt.isValid():
            qdt = QDateTime.currentDateTime()
        return qdt

    def _when_text(self, qdt: QDateTime) -> str:
        return qdt.toString('yy/MM/dd_HH:mm:ss')

    def _graph_pixmap(self) -> QPixmap:
        """현재 Matplotlib Figure를 PNG로 렌더링 후 QPixmap 반환"""
        buf = io.BytesIO()
        try:
            self.plot_canvas.fig.savefig(buf, format='png', dpi=self.plot_canvas.fig.dpi, bbox_inches='tight')
            buf.seek(0)
            qimg = QImage.fromData(buf.read(), 'PNG')
            pm = QPixmap.fromImage(qimg)
        except Exception:
            pm = QPixmap()
        return pm

    def _ensure_popup(self):
        """팝업 창이 독립된 최상위 창으로 생성"""
        if self.alert_popup is None:
            self.alert_popup = AlertPopup(None)
            self.alert_popup.setWindowModality(Qt.NonModal)
            self.alert_popup.closedWithAction.connect(self._on_popup_closed)
        return self.alert_popup

    def _maybe_open_or_update_popup(self, info: dict, event_image_path: Path):
        """팝업 이미지 큐 갱신"""
        self._push_popup_image(event_image_path, self.lot_event_count)
        pop = self._ensure_popup()

        # 발생 시각 (최신 탐지 시간으로 갱신)
        event_dt = self._parse_info_datetime(info)
        self._popup_detect_dt = event_dt

        # 머릿글/정보/그래프/이미지 세팅
        base_info_text = info['stem']
        when_text = self._when_text(event_dt)
        count_text = f"{self.lot_event_count}회"

        pop.set_info(base_info_text, when_text, count_text)
        pop.set_first_detect_time(event_dt)
        pop.set_graph_pixmap(self._graph_pixmap())
        pop.set_images(list(self._popup_images))
        pop.set_event_folder(event_image_path.parent)
        pop.append_log(f"Particle 탐지 {self.lot_event_count}회", event_dt)
        pop.start_blink()

        # 이미 떠 있으면 내용만 갱신, 없으면 표시
        if not pop.isVisible():
            pop.show()
            pop.raise_()
            pop.activateWindow()

    @Slot(str, int)
    def _on_popup_closed(self, reason: str, snooze_min: int):
        """팝업 종료 사유에 따른 로그/무시 설정"""
        if snooze_min > 0:
            self.popup_snooze_until = time.monotonic() + snooze_min * 60
        else:
            self.popup_snooze_until = 0.0

        self._popup_detect_dt = None
        if reason != 'ok':
            self.alert_popup = None


def main():
    app = QApplication(sys.argv)
    gui = ParticleDetectionGUI()
    gui.show()
    if gui.auto_start_on_launch:
        gui.start_realtime()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
