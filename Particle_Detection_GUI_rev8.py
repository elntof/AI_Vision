import os
import sys
import io
from pathlib import Path
import time
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QGroupBox, QSizePolicy, QLineEdit, QLayout, QCheckBox, QTextEdit, QScrollArea,
    QFrame, QSpacerItem
)
from PySide6.QtCore import Qt, QTimer, QSettings, QRect, QThread, QObject, Signal, Slot, QDateTime
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen
import csv
import tempfile
import subprocess
import shutil
from collections import deque

# 한글 폰트 설정 (windows 한글깨짐 방지)
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False

# =========================
# 환경/경로/워치독 관련 설정
# =========================
# 환경 플래그: 경로 및 저장동작 테스트(True) / 운영(False) 모드 구분
LOAD_TEST_MODE = True
SAVE_TEST_MODE = True
INI_TEST_MODE  = True

# 실행파일 실행 시, "자동 시작 동작 수행" 여부 플래그 : 자동 시작(True) / 수동 시작(False)
AUTO_START_ON_LAUNCH = True

# 자동 실행(워치독) 시스템 파일 (프로세스 간 신호/락)
APP_EXE_PREFIX = "Particle"
APP_TAG        = "ParticleDetectionApp"
STOP_FILE      = os.path.join(tempfile.gettempdir(), f"{APP_TAG}.stop")
DISABLE_FILE   = os.path.join(tempfile.gettempdir(), f"{APP_TAG}.disabled")
LOCK_FILE      = os.path.join(tempfile.gettempdir(), f"{APP_TAG}.watchdog.lock")
RUN_BASE_DIR   = Path(tempfile.gettempdir()) / f"{APP_TAG}_run"
ENV_APP_CWD    = "PD_APP_CWD"


def _exe_path():
    """현재 실행 중인 바이너리(혹은 .py) 경로 반환"""
    if getattr(sys, "frozen", False):
        return sys.executable
    return os.path.abspath(sys.argv[0])


def _is_install_location():
    """현재 실행 위치가 설치 경로(D:\\AI Vision)인지 여부 확인"""
    exe_dir = os.path.normpath(os.path.dirname(_exe_path()))
    target_dir = os.path.normpath(r'D:\AI Vision')
    return exe_dir.upper() == target_dir.upper()


def _get_app_cwd():
    """자식/워치독이 사용할 작업 디렉터리(cwd). 스테이징 중에도 설치 경로를 유지"""
    return os.environ.get(ENV_APP_CWD) or os.path.dirname(_exe_path())


def _install_exe_available():
    """설치 폴더에 실행 후보가 존재 감지 및 파일명 변경에도 대응"""
    try:
        install_dir = Path(_get_app_cwd())
        if not install_dir.exists():
            return False
        for p in install_dir.glob("*.exe"):
            if p.name.lower().startswith(APP_EXE_PREFIX.lower()):
                return True
        return False
    except Exception:
        return False


def _find_candidate_install_exe():
    """설치 폴더에서 최신 실행 후보 exe 반환"""
    install_dir = Path(_get_app_cwd())
    if not install_dir.exists():
        return None
    exes = [p for p in install_dir.glob("*.exe") if p.name.lower().startswith(APP_EXE_PREFIX.lower())]
    if not exes:
        return None
    return max(exes, key=lambda p: p.stat().st_mtime)


def _maybe_stage_to_temp():
    """exe 파일 실행 시, 자신을 TEMP로 복사 / '--staged' 로 재실행 / 원본은 즉시 종료(파일 삭제/교체 가능)"""
    if not getattr(sys, "frozen", False):
        return
    if ('--staged' in sys.argv) or ('--no-stage' in sys.argv):
        return

    if _is_install_location():
        RUN_BASE_DIR.mkdir(parents=True, exist_ok=True)
        src = Path(_exe_path())
        sig = f"{int(src.stat().st_mtime)}_{src.stat().st_size}"
        run_dir = RUN_BASE_DIR / sig
        run_dir.mkdir(parents=True, exist_ok=True)

        staged_exe = run_dir / src.name
        try:
            shutil.copy2(src, staged_exe)
        except Exception:
            return

        env = os.environ.copy()
        env[ENV_APP_CWD] = os.path.dirname(_exe_path())
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        subprocess.Popen([str(staged_exe), '--staged'], cwd=env[ENV_APP_CWD], env=env, creationflags=creationflags)
        sys.exit(0)


def _spawn_child_and_wait():
    """자식 GUI 한 번 실행 후 종료 대기"""
    exe = _exe_path()
    if getattr(sys, "frozen", False):
        args = [exe, "--child"]
        cwd = _get_app_cwd()
    else:
        args = [sys.executable, exe, "--child"]
        cwd = os.path.dirname(exe)
    proc = subprocess.Popen(args, cwd=cwd)
    return proc.wait()


def _single_instance_lock():
    """단일 워치독 보장용 락 파일 생성 시도"""
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        return fd
    except FileExistsError:
        return None


def _release_instance_lock(fd):
    """락 해제"""
    try:
        os.close(fd)
    except Exception:
        pass
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except Exception:
        pass


def _acquire_watchdog_lock_with_takeover(timeout_sec=30):
    """takeover (새 --staged 인스턴스가 있으면 STOP_FILE을 생성하고, 락을 재시도하여 인수)"""
    start = time.monotonic()
    requested_stop = False
    while True:
        fd = _single_instance_lock()
        if fd is not None:
            try:
                if os.path.exists(STOP_FILE):
                    os.remove(STOP_FILE)
            except Exception:
                pass
            return fd

        if ('--staged' in sys.argv) and not requested_stop:
            try:
                Path(STOP_FILE).touch()
            except Exception:
                pass
            requested_stop = True

        if time.monotonic() - start > timeout_sec:
            return None
        time.sleep(0.5)


def _should_enable_watchdog():
    """실행파일이 설치 경로('D:\\AI Vision') 또는 '--staged'일 때만 자동 재실행(워치독) 동작"""
    return ('--staged' in sys.argv) or _is_install_location()


def run_watchdog_loop():
    """ 워치독 메인 루프 (자식 GUI를 감시 & 재시작 / exe 삭제 시 자동실행 일시정지 / 새 exe 등장 시 새 빌드가 수행)"""
    lock_fd = _acquire_watchdog_lock_with_takeover(timeout_sec=30)
    if lock_fd is None:
        return

    missing_since = None
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        while True:
            if os.path.exists(STOP_FILE):
                try:
                    os.remove(STOP_FILE)
                except Exception:
                    pass
                break

            # 설치본 부재 → 자식 종료 유도 + 스폰 중단(일시정지)
            if not _install_exe_available():
                try:
                    Path(DISABLE_FILE).touch()
                except Exception:
                    pass
                if missing_since is None:
                    missing_since = time.monotonic()
                time.sleep(1.5)
                continue

            # 설치 exe가 다시 등장 시, 새 바이너리를 1회 실행시켜 자동 인수 유도
            if missing_since is not None:
                try:
                    if os.path.exists(DISABLE_FILE):
                        os.remove(DISABLE_FILE)
                except Exception:
                    pass

                candidate = _find_candidate_install_exe()
                if candidate is not None:
                    try:
                        # 새 exe 1회 기동 → 자체 스테이징 & takeover
                        subprocess.Popen([str(candidate)], cwd=str(candidate.parent), creationflags=creationflags)
                    except Exception:
                        pass

                missing_since = None
                time.sleep(2.0)

            # 일반 루틴 자식이 종료되면 루프가 재시도/쿨다운
            _spawn_child_and_wait()
            time.sleep(10)
    finally:
        _release_instance_lock(lock_fd)

# =========================
# 이미지/CSV/그래프/이벤트 파일 경로 설정
# =========================
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

# =========================
# 파티클 탐지 파라미터
# =========================
MANUAL_THRESHOLD   = 7         # 이진화 임계값
KERNEL_SIZE_TOPHAT = (3, 3)    # tophat 모폴로지 연산 커널 크기
KERNEL_SIZE_MORPH  = (3, 3)    # 모폴로지 클로즈 연산 커널 크기
PARTICLE_AREA_MIN  = 8         # 파티클 최소 면적
PARTICLE_AREA_MAX  = 50        # 파티클 최대 면적

# 파티클 표기 원 파라미터 (유효/노이즈)
VALID_CIRCLE_RADIUS    = 25    # 유효(빨강) 원 반지름
VALID_CIRCLE_THICKNESS = 2     # 유효(빨강) 원 테두리 두께
NOISE_CIRCLE_RADIUS    = 10    # 노이즈(흰색) 원 반지름
NOISE_CIRCLE_THICKNESS = 1     # 노이즈(흰색) 원 테두리 두께

# 밝기 참조 영역 (ROI)
REF_ROI = (70, 90, 20, 20)

# 앵커 밝기 기반 동적 임계값 파라미터
ANCHOR_BRIGHTNESS_REF = 48
ADAPTIVE_THRESHOLD_GAIN = 0.3


def safe_image_load(image_path, max_retries=5, delay=0.2):
    """안전 이미지 로당 (재시도 최대 5회, 0.2초 간격)"""
    for _ in range(max_retries):
        try:
            img_array = np.fromfile(str(image_path), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
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


def particle_detection(image_path, exclude_boxes, threshold=None):
    """파티클 탐지 (OpenCV 파이프라인)"""
    img = safe_image_load(image_path)
    if img is None:
        print(f"❌ 오류: {image_path} 이미지 로딩 실패")
        return None, None, None
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE_TOPHAT)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        thr_value = MANUAL_THRESHOLD if threshold is None else int(threshold)
        thr_value = max(0, min(255, thr_value))
        _, binary_mask = cv2.threshold(tophat, thr_value, 255, cv2.THRESH_BINARY)
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_SIZE_MORPH))
        vis_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 제외영역(2개) 마스킹
        mask = np.ones_like(cleaned_mask) * 255
        for (x, y, w, h) in exclude_boxes:
            if w > 0 and h > 0:
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cv2.rectangle(mask, (x, y), (x+w, y+h), 0, -1)
        final_mask = np.bitwise_and(cleaned_mask, mask)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        particle_info = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            if PARTICLE_AREA_MIN <= area <= PARTICLE_AREA_MAX:
                particle_info.append((cx, cy, area))
        return gray, vis_img, particle_info
    except Exception as e:
        print(f"❌ 오류: {e}")
        return None, None, None


def find_new_images(img_dir, processed_set):
    """디렉토리 내 신규 이미지 탐색 (파일명 기준)"""
    all_imgs = sorted([p for p in img_dir.glob('*') if p.suffix.lower() in ['.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff']])
    return [p for p in all_imgs if p.name not in processed_set]


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
        # 독립 창 보장: 최상위 윈도우 플래그 지정
        self.setWindowFlag(Qt.Window, True)
        self.pixmap = None
        self.box1 = (0, 0, 0, 0)
        self.box2 = (0, 0, 0, 0)
        self.src_shape = None
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(170, 360)
        self.filename = ''
        self.noise_text = ""
        self.anchor_value = None

    def set_noise_text(self, text: str):
        """미리보기 하단에 표시할 Noise 정보 텍스트 설정"""
        self.noise_text = text or ""
        self.update()

    def set_anchor_value(self, value: int | None):
        """미리보기에 표시할 앵커 밝기 값 저장"""
        self.anchor_value = int(value) if value is not None else None
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
        if self.pixmap is not None and self.src_shape is not None:
            painter = QPainter(self)
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

            painter.end()


class BacklogFeeder(QObject):
    """백로그 전용 워커 (QThread)"""
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


# ============================================================
# 탐지 팝업(AlertPopup) - 요구사항 기반 신규 클래스 (독립 창)
# ============================================================
class AlertPopup(QWidget):
    closedWithAction = Signal(str, int)  # reason: 'ok'|'false', snooze_min

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Particle Alert')
        self.resize(1000, 720)  # "크게" 표시
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        # 상단 경고 타이틀
        self.title_label = QLabel('Particle이 탐지 되었습니다 !!')
        f = self.title_label.font()
        f.setPointSize(24)
        f.setBold(True)
        self.title_label.setFont(f)
        self.title_label.setAlignment(Qt.AlignCenter)

        # 우측 상단: 팝업 무시 옵션
        self.ignore_cb = QCheckBox('팝업 무시')
        self.ignore_min_spin = QSpinBox()
        self.ignore_min_spin.setRange(1, 120)
        self.ignore_min_spin.setValue(5)  # Default 5분
        self.ignore_suffix = QLabel('분간')
        small_font = self.ignore_cb.font()
        small_font.setPointSize(9)
        self.ignore_cb.setFont(small_font)
        self.ignore_min_spin.setFont(small_font)
        self.ignore_suffix.setFont(small_font)

        topbar = QHBoxLayout()
        topbar.addWidget(self.title_label, 1)
        topbar.addStretch(1)
        topbar.addWidget(self.ignore_cb)
        topbar.addWidget(self.ignore_min_spin)
        topbar.addWidget(self.ignore_suffix)

        # 기본 정보/시간/누적
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

        info_box = QVBoxLayout()
        info_box.addWidget(self.info_label)
        info_box.addWidget(self.time_label)
        info_box.addWidget(self.elapsed_label)
        info_box.addWidget(self.count_label)

        # 그래프 스냅샷
        self.graph_view = QLabel()
        self.graph_view.setFixedHeight(260)
        self.graph_view.setFrameShape(QFrame.Box)
        self.graph_view.setAlignment(Qt.AlignCenter)

        # 이미지 영역(가로 스크롤)
        self.img_container = QHBoxLayout()
        self.img_container.setSpacing(8)
        self.img_container.addStretch()
        self.img_widget = QWidget()
        self.img_widget.setLayout(self.img_container)
        self.img_scroll = QScrollArea()
        self.img_scroll.setWidgetResizable(True)
        self.img_scroll.setWidget(self.img_widget)
        self.img_scroll.setFixedHeight(260)

        # 로그 영역
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("QTextEdit{font-size:10pt}")
        self.log_edit.setFixedHeight(100)

        # 버튼
        self.btn_false = QPushButton('오탐지')
        self.btn_ok = QPushButton('확인')
        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        btn_bar.addWidget(self.btn_false)
        btn_bar.addWidget(self.btn_ok)

        # 메인 레이아웃
        main = QVBoxLayout(self)
        main.addLayout(topbar)
        main.addSpacing(6)
        main.addLayout(info_box)
        main.addSpacing(6)
        main.addWidget(QLabel('=============================================='))
        main.addWidget(QLabel('그래프'))
        main.addWidget(self.graph_view)
        main.addSpacing(6)
        main.addWidget(QLabel('이미지'))
        main.addWidget(self.img_scroll)
        main.addWidget(QLabel('=============================================='))
        main.addWidget(QLabel('로그 기록 (탐지/종료 등 주요 사항)'))
        main.addWidget(self.log_edit)
        main.addWidget(QLabel('=============================================='))
        main.addLayout(btn_bar)

        # 시그널 연결
        self.btn_ok.clicked.connect(lambda: self._close_with('ok'))
        self.btn_false.clicked.connect(lambda: self._close_with('false'))

        # 배경 깜빡임
        self._blink_red = False
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._toggle_blink)

        # 경과 시간
        self._first_dt = None
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)

        # 이미지 관리
        self._pix_labels: list[QLabel] = []

    def start_blink(self):
        self._blink_timer.start(500)  # 0.5초 주기

    def stop_blink(self):
        self._blink_timer.stop()
        self.setStyleSheet("")

    def _toggle_blink(self):
        self._blink_red = not self._blink_red
        if self._blink_red:
            self.setStyleSheet("background-color: rgba(255,30,30,0.25);")
        else:
            self.setStyleSheet("")

    def set_first_detect_time(self, qdt: QDateTime):
        self._first_dt = qdt
        self._elapsed_timer.start(1000)
        self._update_elapsed()

    def _update_elapsed(self):
        if not self._first_dt:
            self.elapsed_label.setText('[경과 시간] -')
            return
        secs = self._first_dt.secsTo(QDateTime.currentDateTime())
        if secs < 0:
            secs = 0
        mm = secs // 60
        ss = secs % 60
        self.elapsed_label.setText(f"[경과 시간] {mm:02d}분 {ss:02d}초")

    def set_info(self, base_info_text: str, when_text: str, count_text: str):
        self.info_label.setText(f"[기본 정보] Puller / Lot# / Process : {base_info_text}")
        self.time_label.setText(f"[발생 시각] {when_text}")
        self.count_label.setText(f"[누적 횟수] {count_text}")

    def set_graph_pixmap(self, pm: QPixmap):
        if pm is None:
            self.graph_view.clear()
            return
        self.graph_view.setPixmap(pm.scaled(self.graph_view.width(), self.graph_view.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def append_log(self, text: str):
        ts = time.strftime('%H:%M:%S')
        self.log_edit.append(f"[{ts}] {text}")

    def set_images(self, image_paths: list[Path]):
        # 기존 라벨 제거
        for lab in self._pix_labels:
            self.img_container.removeWidget(lab)
            lab.deleteLater()
        self._pix_labels.clear()

        # 최대 5장, 좌->우
        for p in image_paths[:5]:
            pix = self._load_pix(p)
            lab = QLabel()
            lab.setPixmap(pix)
            lab.setAlignment(Qt.AlignCenter)
            lab.setFixedSize(180, 240)
            lab.setFrameShape(QFrame.Box)
            self.img_container.insertWidget(self.img_container.count()-1, lab)
            self._pix_labels.append(lab)

    def _load_pix(self, p: Path) -> QPixmap:
        img = safe_image_load(str(p))
        if img is None:
            return QPixmap()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        qimg = QImage(img.data, w, h, 3*w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(180, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _close_with(self, reason: str):
        self.stop_blink()
        snooze_min = self.ignore_min_spin.value() if self.ignore_cb.isChecked() else 0
        self.closedWithAction.emit(reason, snooze_min)
        self.close()


# ============================================================
# 메인 GUI 위젯
# ============================================================
class ParticleDetectionGUI(QWidget):
    """메인 GUI 위젯"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Real-time Particle Detection GUI')
        self.resize(500, 400)
        self.settings = QSettings(str(INI_SETTINGS_PATH), QSettings.IniFormat)

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

        # Noise Particle 판정 파라미터
        self.prev_particles = []         # 2회 이상 반복 좌표 및 크기 기억
        self.frame_buffer = []           # 최근 10장 파티클 리스트 버퍼
        self.distance_threshold = 5      # 거리 임계값
        self.area_threshold = 3          # 크기 임계값
        self.collecting_initial = True   # 최초 10장 워밍업 플래그

        # 허용 모드 집합 & 래치/홀드 상태
        self.ALLOWED_PULLER_MODES = {"SET", "NECK"}  # 허용 모드
        self.disallowed_mode_latched = False  # 비허용 모드 래치
        self.status_hold_until = 0.0          # 상태 메시지 유지 만료 시각
        self.last_image_seen_at = 0.0         # 마지막 새 이미지 관측 시각
        self.no_new_image_grace_s = 9.0       # 신규 이미지 없음 표시는 9초 이후

        # 팝업/무시/누적
        self.alert_popup: AlertPopup | None = None
        self.popup_snooze_until = 0.0
        self.lot_event_count = 0
        self._popup_images = deque(maxlen=5)  # 최근 탐지 이미지 경로 (좌->우)
        self._popup_first_dt: QDateTime | None = None

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
        self.crop2_x.setValue(int(self.settings.value('crop2_x', 118)))
        self.crop2_y = QSpinBox(maximum=9999)
        self.crop2_y.setValue(int(self.settings.value('crop2_y', 110)))
        self.crop2_w = QSpinBox(maximum=9999)
        self.crop2_w.setValue(int(self.settings.value('crop2_w', 40)))
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

        crop_box = QGroupBox('제외 영역 설정')
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

        # 라이브 테일링 전용 타이머 (백로그 완료 후에만 사용)
        self.tail_timer = QTimer(self)
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

        # STOP/DISABLE 파일 감시 (업데이트/정지/삭제-일시정지 시 GUI 즉시 종료)
        self._stop_timer = QTimer(self)
        self._stop_timer.setInterval(1000)  # 1초마다
        self._stop_timer.timeout.connect(self._check_stop_or_disable_and_quit)
        self._stop_timer.start()

    # =========================
    # 내부 헬퍼
    # =========================
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
            return
        if self.prev_particles:
            text = "; ".join([f"[Noise] X:{x}, Y:{y}, Size:{a}" for (x, y, a) in self.prev_particles])
        else:
            text = "[Noise] None"
        self.preview_label.set_noise_text(text)

    def update_preview(self):
        """이미지 미리보기 / 상부 텍스트 갱신 (최신파일 기준)"""
        try:
            allow_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            cands = [p for p in IMG_INPUT_DIR.iterdir() if p.suffix.lower() in allow_ext]
            if not cands:
                self.preview_label.clear()
                return

            latest = max(cands, key=lambda p: p.stat().st_mtime)
            img = safe_image_load(str(latest))  # BGR 로드 (재시도 포함)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None
            if img is None:
                self.preview_label.clear()
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
            self.preview_label.clear()

    # =========================
    # 시작/중지/옵션 저장
    # =========================
    def start_realtime(self):
        """실시간 파티클 판정 시작 (백로그 → 라이브 테일 순)"""
        # 워밍업 초기화
        self.collecting_initial = True
        self.prev_particles = []
        self.frame_buffer = []
        self.saved_graph_lots.clear()

        self.show_noise_text = False
        self._update_noise_text()

        self.status_label.setText("초기 워밍업 중... (최초 10장 데이터 수집)")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

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
            self._backlog_thread.start()
            self.status_label.setText(f"백로그 처리 시작: {len(backlog)}장")
        else:
            # 백로그가 없으면 라이브 감시 시작
            self.tail_timer.start(200)
            self.status_label.setText("백로그 없음. 라이브 감시 시작.")
        # 초기 Noise 텍스트
        self._update_noise_text()

    def stop_realtime(self):
        """실시간 파티클 판정 중지"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.tail_timer.isActive():
            self.tail_timer.stop()
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
        """crop 영역 설정값 저장"""
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

        self.settings.sync()
        self.status_label.setText("설정 저장됨.")

    # =========================
    # 라이브 테일링/백로그 완료
    # =========================
    def process_new_images_tick(self):
        """(실시간)신규 이미지 처리"""
        new_images = find_new_images(IMG_INPUT_DIR, self.session_processed_images)
        if not new_images:
            # 상태 메세지 깜빡임 방지 (래치/홀드/유예시간 고려)
            now = time.monotonic()
            if self.disallowed_mode_latched:
                return
            if now < self.status_hold_until:
                return
            if (now - self.last_image_seen_at) <= self.no_new_image_grace_s:
                return
            self._set_status("신규 이미지 없음 (폴더 감시 중)")
            return
        for p in new_images:
            self.process_single_image(p)

    @Slot()
    def _on_backlog_finished(self):
        """백로그 완료 콜백 → 라이브 감시 시작"""
        self.status_label.setText("백로그 완료. 라이브 감시 시작.")
        # 최신 샘플 1장으로 미리보기 갱신
        self.update_preview()
        self.tail_timer.start(200)

    # =========================
    # 메인 처리 루틴
    # =========================
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
                self.csv_path = CSV_OUTPUT_DIR / f"{lot_number}.csv"
                self.current_attempt = attempt_value
                self.processed_images.clear()
                self.lot_event_count = 0  # 누적 횟수 리셋
                self._popup_images.clear()
                self._popup_first_dt = None
                if self.alert_popup:
                    self.alert_popup.close()
                    self.alert_popup = None

                # 새 CSV 생성(헤더만)
                try:
                    if (not self.csv_path.exists()) or (self.csv_path.stat().st_size == 0):
                        with open(self.csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                            writer = csv.writer(f)
                            writer.writerow(self.csv_header)
                except Exception as e:
                    self._set_status(f"CSV 초기화 오류: {e}", hold_s=3.0)

                # 기존 CSV가 있으면 중복 방지 세트 재로딩
                if self.csv_path.exists():
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
                self._reset_noise_warmup("LOT 변경 감지 → 워밍업 재시작 (10장)")

                # 빈 CSV 기반 그래프 즉시 초기화
                self.update_graph()

            else:
                if self.current_attempt is None:
                    self.current_attempt = attempt_value
                elif self.current_attempt != attempt_value:
                    self.current_attempt = attempt_value
                    self._reset_noise_warmup("Attempt 변경 감지 → 워밍업 재시작 (10장)")

            # Lot 전환 이후 CSV 기반 중복 세트 로딩이 완료된 상태에서 중복 여부 재확인
            if img_name in self.processed_images:
                self.status_label.setText(f"이미 처리됨: {img_path.name}")
                return

            # 새 이미지 관측 시각 갱신
            self.last_image_seen_at = time.monotonic()

            self.device_info_box.setText('_'.join([info['equip'], info['lot'], info['process'], str(info['attempt'])]))

            # 허용 모드 체크
            if process_name not in self.ALLOWED_PULLER_MODES:
                # 비허용 모드 → 래치 온 (상태 고정)
                self.disallowed_mode_latched = True
                self.show_noise_text = False
                self._update_noise_text()
                self._set_status("지정된 Puller Mode Status가 아님.")

                # 미리보기는 최신 이미지로 갱신하되 앵커값은 숨김
                img_preview = safe_image_load(str(img_path))
                if img_preview is not None:
                    gray_prev = cv2.cvtColor(img_preview, cv2.COLOR_BGR2GRAY)
                    self.preview_label.show_image(
                        gray_prev, [self.get_box1(), self.get_box2()],
                        filename=img_path.name
                    )
                    self.preview_label.set_anchor_value(None)

                self._mark_image_processed(img_path.name)
                return
            else:
                # 허용 모드 → 래치 해제
                if self.disallowed_mode_latched:
                    self.disallowed_mode_latched = False

            # 현재 처리 대상 파일 기준으로 이미지 미리보기 표기 + 앵커값
            img_preview = safe_image_load(str(img_path))
            if img_preview is not None:
                gray_prev = cv2.cvtColor(img_preview, cv2.COLOR_BGR2GRAY)
                anchor_current = compute_anchor_brightness(gray_prev)
                self.preview_label.show_image(gray_prev, [self.get_box1(), self.get_box2()], filename=img_path.name)
                self.preview_label.set_anchor_value(anchor_current)
            else:
                self.preview_label.set_anchor_value(None)

            # 현재 Noise 정보 텍스트 갱신
            adaptive_value = compute_adaptive_threshold(anchor_current)
            self._update_noise_text()

            # 워밍업 구간 (최초 10장 동안은 '판정/CSV/그래프' 수행하지 않음)
            if self.collecting_initial:
                self.status_label.setText(f"초기 워밍업 수집 중... ({len(self.frame_buffer) + 1} / 10)")
                self.show_noise_text = False
                self._update_noise_text()

                # 후보 파티클만 수집(판정/CSV/그래프/이벤트 저장 없음)
                _, _, particle_info = particle_detection(str(img_path), [self.get_box1(), self.get_box2()], adaptive_value)
                if particle_info is None:
                    # 로딩/후보 추출 실패 시 스킵
                    self.status_label.setText(f"오류: {img_path.name} 로딩/후보 추출 실패 (워밍업)")
                    self._mark_image_processed(img_path.name)
                    return

                self.frame_buffer.append(particle_info)

                if len(self.frame_buffer) >= 10:
                    # (x,y,area) 빈도 집계: 2회 이상 등장 좌표를 노이즈 기준으로 채택
                    freq = {}
                    for frame_particles in self.frame_buffer:
                        for (x, y, area) in frame_particles:
                            key = (x, y, area)
                            freq[key] = freq.get(key, 0) + 1
                    self.prev_particles = [k for k, v in freq.items() if v >= 2] or (self.frame_buffer[-1] if self.frame_buffer else [])

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
            gray, overlay_img, particle_info = particle_detection(str(img_path), [self.get_box1(), self.get_box2()], adaptive_value)
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
                noise_text = "[Noise] None" if not self.prev_particles else "; ".join(
                    [f"[Noise] X:{x}, Y:{y}, Size:{a}" for (x, y, a) in self.prev_particles]
                )
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
                    self._push_popup_image(event_path)
                else:
                    self._maybe_open_or_update_popup(info, event_path)

            # 그래프 갱신
            self.update_graph()

            # 처리 완료 → 3초간 상태 홀드로 깜빡임 억제
            self._set_status(f"처리됨: {img_path.name}  - 파티클:{has_particle}", hold_s=3.0)
            self._mark_image_processed(img_path.name)

        except Exception as e:
            self.status_label.setText(f"오류: {e}")

    # =========================
    # CSV & 그래프
    # =========================
    def save_csv(self, row, max_retries=5, delay=0.2):
        """결과 CSV 저장"""
        if not self.csv_path:
            return
        attempt = 0
        while attempt < max_retries:
            try:
                header_exists = self.csv_path.exists()
                with open(self.csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    if not header_exists:
                        writer.writerow(self.csv_header)
                    writer.writerow(row)
                return
            except Exception:
                attempt += 1
                time.sleep(delay)
        print("❌ CSV 저장 최종 실패 - 데이터가 저장되지 않았습니다.")

    def update_graph(self):
        """파티클 그래프 갱신 (Attempt 밴딩 지원)"""
        if not getattr(self, "csv_path", None) or not self.csv_path.exists():
            self.plot_canvas.update_plot([], [], latest_index=None)
            return
        try:
            df = pd.read_csv(self.csv_path)
        except Exception:
            self.plot_canvas.update_plot([], [], latest_index=None)
            return
        if '파티클_크기' in df.columns:
            df['파티클_크기'] = pd.to_numeric(df['파티클_크기'], errors='coerce').fillna(0)
        else:
            df['파티클_크기'] = 0
        attempt_list = None
        if 'Attempt 횟수' in df.columns:
            attempt_list = (
                pd.to_numeric(df['Attempt 횟수'], errors='coerce')
                  .ffill()
                  .fillna(0)
                  .astype(int)
                  .to_numpy()
            )
        latest_index = len(df) - 1 if len(df) > 0 else None
        self.plot_canvas.update_plot(
            df['이미지 파일명'],
            df['파티클_크기'],
            latest_index,
            attempt_list=attempt_list
        )

    # =========================
    # 팝업/그래프 스냅샷/무시 로직
    # =========================
    def _push_popup_image(self, path: Path):
        # 발생 시점 순서대로(좌->우) 유지, 최대 5장
        self._popup_images.append(path)

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
        return qdt.toString('yy/MM/dd_HH:mm')

    def _graph_pixmap(self) -> QPixmap:
        # 현재 Matplotlib Figure를 PNG로 렌더링 후 QPixmap 반환
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
        # 독립된 최상위 창으로 생성(메인 GUI와 별개 윈도우)
        if self.alert_popup is None:
            self.alert_popup = AlertPopup(None)
            self.alert_popup.setWindowModality(Qt.NonModal)
            self.alert_popup.closedWithAction.connect(self._on_popup_closed)
        return self.alert_popup

    def _maybe_open_or_update_popup(self, info: dict, event_image_path: Path):
        # 팝업 이미지 큐 갱신
        self._push_popup_image(event_image_path)
        pop = self._ensure_popup()

        # 최초 발생 시각(팝업 세션 기준)
        if self._popup_first_dt is None:
            self._popup_first_dt = self._parse_info_datetime(info)
        first_dt = self._popup_first_dt

        # 머릿글/정보/그래프/이미지 세팅
        base_info_text = info['stem']  # 예: A6_WY6NQ_NECK_1
        when_text = self._when_text(first_dt)
        count_text = f"{self.lot_event_count}회"

        pop.set_info(base_info_text, when_text, count_text)
        pop.set_first_detect_time(first_dt)
        pop.set_graph_pixmap(self._graph_pixmap())
        pop.set_images(list(self._popup_images))
        pop.append_log(f"탐지 반영: {event_image_path.name}")
        pop.start_blink()

        # 이미 떠 있으면 내용만 갱신, 없으면 표시
        if not pop.isVisible():
            pop.show()
            pop.raise_()
            pop.activateWindow()

    @Slot(str, int)
    def _on_popup_closed(self, reason: str, snooze_min: int):
        # 팝업 종료 사유에 따른 로그/무시 설정
        if snooze_min > 0:
            self.popup_snooze_until = time.monotonic() + snooze_min * 60
        else:
            self.popup_snooze_until = 0.0
        if reason == 'false':
            # 오탐지 로그만 (데이터 롤백은 요구사항에 명시X)
            pass
        # 팝업 세션 초기화 (다음 탐지 시 새 세션으로 경과 시간/이미지 시작)
        self._popup_images.clear()
        self._popup_first_dt = None
        self.alert_popup = None

    # =========================
    # 기타
    # =========================
    def _check_stop_or_disable_and_quit(self):
        try:
            if os.path.exists(STOP_FILE) or os.path.exists(DISABLE_FILE):
                QApplication.quit()
        except Exception:
            pass


# ============================================================
# 진입점
# ============================================================
def main():
    if ('--child' in sys.argv) or (not _should_enable_watchdog()):
        app = QApplication(sys.argv)
        gui = ParticleDetectionGUI()
        gui.show()
        if gui.auto_start_on_launch:
            gui.start_realtime()
        sys.exit(app.exec())
    else:
        _maybe_stage_to_temp()
        run_watchdog_loop()


if __name__ == '__main__':
    main()
