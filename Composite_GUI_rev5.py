import os
import sys
from pathlib import Path
from PIL import Image, ImageQt
from datetime import datetime, timedelta
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout, QSpinBox,
    QSlider, QCheckBox, QGroupBox, QGridLayout, QLineEdit, QSizePolicy
)
from PySide6.QtCore import QThread, Signal, Qt, QSettings, QTimer
from PySide6.QtGui import QPixmap, QPainter, QColor, QPen
import socket
import csv
import io
from smb.SMBConnection import SMBConnection
import traceback
import time
import subprocess
import tempfile
import shutil

STOP_FILE = os.path.join(tempfile.gettempdir(), "CompositeImageApp.stop")
LOCK_FILE = os.path.join(tempfile.gettempdir(), "CompositeImageApp.watchdog.lock")
DISABLE_FILE = os.path.join(tempfile.gettempdir(), "CompositeImageApp.disabled")
RUN_BASE_DIR = Path(tempfile.gettempdir()) / "CompositeImageApp_run"


def _exe_path():
    if getattr(sys, "frozen", False):
        return sys.executable
    return os.path.abspath(sys.argv[0])


def _is_install_location():
    """현재 실행 위치가 설치 경로(D:\AI Vision)인지 여부 판정"""
    exe_dir = os.path.normpath(os.path.dirname(_exe_path()))
    target_dir = os.path.normpath(r'D:\AI Vision')
    return exe_dir.upper() == target_dir.upper()


def _get_app_cwd():
    """자식/워치독이 사용할 작업 디렉터리(cwd) 반환"""
    return os.environ.get('COMPOSITE_APP_CWD') or os.path.dirname(_exe_path())


def _install_exe_available():
    """설치 경로에 실행 후보(exe)의 존재 여부 확인"""
    try:
        install_dir = Path(_get_app_cwd())
        if not install_dir.exists():
            return False
        for p in install_dir.glob("*.exe"):
            if p.name.lower().startswith("composite"):
                return True
        return False
    except Exception:
        return False


def _find_candidate_install_exe():
    """설치 폴더에서 최신 실행 후보 exe 반환"""
    install_dir = Path(_get_app_cwd())
    if not install_dir.exists():
        return None
    exes = [p for p in install_dir.glob("*.exe") if p.name.lower().startswith("composite")]
    if not exes:
        return None
    return max(exes, key=lambda p: p.stat().st_mtime)


def _maybe_stage_to_temp():
    """exe 파일 실행 시, 자신을 TEMP로 복사 / '--staged' 로 재실행 / 원본은 즉시 종료(파인 삭제/교체 가능)"""
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
        env['COMPOSITE_APP_CWD'] = os.path.dirname(_exe_path())
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        subprocess.Popen([str(staged_exe), '--staged'], cwd=env['COMPOSITE_APP_CWD'], env=env, creationflags=creationflags)
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
    """실행파일이 설치 경로('D:\AI Vision') 또는 '--staged'일 때만 자동 재실행(워치독) 동작"""
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
                # 자동실행 재개: DISABLE 해제
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


# ======== SMB 연결 정보 (카메라PC의 호스트명으로 장비PC와의 네트워크 자동 설정) ========
host_name = socket.gethostname()
if host_name.endswith('C'):
    username = host_name[:-1]
else:
    username = host_name
lookup_username = username.upper()

# username-IP 매핑 리스트
equipment_ip_map = {
    "PU-X1": "172.29.136.61" , "PU-X2": "172.29.136.62" , "PU-X3": "172.29.136.63" , "PU-X4": "172.29.136.64" ,
    "PU-X5": "172.29.136.65" , "PU-X6": "172.29.136.66" , "PU-X7": "172.29.136.67" , "PU-X8": "172.29.136.68" ,
    "PU-X9": "172.29.136.69" , "PU-Y0": "172.29.136.70" , "PU-Y1": "172.29.136.71" , "PU-Y2": "172.29.136.72" ,
    "PU-Y3": "172.29.136.73" , "PU-Y4": "172.29.136.74" , "PU-Y5": "172.29.136.75" , "PU-Y6": "172.29.136.76" ,
    "PU-Y7": "172.29.136.77" , "PU-Y8": "172.29.136.78" , "PU-Y9": "172.29.136.79" , "PU-Z0": "172.29.136.80" ,
    "PU-Z1": "172.29.136.81" , "PU-Z2": "172.29.136.82" , "PU-Z3": "172.29.136.83" , "PU-Z4": "172.29.136.84" ,
    "PU-Z5": "172.29.136.85" , "PU-Z6": "172.29.136.86" , "PU-Z7": "172.29.136.87" , "PU-Z8": "172.29.136.88" ,
    "PU-Z9": "172.29.136.89" , "PU-A0": "172.29.136.90" , "PU-A1": "172.29.136.91" , "PU-A2": "172.29.136.92" ,
    "PU-A3": "172.29.136.93" , "PU-A4": "172.29.136.94" , "PU-A5": "172.29.136.95" , "PU-A6": "172.29.136.96" ,
    "PU-A7": "172.29.136.97" , "PU-A8": "172.29.136.98" , "PU-A9": "172.29.136.99" , "PU-B0": "172.29.136.100",
    "PU-B1": "172.29.136.101", "PU-B2": "172.29.136.102", "PU-B3": "172.29.136.103", "PU-B4": "172.29.136.104",
    "PU-B5": "172.29.136.105", "PU-B6": "172.29.136.106", "PU-B7": "172.29.136.107", "PU-B8": "172.29.136.108",
    "PU-B9": "172.29.136.109", "PU-C0": "172.29.136.110", "PU-C1": "172.29.136.111", "PU-C2": "172.29.136.112",
    "PU-C3": "172.29.136.113", "PU-C4": "172.29.136.114", "PU-C5": "172.29.136.115", "PU-C6": "172.29.136.116",
    "PU-C7": "172.29.136.117", "PU-C8": "172.29.136.118", "PU-C9": "172.29.136.119", "PU-D0": "172.29.136.120",
    "PU-D1": "172.29.136.121", "PU-D2": "172.29.136.122", "PU-R1": "172.28.37.201" , "PU-R2": "172.28.37.202" ,
    "PU-R3": "172.28.37.203" , "PU-R4": "172.28.37.204" , "PU-R5": "172.28.37.205" , "PU-R6": "172.28.37.206" ,
    "PU-R7": "172.28.37.207" , "PU-R8": "172.28.37.208" , "PU-R9": "172.28.37.209" , "PU-S0": "172.28.37.210" ,
    "PU-S1": "172.28.37.211" , "PU-S2": "172.28.37.212" , "PU-S3": "172.28.37.213" , "PU-S4": "172.28.37.214" ,
    "PU-S5": "172.28.37.215" , "PU-S6": "172.28.37.216" , "PU-S7": "172.28.37.217" , "PU-S8": "172.28.37.218" ,
    "PU-S9": "172.28.37.219" , "PU-T0": "172.28.37.220" , "PU-T1": "172.28.37.221" , "PU-T2": "172.28.37.222" ,
    "PU-T3": "172.28.37.223" , "PU-T4": "172.28.37.224" , "PU-T5": "172.28.37.225" , "PU-T6": "172.28.37.226" ,
    "PU-T7": "172.28.37.227" , "PU-T8": "172.28.37.228" , "PU-T9": "172.28.37.229" , "PU-U0": "172.28.37.230" ,
    "PU-U1": "172.28.37.231" , "PU-U2": "172.28.37.232" , "PU-U3": "172.28.37.233" , "PU-U4": "172.28.37.234" ,
    "PU-U5": "172.28.37.235" , "PU-U6": "172.28.37.236" , "PU-U7": "172.28.37.237" , "PU-U8": "172.28.37.238" ,
    "PU-U9": "172.28.37.239" , "PU-V0": "172.28.37.240" , "PU-V1": "172.28.37.241" , "PU-V2": "172.28.37.242" ,
    "PU-V3": "172.28.37.243" , "PU-V4": "172.28.37.244" , "PU-V5": "172.28.37.245" , "PU-V6": "172.28.37.246" ,
    "PU-V7": "172.28.37.247" , "PU-V8": "172.28.37.248" , "PU-V9": "172.28.37.249" , "PU-W0": "172.28.37.250" ,
    "PU-W1": "172.28.37.251" , "PU-W2": "172.28.37.252"
}

server_ip = equipment_ip_map.get(lookup_username, None)

password = 'smin'                  # 계정 비밀번호
client_name = host_name            # 카메라PC 네트워크(호스트) 이름
server_name = username             # 장비PC 네트워크(호스트) 이름
domain = ''                        # 도메인이 없으면 빈 문자열
share_name = 'data'                # 설정 IP의 하위 폴더명
# =============================


def maintain_max_files(output_folder, max_files):
    """저장 폴더 내 파일이 설정 값(장) 초과 시 오래된 파일 삭제"""
    files = list(output_folder.glob('*.*'))
    if len(files) <= max_files:
        return
    files.sort(key=lambda f: f.stat().st_mtime)
    to_delete = len(files) - max_files
    for f in files[:to_delete]:
        try:
            f.unlink()
        except Exception as e:
            print(f'파일 삭제 실패: {f} - {e}')


def find_latest_csv_file():
    """SMB에서 가장 마지막으로 수정된 CSV 파일명을 반환"""
    conn = None
    last_exception = None
    for port_try in [139, 445]:
        try:
            is_direct_tcp = (port_try == 445)
            conn = SMBConnection(username, password, client_name, server_name,
                                 domain=domain, use_ntlm_v2=True, is_direct_tcp=is_direct_tcp)
            connected = conn.connect(server_ip, port_try)
            if connected:
                files = conn.listPath(share_name, '/')
                csv_files = [file for file in files if file.filename.lower().endswith('.csv')]
                if not csv_files:
                    conn.close()
                    return None
                latest_file = max(csv_files, key=lambda f: f.last_write_time)
                conn.close()
                return latest_file.filename
            else:
                conn.close()
        except Exception:
            last_exception = traceback.format_exc()
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    return None


def retrieve_csv_text(filename):
    """SMB에서 지정한 CSV 파일 내용을 cp949 인코딩으로 읽어 반환"""
    conn = None
    last_exception = None
    for port_try in [139, 445]:
        try:
            is_direct_tcp = (port_try == 445)
            conn = SMBConnection(username, password, client_name, server_name,
                                 domain=domain, use_ntlm_v2=True, is_direct_tcp=is_direct_tcp)
            connected = conn.connect(server_ip, port_try)
            if connected:
                file_obj = io.BytesIO()
                conn.retrieveFile(share_name, filename, file_obj)
                file_obj.seek(0)
                csv_text = file_obj.read().decode('cp949')
                conn.close()
                return csv_text
            else:
                conn.close()
        except Exception:
            last_exception = traceback.format_exc()
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
    return None


def parse_datetime(date_str, time_str):
    """CSV의 'Date'와 'Time' 칼럼 값을 활용하여 datetime 객체로 변환"""
    try:
        dt = datetime.strptime(date_str.strip(), '%Y-%m-%d')
        date_part = dt.strftime('%Y%m%d')
    except Exception:
        date_part = date_str.replace('-', '')
    try:
        time_lower = time_str.lower()
        is_pm = '오후' in time_lower or 'pm' in time_lower
        is_am = '오전' in time_lower or 'am' in time_lower
        time_clean = time_str.replace('오전', '').replace('오후', '').replace('AM', '').replace('PM', '').strip()
        dt_time = datetime.strptime(time_clean, '%I:%M:%S')
        hour = dt_time.hour
        if is_pm and hour < 12:
            hour += 12
        if is_am and hour == 12:
            hour = 0
        dt_full = datetime.strptime(date_part, '%Y%m%d').replace(
            hour=hour, minute=dt_time.minute, second=dt_time.second)
        return dt_full
    except Exception:
        return None


def find_closest_row(csv_text, target_dt):
    """주어진 CSV 텍스트에서 target_dt에 가장 가까운 날짜/시간을 가진 row를 반환"""
    f = io.StringIO(csv_text)
    reader = csv.DictReader(f)
    min_diff = timedelta.max
    selected_row = None
    for row in reader:
        row_date = row.get('Date', '')
        row_time = row.get('Time', '')
        row_dt = parse_datetime(row_date, row_time)
        if row_dt is None:
            continue
        diff = abs(target_dt - row_dt)
        if diff < min_diff:
            min_diff = diff
            selected_row = row
    return selected_row, None


def extract_row_info(row):
    """CSV에서 장비명, Lot#, Status, 날짜/시간 등의 정보 추출 후 딕셔너리로 반환"""
    if row is None:
        return {
            "date_time": "",
            "equipment": "",
            "lot_number": "",
            "status_display": "",
            "status_text": ""
        }

    date_str = row.get('Date', '')
    time_str = row.get('Time', '')
    run_number = row.get('Run Number', '')
    puller_status_raw = row.get('Puller Mode Status', '')
    neck_attempt_raw = row.get('Neck Attempts', '')
    bottom_heater_raw = row.get('Bottom Heater Set Point', '')
    seed_lift_raw = row.get('Seed Lift Set Point', '')

    dt_full = parse_datetime(date_str, time_str)
    if dt_full:
        date_time = dt_full.strftime('%Y%m%d_%H%M%S')
    else:
        date_time = ''

    equipment = host_name[3:5]  # host_name에서 4~5번째 글자로 장비명 추출
    lot_number = run_number

    puller_status_str = puller_status_raw.strip() if isinstance(puller_status_raw, str) else str(puller_status_raw)
    try:
        puller_status = int(float(puller_status_str))
    except (ValueError, TypeError):
        puller_status = None

    try:
        neck_attempt = int(float(neck_attempt_raw))
    except (ValueError, TypeError):
        neck_attempt = 0

    try:
        bottom_heater = float(bottom_heater_raw)
    except (ValueError, TypeError):
        bottom_heater = 0

    try:
        seed_lift = float(seed_lift_raw)
    except (ValueError, TypeError):
        seed_lift = 0

    status_map = {
        0: "IDLE",
        4: "TAKEOVER",
        5: "PUMPDOWN",
        90: "PULLOUT",
        94: "POST PULLOUT"
    }

    if puller_status is not None:
        if puller_status in status_map:
            status_text = status_map[puller_status]
        elif 10 <= puller_status <= 19:
            status_text = "MELTDOWN"
        elif 20 <= puller_status <= 29:
            status_text = "STABILIZATION"
        elif 30 <= puller_status <= 39:
            # [mode 30~39 조건 下] ② BH Power≥10 → "REMELT" / ② Neck Att≥1 & Seed Lift≥0.1 → "NECK" / ③ 아니면 "NECK(STAB)"
            if bottom_heater >= 10:
                status_text = "REMELT"
            else:
                status_text = "NECK" if (neck_attempt >= 1 and seed_lift >= 0.1) else "NECK(STAB)"
        elif 40 <= puller_status <= 49:
            status_text = "CROWN"
        elif 50 <= puller_status <= 59:
            status_text = "SHOULDER"
        elif 60 <= puller_status <= 69:
            status_text = "BODY"
        elif 70 <= puller_status <= 79:
            status_text = "TAIL"
        elif 80 <= puller_status <= 89:
            status_text = "SHUTDOWN"
        else:
            status_text = f"알 수 없는 상태({puller_status})"

        status_display = f"{puller_status} / {status_text}"
    else:
        status_display = f"{puller_status_str} / 상태 정보 없음"

    return {
        "date_time": date_time,
        "equipment": equipment,
        "lot_number": lot_number,
        "status_display": status_display,
        "status_text": status_text
    }


# Attempt_NEW 계산 (온라인/스트리밍)
INTEREST_STATUSES = ("NECK", "CROWN", "SHOULDER", "BODY")


def find_closest_row_and_attempt(csv_text, target_dt):
    """Attempt_NEW를 온라인 방식으로 계산하여 반환"""
    f = io.StringIO(csv_text)
    reader = csv.DictReader(f)

    min_diff = timedelta.max
    selected_row = None
    selected_attempt = 0
    selected_info = None

    counters = {s: 0 for s in INTEREST_STATUSES}
    prev_status = None
    prev_lot = None

    for row in reader:
        info = extract_row_info(row)
        lot = info.get("lot_number", "")
        status_text = info.get("status_text", "")

        # 동일 csv 내에서 Lot# 변경 시 카운터 리셋
        if lot and lot != prev_lot:
            counters = {s: 0 for s in INTEREST_STATUSES}
            prev_status = None
            prev_lot = lot

        # Attempt_NEW 계산 (설정된 Status 진입 시 이전 row와 다르면 해당 상태 카운터 +1)
        if status_text in counters:
            if prev_status != status_text:
                counters[status_text] += 1
            attempt_now = counters[status_text]
        else:
            attempt_now = 0

        row_dt = parse_datetime(row.get('Date', ''), row.get('Time', ''))
        if row_dt is None:
            prev_status = status_text
            continue

        diff = abs(target_dt - row_dt)
        if diff < min_diff:
            min_diff = diff
            selected_row = row
            selected_attempt = attempt_now
            selected_info = info

        prev_status = status_text

    return selected_row, selected_attempt, selected_info


class ImageSaverThread(QThread):
    """이미지 저장 및 CSV 정보를 활용, 주기적으로 이미지 파일을 저장하는 쓰레드"""

    update_status = Signal(str)
    update_preview = Signal()
    update_device_name = Signal(str)

    def __init__(self, source_file, output_folder, crop_rect, get_format, get_quality, get_interval, get_device_name, get_delete_interval_hours):
        """이미지 저장에 필요한 파라미터(파일 경로, 형식, 크롭 영역 등)를 받으며 쓰레드 객체 초기화"""
        super().__init__()
        self.source_file = source_file
        self.output_folder = output_folder
        self.crop_rect = crop_rect
        self.get_format = get_format
        self.get_quality = get_quality
        self.get_interval = get_interval
        self.get_device_name = get_device_name
        self.get_delete_interval_hours = get_delete_interval_hours
        self.running = False
        self.last_delete_time = None

    def run(self):
        """이미지 저장 루프(Interval 이미지 파일 로드/크롭/저장, 최신 CSV 파일에서 이미지 저장 시각과 가장 가까운 row를 찾아 파일명에 반영)"""
        self.running = True
        while self.running:
            loop_start = time.monotonic()

            if not self.output_folder.exists():
                try:
                    self.output_folder.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    self.update_status.emit(f'폴더 생성 실패: {e}')
                    return

            if self.source_file.exists():
                try:
                    with Image.open(self.source_file) as img:
                        img = img.convert('L')
                        cx, cy, cw, ch = self.crop_rect()
                        img_w, img_h = img.size
                        if (cx + cw > img_w) or (cy + ch > img_h):
                            self.update_status.emit(f'Crop 영역({cx},{cy},{cw},{ch})이 원본 이미지({img_w},{img_h})보다 큽니다.')
                            return
                        cropped = img.crop((cx, cy, cx+cw, cy+ch))

                        formats = self.get_format()
                        now = datetime.now()
                        dt_str = now.strftime("%Y%m%d_%H%M%S")

                        latest_csv_fname = find_latest_csv_file()
                        csv_row_info = {
                            "equipment": "",
                            "lot_number": "",
                            "status_display": "",
                            "date_time": "",
                            "status_text": ""
                        }
                        attempt_new = 0
                        if latest_csv_fname:
                            csv_text = retrieve_csv_text(latest_csv_fname)
                            if csv_text:
                                # 현 시각과 가장 가까운 row의 정보 획득
                                closest_row, attempt_new, csv_row_info = find_closest_row_and_attempt(csv_text, now)
                            else:
                                csv_row_info["status_display"] = "CSV 내용 없음"
                        else:
                            csv_row_info["status_display"] = "CSV 파일 없음"

                        device_name_full = f"{host_name} / {latest_csv_fname.rsplit('.', 1)[0] if latest_csv_fname else 'N/A'}"
                        self.update_device_name.emit(device_name_full)

                        device_name_for_fname = host_name[3:5]
                        lot_for_fname = csv_row_info.get("lot_number", "") or "LotUnknown"
                        status_for_fname = (csv_row_info.get("status_text", "") or "StatusUnknown").replace(" ", "_").replace("/", "-")
                        attempt_tag_for_fname = f"{attempt_new}"
                        base_fname = f"{device_name_for_fname}_{lot_for_fname}_{status_for_fname}_{attempt_tag_for_fname}_{dt_str}"

                        if "BMP" in formats:
                            fname = f"{base_fname}.bmp"
                            save_path = self.output_folder / fname
                            cropped.save(save_path, format="BMP")
                            self.update_status.emit(f'저장: {fname}')
                        if "JPEG" in formats:
                            fname = f"{base_fname}.jpg"
                            save_path = self.output_folder / fname
                            cropped.save(save_path, format="JPEG", quality=self.get_quality())
                            self.update_status.emit(f'저장: {fname}')

                        # 유지 개수(삭제) 정책: 주기적으로 오래된 파일 삭제
                        maintain_max_files(self.output_folder, self.get_delete_interval_hours())

                        now_time = datetime.now()
                        current_delete_interval_hours = self.get_delete_interval_hours()
                        if (self.last_delete_time is None or
                                (now_time - self.last_delete_time).total_seconds() >= current_delete_interval_hours * 3600):
                            maintain_max_files(self.output_folder, self.get_delete_interval_hours())
                            self.last_delete_time = now_time

                    self.update_preview.emit()
                except Exception as e:
                    self.update_status.emit(f'저장 실패: {e}')
            else:
                self.update_status.emit('소스 파일이 존재하지 않음!')

            # 고정 레이트 수면: 처리시간 포함해 간격 맞추기
            target = float(self.get_interval())
            elapsed = time.monotonic() - loop_start
            sleep_left = max(0.0, target - elapsed)

            # 100ms 단위로 잘게 잠자면서 종료 신호 체크
            ticks = int(sleep_left / 0.1)
            for _ in range(ticks):
                if not self.running:
                    return
                self.msleep(100)
            rem_ms = int((sleep_left - ticks * 0.1) * 1000)
            if rem_ms > 0:
                if not self.running:
                    return
                self.msleep(rem_ms)

    def stop(self):
        """이미지 저장 루프 종료"""
        self.running = False


class PreviewLabel(QLabel):
    """QLabel: 이미지 미리보기(크롭 영역 표시) 클래스"""

    def __init__(self, *args, **kwargs):
        """QLabel 기본설정 및 crop 정보 준비"""
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignCenter)
        self.crop_rect = (0, 0, 100, 100)
        self.img = None

    def set_image(self, pil_img, crop_rect=None):
        """PIL 이미지를 QLabel에 세팅 및 crop 정보 갱신"""
        self.img = pil_img.convert('L')
        if crop_rect:
            self.crop_rect = crop_rect
        self.update()

    def paintEvent(self, event):
        """QLabel에 이미지와 crop 영역 사각형 그림"""
        super().paintEvent(event)
        if self.img:
            qimg = ImageQt.ImageQt(self.img)
            pixmap = QPixmap.fromImage(qimg)
            w = self.width()
            h = self.height()
            scaled = pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_w, img_h = self.img.size
            scale = min(w / img_w, h / img_h)
            offset_x = (w - img_w * scale) / 2
            offset_y = (h - img_h * scale) / 2
            painter = QPainter(self)
            painter.drawPixmap(int(offset_x), int(offset_y), scaled)
            cx, cy, cw, ch = self.crop_rect
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(
                int(offset_x + cx * scale),
                int(offset_y + cy * scale),
                int(cw * scale),
                int(ch * scale)
            )
            painter.end()


class MainWindow(QWidget):
    """메인 GUI 윈도우 클래스 (설정 로드/저장, 사용자 입력 위젯, 이미지 저장 스레드 관리 등 주요 UI 관리)"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Composite Image Saver')
        self.setFixedSize(600, 290)

        USE_TEST_MODE = False   # 이미지(composite) 로드 경로 (True: Test용 / False: 실제 장비 적용)
        INI_TEST_MODE = False   # 설정 파일(ini) 저장 경로 (True: Test용 / False: 실제 장비 적용)

        if USE_TEST_MODE:
            self.source_file = Path('composite.bmp')
        else:
            self.source_file = Path(r'Z:/composite.bmp')

        if INI_TEST_MODE:
            try:
                ini_base_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                ini_base_dir = os.getcwd()
            settings_path = os.path.join(ini_base_dir, "CompositeImageApp.ini")
        else:
            settings_path = r"D:\AI Vision\CompositeImageApp.ini"

        self.settings = QSettings(settings_path, QSettings.IniFormat)

        self.output_folder = Path('Images')
        self.output_folder.mkdir(exist_ok=True)

        self.status_label = QLabel('대기 중.')
        self.start_button = QPushButton('시작')
        self.stop_button = QPushButton('종료')
        self.stop_button.setEnabled(False)
        self.save_opt_button = QPushButton('설정 저장')

        # 윈도우 상부 텍스트 박스(장비/시작/종료/설정) 설정
        self.device_edit = QLineEdit()
        self.device_edit.setReadOnly(True)
        self.device_edit.setAlignment(Qt.AlignCenter)
        self.device_edit.setMinimumWidth(120)
        self.device_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.start_button.setMinimumWidth(45)
        self.start_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_button.setMinimumWidth(45)
        self.stop_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.save_opt_button.setMinimumWidth(75)
        self.save_opt_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # "장비(호스트)/Lot#" 텍스트 박스와 시작/종료/설정 버튼을 같은 행에 배치
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.device_edit)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.save_opt_button)

        # Spin 박스(저장 Interval) 생성 및 설정
        self.interval_spin = QSpinBox()
        self.interval_spin.setMinimum(1)
        self.interval_spin.setMaximum(60)
        self.interval_spin.setValue(4)
        self.interval_spin.setFixedWidth(85)
        self.interval_label = QLabel('저장(초):')

        # Spin 박스(삭제 Interval) 생성 및 설정
        self.delete_interval_spin = QSpinBox()
        self.delete_interval_spin.setMinimum(1)
        self.delete_interval_spin.setMaximum(1000)
        self.delete_interval_spin.setValue(1000)
        self.delete_interval_spin.setFixedWidth(85)
        self.delete_interval_label = QLabel('삭제(장):')

        # 그룹 박스(Interval Parameter) 생성 및 설정
        interval_group = QGroupBox('Interval Parameter')
        interval_layout = QGridLayout()
        interval_layout.addWidget(self.interval_label, 0, 0)
        interval_layout.addWidget(self.interval_spin, 0, 1)
        interval_layout.addWidget(self.delete_interval_label, 0, 2)
        interval_layout.addWidget(self.delete_interval_spin, 0, 3)
        interval_group.setLayout(interval_layout)

        # 체크 박스(BMP, JPEG) 생성
        self.bmp_check = QCheckBox('BMP(원본)')
        self.jpg_check = QCheckBox('JPEG')
        self.bmp_check.setChecked(False)
        self.jpg_check.setChecked(True)
        format_layout = QHBoxLayout()
        format_layout.addWidget(self.bmp_check)
        format_layout.addWidget(self.jpg_check)

        # Spin 박스(Crop 영역 지정) 생성 및 설정
        self.crop_x_label = QLabel("X:")
        self.crop_x = QSpinBox()
        self.crop_x.setMaximum(9999)
        self.crop_x.setFixedWidth(120)
        self.crop_y_label = QLabel("Y:")
        self.crop_y = QSpinBox()
        self.crop_y.setMaximum(9999)
        self.crop_y.setFixedWidth(120)
        self.crop_w_label = QLabel("W:")
        self.crop_w = QSpinBox()
        self.crop_w.setMaximum(9999)
        self.crop_w.setFixedWidth(120)
        self.crop_h_label = QLabel("H:")
        self.crop_h = QSpinBox()
        self.crop_h.setMaximum(9999)
        self.crop_h.setFixedWidth(120)

        # 그룹 박스(Crop Parameter) 생성 및 설정
        crop_group = QGroupBox('Crop Parameters')
        crop_layout = QGridLayout()
        crop_layout.addWidget(self.crop_x_label, 0, 0)
        crop_layout.addWidget(self.crop_x, 0, 1)
        crop_layout.addWidget(self.crop_y_label, 0, 2)
        crop_layout.addWidget(self.crop_y, 0, 3)
        crop_layout.addWidget(self.crop_w_label, 1, 0)
        crop_layout.addWidget(self.crop_w, 1, 1)
        crop_layout.addWidget(self.crop_h_label, 1, 2)
        crop_layout.addWidget(self.crop_h, 1, 3)
        crop_group.setLayout(crop_layout)

        # 슬라이더(JPEG 품질) 생성 및 설정
        qual_layout = QHBoxLayout()
        self.quality_label = QLabel('JPEG 품질: 90')
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(50, 100)
        self.quality_slider.setValue(90)
        qual_layout.addWidget(self.quality_label)
        qual_layout.addWidget(self.quality_slider)

        # 레이아웃(컨트롤 관련 위젯 수직 배치) 생성 및 설정
        control_layout = QVBoxLayout()
        control_layout.addWidget(self.status_label)
        control_layout.addLayout(button_layout)
        control_layout.addWidget(interval_group)
        control_layout.addWidget(crop_group)
        control_layout.addLayout(format_layout)
        control_layout.addLayout(qual_layout)
        control_layout.addStretch()

        # 레이아웃(이미지 미리보기) 생성 및 설정
        preview_panel = QVBoxLayout()
        self.preview_label = PreviewLabel()
        self.preview_label.setFixedSize(250, 250)
        preview_panel.addWidget(self.preview_label, alignment=Qt.AlignTop)
        preview_panel.addStretch()

        # 레이아웃(메인 윈도우) 생성 및 설정
        main_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addSpacing(10)
        main_layout.addLayout(preview_panel)
        self.setLayout(main_layout)

        # 위젯들(버튼 및 체크박스 등)의 시그널과 함수 연결
        self.image_thread = None
        self.start_button.clicked.connect(self.start_saving)
        self.stop_button.clicked.connect(self.stop_saving)
        self.save_opt_button.clicked.connect(self.save_options)
        self.bmp_check.stateChanged.connect(self.on_format_check)
        self.jpg_check.stateChanged.connect(self.on_format_check)

        # 크롭 영역 및 슬라이더 변경 시 동작 연결
        for spin in (self.crop_x, self.crop_y, self.crop_w, self.crop_h):
            spin.valueChanged.connect(self.update_preview)
        self.quality_slider.valueChanged.connect(self.set_quality)
        self.quality_slider.valueChanged.connect(self.update_preview)

        # 저장/삭제 Interval 변경 시 설정 저장
        self.interval_spin.valueChanged.connect(self.save_options)
        self.delete_interval_spin.valueChanged.connect(self.save_options)

        # 초기 설정 파일을 읽어 GUI에 반영
        self.load_options()
        self.set_quality(self.quality_slider.value())
        self.on_format_check()
        self.update_preview()

        # 실행파일 실행 시, "자동 시작 동작 수행" 여부 플래그 : 자동 시작(True) / 수동 시작(False)
        self.auto_start_enabled = True
        self.auto_start_triggered = False

        if self.auto_start_enabled:
            QTimer.singleShot(0, self.handle_auto_start)

        # STOP/DISABLE 파일 감시 (업데이트/정지/삭제-일시정지 시 GUI 즉시 종료)
        self._stop_timer = QTimer(self)
        self._stop_timer.setInterval(1000)  # 1초마다
        self._stop_timer.timeout.connect(self._check_stop_or_disable_and_quit)
        self._stop_timer.start()

    def start_saving(self):
        """이미지 저장 쓰레드 시작"""
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText('저장 시작...')
        self.image_thread = ImageSaverThread(
            self.source_file,
            self.output_folder,
            self.get_crop_rect,
            self.get_selected_formats,
            self.get_quality,
            self.get_interval,
            self.get_device_name,
            self.get_delete_interval_hours
        )
        self.image_thread.update_status.connect(self.status_label.setText)
        self.image_thread.update_preview.connect(self.update_preview)
        self.image_thread.update_device_name.connect(self.update_device_name_from_thread)
        self.image_thread.start()

    def handle_auto_start(self):
        """실행 파일 실행 시 자동으로 시작 버튼을 눌러주는 핸들러"""
        if self.auto_start_enabled and not self.auto_start_triggered:
            self.auto_start_triggered = True
            self.start_saving()

    def stop_saving(self):
        """이미지 저장 쓰레드 중지"""
        if self.image_thread:
            self.image_thread.stop()
            self.image_thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText('저장 중지.')

    def closeEvent(self, event):
        """윈도우 종료 시 쓰레드도 정리"""
        self.stop_saving()
        event.accept()

    def get_crop_rect(self):
        return (self.crop_x.value(), self.crop_y.value(),
                self.crop_w.value(), self.crop_h.value())

    def get_selected_formats(self):
        formats = []
        if self.bmp_check.isChecked():
            formats.append('BMP')
        if self.jpg_check.isChecked():
            formats.append('JPEG')
        return formats

    def get_quality(self):
        return self.quality_slider.value()

    def get_interval(self):
        return self.interval_spin.value()

    def get_delete_interval_hours(self):
        return self.delete_interval_spin.value()

    def get_device_name(self):
        return self.device_edit.text()

    def set_quality(self, val):
        self.quality_label.setText(f'JPEG 품질: {val}')

    def on_format_check(self):
        """포맷 체크 박스 상태에 따른 UI 동작 제어"""
        sender = self.sender()
        if not self.bmp_check.isChecked() and not self.jpg_check.isChecked():
            if sender == self.bmp_check:
                self.jpg_check.setChecked(True)
            else:
                self.bmp_check.setChecked(True)
        if self.bmp_check.isChecked() and self.jpg_check.isChecked():
            if sender == self.bmp_check:
                self.jpg_check.setChecked(False)
            else:
                self.bmp_check.setChecked(False)
        self.quality_slider.setEnabled(self.jpg_check.isChecked())
        self.quality_label.setEnabled(self.jpg_check.isChecked())

    def update_preview(self):
        """소스 이미지를 불러와서 프리뷰와 크롭 미리보기 업데이트"""
        if not self.source_file.exists():
            self.preview_label.clear()
            self.preview_label.setText('이미지 없음')
            return
        try:
            with Image.open(self.source_file) as img:
                img = img.convert('L')
                crop_rect = self.get_crop_rect()
                self.preview_label.set_image(img.copy(), crop_rect)
        except Exception as e:
            self.preview_label.clear()
            self.preview_label.setText(str(e))

    def save_options(self):
        """GUI 옵션을 설정 파일에 저장"""
        self.settings.setValue('crop_x', self.crop_x.value())
        self.settings.setValue('crop_y', self.crop_y.value())
        self.settings.setValue('crop_w', self.crop_w.value())
        self.settings.setValue('crop_h', self.crop_h.value())
        self.settings.setValue('interval', self.interval_spin.value())
        self.settings.setValue('delete_interval', self.delete_interval_spin.value())
        self.settings.setValue('bmp_checked', self.bmp_check.isChecked())
        self.settings.setValue('jpg_checked', self.jpg_check.isChecked())
        self.settings.setValue('jpeg_quality', self.quality_slider.value())
        self.settings.sync()
        self.status_label.setText('설정 저장됨.')

    def load_options(self):
        """설정 파일에서 옵션 읽어 GUI에 적용"""
        self.crop_x.setValue(int(self.settings.value('crop_x', 103)))
        self.crop_y.setValue(int(self.settings.value('crop_y', 0)))
        self.crop_w.setValue(int(self.settings.value('crop_w', 280)))
        self.crop_h.setValue(int(self.settings.value('crop_h', 600)))
        self.interval_spin.setValue(int(self.settings.value('interval', 4)))
        self.delete_interval_spin.setValue(int(self.settings.value('delete_interval', 1000)))
        self.bmp_check.setChecked(self.settings.value('bmp_checked', 'False') in ['true', 'True', True])
        self.jpg_check.setChecked(self.settings.value('jpg_checked', 'True') in ['true', 'True', True])
        self.device_edit.setText(str(self.settings.value('device_name', '')))
        qual = int(self.settings.value('jpeg_quality', 90))
        self.quality_slider.setValue(qual)

    def update_device_name_from_thread(self, full_device_name):
        """ImageSaverThread에서 보낸 최신 장비명으로 GUI 장비명 입력란 갱신"""
        self.device_edit.setText(full_device_name)

    def _check_stop_or_disable_and_quit(self):
        """STOP/DISABLE 감지 시 GUI 자발 종료 (업데이트 인수 & 삭제 후 멈춤 즉시 반영)"""
        if os.path.exists(STOP_FILE) or os.path.exists(DISABLE_FILE):
            QApplication.quit()


def run_gui():
    """QApplication 인스턴스의 중복 실행 문제 방지"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    else:
        try:
            app.quit()
        except Exception:
            pass
        app = QApplication(sys.argv)

    window = MainWindow()

    # GUI 시작 시 최신 CSV 파일명과 호스트명을 장비명 입력란에 반영
    latest_csv_fname = find_latest_csv_file()
    host_and_csvname = f"{host_name} / {latest_csv_fname.rsplit('.',1)[0] if latest_csv_fname else 'N/A'}"
    window.device_edit.setText(host_and_csvname)

    window.show()
    app.exec()


if __name__ == '__main__':
    # 즉시 워치독/자식 중지 (--stop 인자로 실행하면 STOP_FILE 생성 후 종료)
    if '--stop' in sys.argv:
        Path(STOP_FILE).touch()
        sys.exit(0)

    _maybe_stage_to_temp()

    # 워치독 조건 (설치 경로이거나 --staged 일 때만)
    if ('--no-watchdog' in sys.argv) or (not _should_enable_watchdog()):
        run_gui()
    else:
        if '--child' in sys.argv:
            run_gui()
        else:
            run_watchdog_loop()
