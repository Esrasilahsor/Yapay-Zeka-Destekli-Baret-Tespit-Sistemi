# -*- coding: utf-8 -*-
"""
İş Sağlığı ve Güvenliği (İSG) Yapay Zekâ Destekli Baret Tanıma Sistemi
YOLOv8 + PyQt5 + Arduino Uno R3 Endüstriyel Canlı İzleme ve Denetim Paneli
"""

import sys
import os
import time
from datetime import datetime
import threading

# Windows PyTorch DLL çakışmasını (WinError 1114) önlemek için torch ve ultralytics
# kütüphaneleri PyQt5'ten önce hafızaya yüklenmelidir.
try:
    import torch
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception:
    HAS_YOLO = False

import cv2
import numpy as np

# Windows hoparlör sesli ikaz modülü
try:
    import winsound
    HAS_WINSOUND = True
except Exception:
    HAS_WINSOUND = False

# Arduino seri iletişim modülü
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except Exception:
    HAS_SERIAL = False

from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QDateTime, QMutex
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame, QMessageBox, QSlider,
    QGroupBox, QStatusBar, QAbstractItemView, QSplitter
)


# ==============================================================================
# 1. ARDUINO UNO R3 SERİ HABERLEŞME YÖNETİCİSİ (HATA KORUMALI)
# ==============================================================================
class ArduinoManager:
    """
    Arduino Uno R3 ile donanımsal iletişim sağlayan yönetici sınıf.
    Bağlantı kopsa veya cihaz takılı olmasa bile sistemi asla çökertmez.
    """
    def __init__(self):
        self.serial_port = None
        self.is_connected = False
        self.current_port_name = ""
        self.last_sent_state = None  # b'0' veya b'1'
        self.last_sent_time = 0
        self.lock = threading.Lock()

    @staticmethod
    def get_available_ports():
        """Sistemdeki aktif seri portları (COM) listeler."""
        if not HAS_SERIAL:
            return []
        try:
            ports = serial.tools.list_ports.comports()
            return [port.device for port in ports]
        except Exception:
            return []

    def connect(self, port_name, baudrate=9600):
        """Belirtilen COM portu üzerinden Arduino'ya bağlanır."""
        if not HAS_SERIAL:
            return False, "pyserial kütüphanesi sistemde bulunamadı!"

        with self.lock:
            try:
                if self.serial_port and self.serial_port.is_open:
                    self.serial_port.close()

                self.serial_port = serial.Serial(
                    port=port_name,
                    baudrate=baudrate,
                    timeout=1,
                    write_timeout=1
                )
                time.sleep(1.5)  # Arduino bootloader yeniden başlatma beklemesi
                self.is_connected = True
                self.current_port_name = port_name
                self.last_sent_state = None
                return True, f"Bağlandı ({port_name} - {baudrate} baud)"
            except Exception as e:
                self.is_connected = False
                self.serial_port = None
                return False, f"Bağlantı Hatası: {str(e)}"

    def disconnect(self):
        """Seri port bağlantısını temiz şekilde sonlandırır."""
        with self.lock:
            if self.serial_port and self.serial_port.is_open:
                try:
                    self.serial_port.write(b'0')  # Alarmları kapat
                    time.sleep(0.05)
                    self.serial_port.close()
                except Exception:
                    pass
            self.is_connected = False
            self.serial_port = None
            self.last_sent_state = None

    def send_state(self, state='standby', is_hazard: bool = None):
        """
        Duruma göre Arduino'ya komut iletir:
        - 'hazard' / is_hazard=True  : Baretsiz İhlal -> '1' (Kırmızı LED + Kesikli Buzzer)
        - 'safe'                     : Baretli Güvenli -> '0' (Yeşil LED)
        - 'standby' / is_hazard=False: Çalışan Yok / Bekleme -> 'C' (Tüm İkazlar Kapalı)
        """
        if not self.is_connected or not self.serial_port:
            return False

        if is_hazard is not None:
            command_byte = b'1' if is_hazard else b'C'
        elif state == 'hazard' or state is True or state == '1':
            command_byte = b'1'
        elif state == 'safe' or state == '0':
            command_byte = b'0'
        else:
            command_byte = b'C'

        current_time = time.time()

        # Buffer taşmasını önlemek için durum değiştiğinde veya 1 sn sonra gönder
        if command_byte == self.last_sent_state and (current_time - self.last_sent_time) < 1.0:
            return True

        with self.lock:
            try:
                self.serial_port.write(command_byte)
                self.serial_port.flush()
                self.last_sent_state = command_byte
                self.last_sent_time = current_time
                return True
            except Exception:
                # İletişim koptuysa yazılımsal moda güvenli geçiş yap
                self.is_connected = False
                try:
                    self.serial_port.close()
                except Exception:
                    pass
                self.serial_port = None
                return False


# ==============================================================================
# 2. VİDEO VE YOLOV8 ÇIKARIM İŞ PARÇACIĞI (QTHREAD)
# ==============================================================================
class VideoDetectionThread(QThread):
    """
    Video yakalama ve YOLOv8 çıkarımını ana GUI'yi dondurmadan
    30 FPS akıcılıkta yürüten arka plan iş parçacığı.
    """
    frame_processed = pyqtSignal(QImage, int, int, int, list, np.ndarray)
    error_occurred = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    fps_measured = pyqtSignal(float)

    def __init__(self, camera_index=0, model_path="best.pt"):
        super().__init__()
        self.camera_index = camera_index
        self.model_path = model_path
        self.conf_threshold = 0.50
        self.iou_threshold = 0.45
        self.running = False
        self.model = None
        self.device = "cpu"
        self.mutex = QMutex()
        self.tracked_targets = []  # Zamansal kararlılık ve hareket yumuşatma takibi (Debounce Tracker)

    def set_confidence(self, conf: float):
        """Güven eşiğini dinamik olarak günceller."""
        self.mutex.lock()
        self.conf_threshold = conf
        self.mutex.unlock()

    def set_camera_index(self, index: int):
        """Kamera aygıt indeksini günceller."""
        self.camera_index = index

    def load_model(self):
        """Eğitilmiş best.pt YOLO modelini hafızaya yükler."""
        if not HAS_YOLO:
            self.status_updated.emit("⚠️ Ultralytics / PyTorch kütüphaneleri eksik!")
            return False

        # Öncelikli olarak best.pt, yoksa models/best.pt yoluna bak
        target_path = self.model_path
        if not os.path.exists(target_path):
            alt_path = os.path.join("models", "best.pt")
            if os.path.exists(alt_path):
                target_path = alt_path

        if not os.path.exists(target_path):
            self.status_updated.emit(f"⚠️ Model dosyası '{self.model_path}' bulunamadı. Lütfen best.pt dosyasını ekleyin.")
            return False

        try:
            self.device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
            self.model = YOLO(target_path)
            self.status_updated.emit(f"Model Yüklendi: {target_path} (Cihaz: {self.device.upper()})")
            return True
        except Exception as e:
            self.model = None
            self.status_updated.emit(f"Model Yükleme Hatası: {str(e)}")
            return False

    def run(self):
        """30 FPS hedefiyle kamera okuma ve YOLOv8 tespit döngüsü."""
        self.running = True

        # Model henüz yüklenmediyse yüklemeyi dene
        if self.model is None:
            self.load_model()

        # Kamera yakalama nesnesini başlat (Windows için DirectShow öncelikli)
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            self.error_occurred.emit("Kamera Bağlantı Hatası: Aygıt Bulunamadı!")
            self.running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # 30 FPS zamanlama değişkenleri
        target_frame_time = 1.0 / 30.0
        fps_timer = time.time()
        fps_counter = 0

        while self.running:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Model çıkarımı ve görsel kutulama
            annotated_frame = frame.copy()
            helmet_count = 0
            head_count = 0
            detections = []

            self.mutex.lock()
            active_conf = self.conf_threshold
            self.mutex.unlock()

            if self.model is not None:
                try:
                    # Akıllı ve İSG Öncelikli YOLOv8 Çıkarımı
                    # agnostic_nms=True ile aynı bölgedeki farklı sınıf kutuları teke indirgenir
                    results = self.model(
                        annotated_frame,
                        conf=min(active_conf, 0.35),
                        iou=self.iou_threshold,
                        agnostic_nms=True,
                        verbose=False,
                        device=self.device
                    )

                    if results and len(results) > 0:
                        raw_boxes = results[0].boxes
                        frame_h, frame_w = annotated_frame.shape[:2]

                        candidates = []
                        for box in raw_boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            class_name = self.model.names.get(cls_id, str(cls_id)).lower()

                            bw = x2 - x1
                            bh = y2 - y1

                            # Kenar ve minik arka plan nesne filtresi (Odadaki lamba, köşe askılığı vs.)
                            if (x1 <= 5 or x2 >= frame_w - 5) and (bw < 50 or bh < 50):
                                continue

                            # İSG Eşik Mantığı:
                            # 1. Baretsiz Baş (head, class 0): Daha hassas yakalanmalı (Güvenlik Önceliklidir)
                            is_head_cand = (cls_id == 0 or 'head' in class_name) and (conf >= max(0.35, active_conf * 0.80))

                            # 2. Baret (helmet / hi-viz helmet, class 1, 2):
                            # Saç parlaması ve yuvarlak oda nesnelerinden kaynaklanan sahte baretleri engellemek için
                            # en az 0.68 veya active_conf gereklidir
                            is_helmet_cand = (cls_id in (1, 2) or 'helmet' in class_name) and (conf >= max(active_conf, 0.68))

                            if is_head_cand:
                                candidates.append({
                                    'type': 'head',
                                    'conf': conf,
                                    'box': (x1, y1, x2, y2),
                                    'cls_id': 0
                                })
                            elif is_helmet_cand:
                                candidates.append({
                                    'type': 'helmet',
                                    'conf': conf,
                                    'box': (x1, y1, x2, y2),
                                    'cls_id': 1
                                })

                        # Çakışma ve İSG Hakem Mantığı (Spatial Arbitration)
                        # Aynı bölgede hem kafa hem baret adayı varsa, baret skoru kafadan ezici üstün değilse İHLAL sayılır
                        final_boxes = []
                        for i, c1 in enumerate(candidates):
                            suppressed = False
                            for j, c2 in enumerate(candidates):
                                if i == j:
                                    continue
                                b1 = c1['box']
                                b2 = c2['box']
                                ix1 = max(b1[0], b2[0])
                                iy1 = max(b1[1], b2[1])
                                ix2 = min(b1[2], b2[2])
                                iy2 = min(b1[3], b2[3])
                                inter_w = max(0, ix2 - ix1)
                                inter_h = max(0, iy2 - iy1)
                                inter_area = inter_w * inter_h
                                if inter_area > 0:
                                    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                                    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                                    overlap = inter_area / float(min(area1, area2))
                                    if overlap > 0.40:
                                        # İki aday aynı nesneye ait:
                                        if c1['type'] == 'helmet' and c2['type'] == 'head':
                                            # Baret skoru kafadan en az 0.20 yüksek değilse bareti bastır, kafayı (ihlal) seç!
                                            if c1['conf'] < c2['conf'] + 0.20:
                                                suppressed = True
                                                break
                                        elif c1['type'] == 'head' and c2['type'] == 'helmet':
                                            if c2['conf'] >= c1['conf'] + 0.20:
                                                suppressed = True
                                                break
                            if not suppressed:
                                final_boxes.append(c1)

                        # 3. ZAMANSAL KARARLILIK VE HAREKET TAKİPÇİSİ (TEMPORAL DEBOUNCE TRACKER)
                        # Ani kafa hareketleri ve hareket bulanıklığında (motion blur) titremeyi önler
                        matched_indices = set()
                        updated_targets = []

                        for item in final_boxes:
                            ix1, iy1, ix2, iy2 = item['box']
                            best_iou = 0.0
                            best_match = None
                            best_idx = -1

                            for idx, target in enumerate(self.tracked_targets):
                                if idx in matched_indices:
                                    continue
                                tx1, ty1, tx2, ty2 = target['box']
                                xx1 = max(ix1, tx1)
                                yy1 = max(iy1, ty1)
                                xx2 = min(ix2, tx2)
                                yy2 = min(iy2, ty2)
                                iw = max(0, xx2 - xx1)
                                ih = max(0, yy2 - yy1)
                                inter = iw * ih
                                if inter > 0:
                                    union = (ix2 - ix1) * (iy2 - iy1) + (tx2 - tx1) * (ty2 - ty1) - inter
                                    iou = inter / float(union) if union > 0 else 0
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_match = target
                                        best_idx = idx

                            if best_match is not None and best_iou > 0.20:
                                matched_indices.add(best_idx)
                                best_match['missed'] = 0
                                best_match['history'].append(item['type'])
                                if len(best_match['history']) > 8:
                                    best_match['history'].pop(0)

                                # Kutu yumuşatma (Jitter / Titreme önleyici Exponential Smoothing)
                                smooth_box = (
                                    int(0.65 * ix1 + 0.35 * best_match['box'][0]),
                                    int(0.65 * iy1 + 0.35 * best_match['box'][1]),
                                    int(0.65 * ix2 + 0.35 * best_match['box'][2]),
                                    int(0.65 * iy2 + 0.35 * best_match['box'][3]),
                                )
                                best_match['box'] = smooth_box
                                best_match['conf'] = round(0.70 * item['conf'] + 0.30 * best_match['conf'], 2)

                                # Zamansal Kararlılık Kararı (Histerezis):
                                # İSG Kuralı: Son 8 karede en az 2 kare kafa görüldüyse -> BARETSİZ
                                head_votes = best_match['history'].count('head')
                                helmet_votes = best_match['history'].count('helmet')

                                if head_votes >= 2:
                                    best_match['type'] = 'head'
                                elif helmet_votes >= 5:
                                    best_match['type'] = 'helmet'
                                # Aksi halde mevcut durumunu korur (ani sıçrama engellenir)

                                updated_targets.append(best_match)
                            else:
                                # Yeni tespit edilen hedef
                                new_target = {
                                    'box': (ix1, iy1, ix2, iy2),
                                    'type': item['type'],
                                    'conf': item['conf'],
                                    'history': [item['type']],
                                    'missed': 0
                                }
                                updated_targets.append(new_target)

                        # Geçici olarak kaçırılan hedeflere 3 kare tolerans tanı (kırpışmayı önler)
                        for idx, target in enumerate(self.tracked_targets):
                            if idx not in matched_indices:
                                target['missed'] += 1
                                if target['missed'] <= 3:
                                    updated_targets.append(target)

                        self.tracked_targets = updated_targets

                        # 4. Kararlı Hedefleri Çizme ve Sayma
                        for target in self.tracked_targets:
                            x1, y1, x2, y2 = target['box']
                            conf = target['conf']
                            t_type = target['type']

                            if t_type == 'head':
                                head_count += 1
                                box_color = (68, 68, 239)  # Kırmızı (#EF4444)
                                box_thickness = 3
                                label_text = f"Baretsiz! - %{int(conf * 100)}"
                                detections.append({
                                    'class_id': 0,
                                    'label': 'Baretsiz',
                                    'conf': conf,
                                    'box': (x1, y1, x2, y2)
                                })
                            else:
                                helmet_count += 1
                                box_color = (94, 197, 34)  # Yeşil (#22C55E)
                                box_thickness = 2
                                label_text = f"Baretli - %{int(conf * 100)}"
                                detections.append({
                                    'class_id': 1,
                                    'label': 'Baretli',
                                    'conf': conf,
                                    'box': (x1, y1, x2, y2)
                                })

                            # Bounding Box Çizimi
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, box_thickness)

                            # Etiket Arka Plan Rozeti
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.55
                            font_thickness = 2
                            (txt_w, txt_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                            
                            badge_y1 = max(0, y1 - txt_h - 10)
                            badge_y2 = y1
                            cv2.rectangle(
                                annotated_frame,
                                (x1, badge_y1),
                                (x1 + txt_w + 10, badge_y2),
                                box_color,
                                -1
                            )
                            cv2.putText(
                                annotated_frame,
                                label_text,
                                (x1 + 5, y1 - 4),
                                font,
                                font_scale,
                                (255, 255, 255),
                                font_thickness,
                                cv2.LINE_AA
                            )
                    else:
                        # Eğer bu karede hiçbir nesne tespit edilemediyse eski hedefleri yavaşça düşür
                        for target in self.tracked_targets:
                            target['missed'] += 1
                        self.tracked_targets = [t for t in self.tracked_targets if t['missed'] <= 2]

                except Exception as ex:
                    # Model çıkarım hatasını yut ve kamerayı devam ettir
                    pass

            total_count = helmet_count + head_count

            # BGR karesini PyQt QImage nesnesine dönüştürme
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

            # Ana arayüze sinyali ilet
            self.frame_processed.emit(
                qt_image,
                helmet_count,
                head_count,
                total_count,
                detections,
                annotated_frame
            )

            # FPS Ölçümü
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                calc_fps = fps_counter / (time.time() - fps_timer)
                self.fps_measured.emit(calc_fps)
                fps_counter = 0
                fps_timer = time.time()

            # 30 FPS Akıcılık ve CPU Rahatlatma Zamanlaması
            elapsed = time.time() - loop_start
            sleep_time = target_frame_time - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)
            else:
                time.sleep(0.001)

        cap.release()

    def stop(self):
        """Kamera iş parçacığını güvenli durdurur."""
        self.running = False
        self.wait(2000)


# ==============================================================================
# 3. ANA ARAYÜZ PENCERESİ (MODERN ENDÜSTRİYEL KONTROL KONSOLU)
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🛡️ İSG Baret Tanıma ve Güvenlik Denetim Sistemi | Kontrol Paneli")
        self.resize(1280, 820)
        self.setMinimumSize(1024, 700)

        # Temel Dizinler
        self.violation_dir = "ihlal_kayitlari"
        os.makedirs(self.violation_dir, exist_ok=True)
        self.csv_log_path = os.path.join(self.violation_dir, "ihlal_kayitlari.csv")
        self._ensure_csv_header()

        # İhlal ve İkaz Yönetimi
        self.last_violation_time = 0
        self.violation_cooldown = 3.0  # 3 saniye soğuma süresi (disk ve ses spamini önler)
        self.total_violations_session = 0
        self.is_flashing = False
        self.flash_state = False

        # Donanım ve İş Parçacıkları
        self.arduino = ArduinoManager()
        self.video_thread = None
        self.latest_frame = None

        # Görsel Yanıp Sönme Zamanlayıcısı (Alarm Banner için)
        self.blink_timer = QTimer(self)
        self.blink_timer.setInterval(500)
        self.blink_timer.timeout.connect(self._toggle_blink)

        # Canlı Saat Zamanlayıcısı
        self.clock_timer = QTimer(self)
        self.clock_timer.setInterval(1000)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start()

        # Arayüz Bileşenlerini Kur ve QSS Uygula
        self._setup_ui()
        self._apply_qss()

        # Başlangıçta Seri Portları Tara
        self._refresh_serial_ports()

    def _ensure_csv_header(self):
        """Kayıt kütüğü CSV dosyasının başlık satırını garanti eder."""
        if not os.path.exists(self.csv_log_path):
            try:
                with open(self.csv_log_path, "w", encoding="utf-8") as f:
                    f.write("Tarih,Saat,BaretliSayisi,BaretsizSayisi,EnYuksekGuven,FotografYolu\n")
            except Exception:
                pass

    # --------------------------------------------------------------------------
    # ARAYÜZ YERLEŞİMİ (UI SETUP)
    # --------------------------------------------------------------------------
    def _setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(12)

        # 1. ÜST BAŞLIK ÇUBUĞU (HEADER)
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(16, 10, 16, 10)

        title_box = QVBoxLayout()
        main_title = QLabel("🛡️ İSG BARET VE GÜVENLİK DENETİM MERKEZİ")
        main_title.setObjectName("mainTitle")
        sub_title = QLabel("YOLOv8 Yapay Zekâ Destekli Gerçek Zamanlı Saha Güvenlik Kontrol Konsolu")
        sub_title.setObjectName("subTitle")
        title_box.addWidget(main_title)
        title_box.addWidget(sub_title)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        # Canlı Saat ve FPS Göstergeleri
        stats_header_layout = QHBoxLayout()
        self.fps_badge = QLabel("FPS: --")
        self.fps_badge.setObjectName("badgeFps")
        self.clock_badge = QLabel(QDateTime.currentDateTime().toString("dd.MM.yyyy HH:mm:ss"))
        self.clock_badge.setObjectName("badgeClock")
        stats_header_layout.addWidget(self.fps_badge)
        stats_header_layout.addWidget(self.clock_badge)
        header_layout.addLayout(stats_header_layout)
        main_layout.addWidget(header_frame)

        # 2. ORTA BÖLÜM: KAMERA GÖRÜNTÜSÜ VE SAĞ KONTROL PANELİ
        middle_splitter = QSplitter(Qt.Horizontal)

        # SOL: Video Akışı Konteynırı
        video_container = QFrame()
        video_container.setObjectName("videoContainer")
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(12, 12, 12, 12)
        video_layout.setSpacing(8)

        # Video Üst Rozeti
        video_header = QHBoxLayout()
        self.cam_status_lbl = QLabel("● KAMERA DURDURULDU")
        self.cam_status_lbl.setObjectName("camStatusStopped")
        video_header.addWidget(self.cam_status_lbl)
        video_header.addStretch()
        self.res_lbl = QLabel("Çözünürlük: 640x480 @ 30 FPS")
        self.res_lbl.setStyleSheet("color: #94A3B8; font-size: 11px;")
        video_header.addWidget(self.res_lbl)
        video_layout.addLayout(video_header)

        # Video Görüntüleme Alanı
        self.video_label = QLabel()
        self.video_label.setObjectName("videoScreen")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setText("Kamera Akışını Başlatmak İçin 'Kamerayı Başlat' Butonuna Tıklayın")
        video_layout.addWidget(self.video_label, 1)

        middle_splitter.addWidget(video_container)

        # SAĞ: Kontrol ve Gösterge Paneli
        right_panel = QFrame()
        right_panel.setObjectName("rightPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(14, 12, 14, 12)
        right_layout.setSpacing(10)

        # GÜVENLİK DURUM BARI (EN ÜSTTE DİKKAT ÇEKİCİ - 3 DURUMLU)
        self.status_bar_banner = QLabel("👁 SAHA İZLENİYOR (BEKLEMEDE)")
        self.status_bar_banner.setObjectName("bannerStandby")
        self.status_bar_banner.setAlignment(Qt.AlignCenter)
        self.status_bar_banner.setMinimumHeight(44)
        right_layout.addWidget(self.status_bar_banner)

        # SAYAÇ KARTLARI (3 KART: YEŞİL, KIRMIZI, MAVİ)
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(8)

        # Yeşil Kart: Baretli Çalışan
        self.card_helmet = QFrame()
        self.card_helmet.setObjectName("cardHelmet")
        c1_layout = QVBoxLayout(self.card_helmet)
        c1_layout.setContentsMargins(8, 8, 8, 8)
        lbl_c1_title = QLabel("BARETLİ")
        lbl_c1_title.setObjectName("cardTitle")
        self.val_helmet = QLabel("0")
        self.val_helmet.setObjectName("cardValueGreen")
        self.val_helmet.setAlignment(Qt.AlignCenter)
        lbl_c1_sub = QLabel("Güvenli Çalışan")
        lbl_c1_sub.setObjectName("cardSub")
        lbl_c1_sub.setAlignment(Qt.AlignCenter)
        c1_layout.addWidget(lbl_c1_title)
        c1_layout.addWidget(self.val_helmet)
        c1_layout.addWidget(lbl_c1_sub)
        cards_layout.addWidget(self.card_helmet)

        # Kırmızı Kart: Baretsiz Çalışan (KRİTİK İHLAL)
        self.card_violation = QFrame()
        self.card_violation.setObjectName("cardViolation")
        c2_layout = QVBoxLayout(self.card_violation)
        c2_layout.setContentsMargins(8, 8, 8, 8)
        lbl_c2_title = QLabel("BARETSİZ")
        lbl_c2_title.setObjectName("cardTitle")
        self.val_head = QLabel("0")
        self.val_head.setObjectName("cardValueRed")
        self.val_head.setAlignment(Qt.AlignCenter)
        lbl_c2_sub = QLabel("Kritik İhlal!")
        lbl_c2_sub.setObjectName("cardSub")
        lbl_c2_sub.setAlignment(Qt.AlignCenter)
        c2_layout.addWidget(lbl_c2_title)
        c2_layout.addWidget(self.val_head)
        c2_layout.addWidget(lbl_c2_sub)
        cards_layout.addWidget(self.card_violation)

        # Mavi Kart: Toplam Kişi
        self.card_total = QFrame()
        self.card_total.setObjectName("cardTotal")
        c3_layout = QVBoxLayout(self.card_total)
        c3_layout.setContentsMargins(8, 8, 8, 8)
        lbl_c3_title = QLabel("TOPLAM")
        lbl_c3_title.setObjectName("cardTitle")
        self.val_total = QLabel("0")
        self.val_total.setObjectName("cardValueBlue")
        self.val_total.setAlignment(Qt.AlignCenter)
        lbl_c3_sub = QLabel("Tespit Edilen")
        lbl_c3_sub.setObjectName("cardSub")
        lbl_c3_sub.setAlignment(Qt.AlignCenter)
        c3_layout.addWidget(lbl_c3_title)
        c3_layout.addWidget(self.val_total)
        c3_layout.addWidget(lbl_c3_sub)
        cards_layout.addWidget(self.card_total)

        right_layout.addLayout(cards_layout)

        # KAMERA KONTROL GRUBU
        cam_group = QGroupBox("Kamera Kontrolleri")
        cam_group_layout = QVBoxLayout(cam_group)
        cam_btns = QHBoxLayout()
        self.btn_start_cam = QPushButton("▶ Kamerayı Başlat")
        self.btn_start_cam.setObjectName("btnStart")
        self.btn_start_cam.clicked.connect(self.start_camera)

        self.btn_stop_cam = QPushButton("⏹ Kamerayı Durdur")
        self.btn_stop_cam.setObjectName("btnStop")
        self.btn_stop_cam.setEnabled(False)
        self.btn_stop_cam.clicked.connect(self.stop_camera)

        cam_btns.addWidget(self.btn_start_cam)
        cam_btns.addWidget(self.btn_stop_cam)
        cam_group_layout.addLayout(cam_btns)

        cam_select_layout = QHBoxLayout()
        cam_select_layout.addWidget(QLabel("Kamera Seçimi:"))
        self.combo_cam = QComboBox()
        self.combo_cam.addItems(["Dahili Web Kamerası (0)", "Harici USB Kamera (1)", "Harici USB Kamera (2)"])
        cam_select_layout.addWidget(self.combo_cam)
        cam_group_layout.addLayout(cam_select_layout)
        right_layout.addWidget(cam_group)

        # ARDUINO UNO R3 ENTEGRASYON GRUBU
        ard_group = QGroupBox("Arduino Uno R3 Entegrasyonu")
        ard_group_layout = QVBoxLayout(ard_group)

        port_select_layout = QHBoxLayout()
        port_select_layout.addWidget(QLabel("COM Port:"))
        self.combo_ports = QComboBox()
        self.combo_ports.setMinimumWidth(120)
        self.btn_refresh_ports = QPushButton("🔄")
        self.btn_refresh_ports.setToolTip("Seri Portları Yenile")
        self.btn_refresh_ports.setMaximumWidth(36)
        self.btn_refresh_ports.clicked.connect(self._refresh_serial_ports)

        port_select_layout.addWidget(self.combo_ports, 1)
        port_select_layout.addWidget(self.btn_refresh_ports)
        ard_group_layout.addLayout(port_select_layout)

        ard_btn_layout = QHBoxLayout()
        self.btn_arduino_connect = QPushButton("🔌 Arduino'ya Bağlan")
        self.btn_arduino_connect.setObjectName("btnArduino")
        self.btn_arduino_connect.clicked.connect(self._toggle_arduino_connection)
        ard_btn_layout.addWidget(self.btn_arduino_connect)
        ard_group_layout.addLayout(ard_btn_layout)

        self.arduino_status_badge = QLabel("Arduino Bağlı Değil (Yazılımsal Mod)")
        self.arduino_status_badge.setObjectName("badgeArduinoOff")
        self.arduino_status_badge.setAlignment(Qt.AlignCenter)
        ard_group_layout.addWidget(self.arduino_status_badge)
        right_layout.addWidget(ard_group)

        # YAPAY ZEKA MODEL VE AYARLAR GRUBU
        ai_group = QGroupBox("Algılama ve Eşik Ayarları")
        ai_layout = QVBoxLayout(ai_group)

        slider_layout = QHBoxLayout()
        self.lbl_conf = QLabel("Güven Eşiği (Conf): %55")
        slider_layout.addWidget(self.lbl_conf)
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 95)
        self.slider_conf.setValue(55)
        self.slider_conf.valueChanged.connect(self._on_conf_changed)
        slider_layout.addWidget(self.slider_conf)
        ai_layout.addLayout(slider_layout)

        # Manuel Anlık Fotoğraf Çek Butonu
        self.btn_snapshot = QPushButton("📸 Anlık Görüntü Al (Snapshot)")
        self.btn_snapshot.setObjectName("btnSnapshot")
        self.btn_snapshot.clicked.connect(self._take_manual_snapshot)
        ai_layout.addWidget(self.btn_snapshot)

        right_layout.addWidget(ai_group)
        right_layout.addStretch()

        middle_splitter.addWidget(right_panel)
        middle_splitter.setStretchFactor(0, 7)
        middle_splitter.setStretchFactor(1, 3)
        main_layout.addWidget(middle_splitter, 1)

        # 3. ALT BÖLÜM: OLAY GÜNLÜĞÜ (LOG TABLOSU)
        log_frame = QFrame()
        log_frame.setObjectName("logFrame")
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(12, 8, 12, 8)
        log_layout.setSpacing(6)

        log_header = QHBoxLayout()
        lbl_log_title = QLabel("📋 Olay Günlüğü (Son İhlal Kayıtları)")
        lbl_log_title.setObjectName("logHeaderTitle")
        self.lbl_total_violations = QLabel("Oturum Toplam İhlali: 0")
        self.lbl_total_violations.setStyleSheet("color: #EF4444; font-weight: bold;")
        log_header.addWidget(lbl_log_title)
        log_header.addStretch()
        log_header.addWidget(self.lbl_total_violations)
        log_layout.addLayout(log_header)

        # İhlal Tablosu
        self.log_table = QTableWidget(0, 5)
        self.log_table.setHorizontalHeaderLabels([
            "Zaman Damgası", "Olay Türü", "Baretli Sayısı", "Baretsiz Sayısı", "Kanıt Fotoğrafı"
        ])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.log_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.log_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.log_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.log_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.log_table.setMaximumHeight(150)
        log_layout.addWidget(self.log_table)

        main_layout.addWidget(log_frame)

        # 4. DURUM ÇUBUĞU (STATUS BAR)
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.status_model_label = QLabel("Model: 'best.pt' aranıyor...")
        self.status_model_label.setStyleSheet("color: #94A3B8; padding-left: 8px;")
        self.statusBar.addWidget(self.status_model_label, 1)

    # --------------------------------------------------------------------------
    # QSS STİL TEMASI (MODERN ENDÜSTRİYEL KOYU TEMA)
    # --------------------------------------------------------------------------
    def _apply_qss(self):
        dark_qss = """
        QMainWindow {
            background-color: #0B1120;
        }
        QWidget {
            color: #E2E8F0;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 13px;
        }
        QFrame#headerFrame {
            background-color: #1E293B;
            border-radius: 8px;
            border: 1px solid #334155;
        }
        QLabel#mainTitle {
            font-size: 18px;
            font-weight: 700;
            color: #38BDF8;
            letter-spacing: 0.5px;
        }
        QLabel#subTitle {
            font-size: 12px;
            color: #94A3B8;
        }
        QLabel#badgeFps {
            background-color: #0F172A;
            border: 1px solid #38BDF8;
            color: #38BDF8;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: bold;
        }
        QLabel#badgeClock {
            background-color: #0F172A;
            border: 1px solid #475569;
            color: #CBD5E1;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
        }
        QFrame#videoContainer {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 10px;
        }
        QLabel#camStatusStopped {
            color: #EF4444;
            font-weight: bold;
            font-size: 12px;
        }
        QLabel#camStatusActive {
            color: #22C55E;
            font-weight: bold;
            font-size: 12px;
        }
        QLabel#videoScreen {
            background-color: #020617;
            border: 2px dashed #334155;
            border-radius: 8px;
            color: #64748B;
            font-size: 14px;
        }
        QFrame#rightPanel {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 10px;
        }
        /* Güvenlik Durum Barı */
        QLabel#bannerStandby {
            background-color: #1E293B;
            border: 2px solid #38BDF8;
            color: #E2E8F0;
            font-size: 15px;
            font-weight: bold;
            border-radius: 8px;
            letter-spacing: 0.5px;
        }
        QLabel#bannerSafe {
            background-color: #14532D;
            border: 2px solid #22C55E;
            color: #DCFCE7;
            font-size: 15px;
            font-weight: bold;
            border-radius: 8px;
            letter-spacing: 0.5px;
        }
        QLabel#bannerViolation1 {
            background-color: #EF4444;
            border: 2px solid #FCA5A5;
            color: #FFFFFF;
            font-size: 15px;
            font-weight: bold;
            border-radius: 8px;
            letter-spacing: 0.5px;
        }
        QLabel#bannerViolation2 {
            background-color: #7F1D1D;
            border: 2px solid #EF4444;
            color: #FEE2E2;
            font-size: 15px;
            font-weight: bold;
            border-radius: 8px;
            letter-spacing: 0.5px;
        }
        /* Sayaç Kartları */
        QFrame#cardHelmet {
            background-color: #064E3B;
            border: 1px solid #22C55E;
            border-radius: 8px;
        }
        QFrame#cardViolation {
            background-color: #7F1D1D;
            border: 1px solid #EF4444;
            border-radius: 8px;
        }
        QFrame#cardTotal {
            background-color: #1E3A8A;
            border: 1px solid #3B82F6;
            border-radius: 8px;
        }
        QLabel#cardTitle {
            font-size: 11px;
            font-weight: 700;
            color: #E2E8F0;
            letter-spacing: 0.5px;
        }
        QLabel#cardValueGreen {
            font-size: 28px;
            font-weight: 900;
            color: #4ADE80;
        }
        QLabel#cardValueRed {
            font-size: 28px;
            font-weight: 900;
            color: #F87171;
        }
        QLabel#cardValueBlue {
            font-size: 28px;
            font-weight: 900;
            color: #60A5FA;
        }
        QLabel#cardSub {
            font-size: 10px;
            color: #CBD5E1;
        }
        /* Gruplar */
        QGroupBox {
            border: 1px solid #334155;
            border-radius: 8px;
            margin-top: 14px;
            font-weight: bold;
            color: #94A3B8;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
        /* Butonlar */
        QPushButton {
            background-color: #334155;
            border: 1px solid #475569;
            color: #F8FAFC;
            border-radius: 6px;
            padding: 8px 14px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #475569;
            border-color: #64748B;
        }
        QPushButton#btnStart {
            background-color: #059669;
            border: 1px solid #10B981;
        }
        QPushButton#btnStart:hover {
            background-color: #10B981;
        }
        QPushButton#btnStop {
            background-color: #DC2626;
            border: 1px solid #EF4444;
        }
        QPushButton#btnStop:hover {
            background-color: #EF4444;
        }
        QPushButton#btnArduino {
            background-color: #2563EB;
            border: 1px solid #3B82F6;
        }
        QPushButton#btnArduino:hover {
            background-color: #3B82F6;
        }
        QPushButton#btnSnapshot {
            background-color: #0284C7;
            border: 1px solid #38BDF8;
        }
        QPushButton#btnSnapshot:hover {
            background-color: #38BDF8;
        }
        QLabel#badgeArduinoOff {
            background-color: #334155;
            border: 1px solid #475569;
            color: #FBBF24;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 11px;
        }
        QLabel#badgeArduinoOn {
            background-color: #064E3B;
            border: 1px solid #22C55E;
            color: #4ADE80;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 11px;
        }
        QComboBox {
            background-color: #0F172A;
            border: 1px solid #475569;
            border-radius: 6px;
            padding: 6px 10px;
            color: #F8FAFC;
        }
        QComboBox::drop-down {
            border: none;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #334155;
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background: #38BDF8;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #F8FAFC;
            border: 1px solid #38BDF8;
            width: 14px;
            margin-top: -4px;
            margin-bottom: -4px;
            border-radius: 7px;
        }
        /* Log Tablosu */
        QFrame#logFrame {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 8px;
        }
        QLabel#logHeaderTitle {
            font-weight: 700;
            color: #38BDF8;
            font-size: 13px;
        }
        QTableWidget {
            background-color: #0F172A;
            border: 1px solid #334155;
            border-radius: 6px;
            gridline-color: #1E293B;
            color: #E2E8F0;
        }
        QTableWidget::item {
            padding: 4px;
            border-bottom: 1px solid #1E293B;
        }
        QHeaderView::section {
            background-color: #1E293B;
            color: #94A3B8;
            padding: 5px;
            font-weight: 600;
            border: none;
            border-bottom: 1px solid #334155;
        }
        """
        self.setStyleSheet(dark_qss)

    # --------------------------------------------------------------------------
    # KAMERA VE İŞ PARÇACIĞI KONTROLÜ
    # --------------------------------------------------------------------------
    def start_camera(self):
        """Kamerayı ve YOLO iş parçacığını başlatır."""
        cam_idx = self.combo_cam.currentIndex()

        self.video_thread = VideoDetectionThread(camera_index=cam_idx, model_path="best.pt")
        self.video_thread.set_confidence(self.slider_conf.value() / 100.0)
        self.video_thread.frame_processed.connect(self.on_frame_received)
        self.video_thread.error_occurred.connect(self.on_camera_error)
        self.video_thread.status_updated.connect(self.on_status_updated)
        self.video_thread.fps_measured.connect(self.on_fps_measured)

        self.video_thread.start()

        self.btn_start_cam.setEnabled(False)
        self.btn_stop_cam.setEnabled(True)
        self.combo_cam.setEnabled(False)
        self.cam_status_lbl.setText("● CANLI AKIŞ DEVAM EDİYOR (30 FPS)")
        self.cam_status_lbl.setObjectName("camStatusActive")
        self.cam_status_lbl.setStyle(self.cam_status_lbl.style())

    def stop_camera(self):
        """Kamerayı ve arka plan iş parçacığını durdurur."""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        self.btn_start_cam.setEnabled(True)
        self.btn_stop_cam.setEnabled(False)
        self.combo_cam.setEnabled(True)
        self.cam_status_lbl.setText("● KAMERA DURDURULDU")
        self.cam_status_lbl.setObjectName("camStatusStopped")
        self.cam_status_lbl.setStyle(self.cam_status_lbl.style())
        self.fps_badge.setText("FPS: --")
        self.video_label.setText("Kamera Akışı Durduruldu")

        # Güvenlik durumunu bekleme moduna çek
        self._set_standby_ui()
        self.arduino.send_state('standby')

    @pyqtSlot(QImage, int, int, int, list, np.ndarray)
    def on_frame_received(self, q_img, helmet_cnt, head_cnt, total_cnt, detections, raw_frame):
        """Kamera iş parçacığından gelen her yeni kareyi işler ve arayüzü günceller."""
        # 1. Görüntüyü ekrana yansıt
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        self.latest_frame = raw_frame

        # 2. Sayaç Kartlarını Güncelle
        self.val_helmet.setText(str(helmet_cnt))
        self.val_head.setText(str(head_cnt))
        self.val_total.setText(str(total_cnt))

        # 3. İhlal Mantığı ve Alarm Tetikleme (3 Durumlu Profesyonel İSG Hakemi)
        if head_cnt > 0:
            # Durum 1: İHLAL (Baretsiz çalışan tespit edildi)
            self._trigger_hazard_ui(head_cnt)
            self._handle_violation_event(head_cnt, helmet_cnt, detections, raw_frame)
            self.arduino.send_state('hazard')
        elif helmet_cnt > 0:
            # Durum 2: GÜVENLİ (Sadece baretli çalışanlar var, baretsiz yok)
            self._set_safe_ui(helmet_cnt)
            self.arduino.send_state('safe')
        else:
            # Durum 3: BEKLEME (Saha boş veya tespit edilen çalışan yok)
            self._set_standby_ui()
            self.arduino.send_state('standby')

    @pyqtSlot(str)
    def on_camera_error(self, message):
        """Kamera bağlantı hatası oluştuğunda tetiklenir."""
        self.stop_camera()
        QMessageBox.critical(self, "Kamera Hatası", message)

    @pyqtSlot(str)
    def on_status_updated(self, msg):
        """Model veya donanım durum bildirimlerini gösterir."""
        self.status_model_label.setText(msg)

    @pyqtSlot(float)
    def on_fps_measured(self, fps_val):
        """Canlı FPS değerini göstergede günceller."""
        self.fps_badge.setText(f"FPS: {fps_val:.1f}")

    # --------------------------------------------------------------------------
    # İHLAL KAYIT, SESLİ İKAZ VE LOG YÖNETİMİ
    # --------------------------------------------------------------------------
    def _handle_violation_event(self, head_cnt, helmet_cnt, detections, frame):
        """Baretsiz çalışan tespit edildiğinde ses çalar ve kanıt fotoğrafı kaydeder."""
        now = time.time()
        # Cooldown koruması: En az 3 saniye aralıkla disk ve ses tetikle
        if now - self.last_violation_time < self.violation_cooldown:
            return

        self.last_violation_time = now
        self.total_violations_session += 1
        self.lbl_total_violations.setText(f"Oturum Toplam İhlali: {self.total_violations_session}")

        # 1. Sesli İkaz (Arka Planda Donma Yaşanmadan Çalınır)
        self._play_alert_sound()

        # 2. Ekran Görüntüsü Kaydı (ihlal_YYYYMMDD_HHMMSS.jpg)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ihlal_{timestamp_str}.jpg"
        filepath = os.path.join(self.violation_dir, filename)

        # Kanıt fotoğrafına tarih filigranı bas
        save_img = frame.copy()
        h, w, _ = save_img.shape
        stamp_text = f"IHLAL: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Baretsiz: {head_cnt}"
        cv2.putText(save_img, stamp_text, (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        try:
            cv2.imwrite(filepath, save_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception as e:
            print(f"Fotoğraf kaydedilemedi: {e}")

        # En yüksek güven skorunu al
        max_conf = 0.0
        for d in detections:
            if d['class_id'] == 0:
                max_conf = max(max_conf, d['conf'])

        # 3. CSV Kütüğüne Yaz
        try:
            with open(self.csv_log_path, "a", encoding="utf-8") as f:
                date_str = datetime.now().strftime("%Y-%m-%d")
                time_str = datetime.now().strftime("%H:%M:%S")
                f.write(f"{date_str},{time_str},{helmet_cnt},{head_cnt},{max_conf:.2f},{filepath}\n")
        except Exception:
            pass

        # 4. Arayüz Tablosuna Ekle (En Üst Satıra)
        self._add_log_to_table(head_cnt, helmet_cnt, max_conf, filename)

    def _play_alert_sound(self):
        """Hoparlörden ikaz sesi çalar (GUI kilitlenmesini engellemek için daemon thread'de)."""
        def _sound_worker():
            if HAS_WINSOUND:
                try:
                    # Windows için 1200 Hz, 350 ms sesli ikaz tonu
                    winsound.Beep(1200, 350)
                except Exception:
                    pass

        threading.Thread(target=_sound_worker, daemon=True).start()

    def _add_log_to_table(self, head_cnt, helmet_cnt, conf, filename):
        """Olay günlüğü tablosuna yeni ihlali en üst satır olarak ekler."""
        self.log_table.insertRow(0)
        time_text = datetime.now().strftime("%H:%M:%S")

        item_time = QTableWidgetItem(time_text)
        item_time.setTextAlignment(Qt.AlignCenter)

        item_type = QTableWidgetItem("⚠️ BARETSİZ İHLAL")
        item_type.setForeground(QColor("#EF4444"))
        item_type.setTextAlignment(Qt.AlignCenter)

        item_helmet = QTableWidgetItem(str(helmet_cnt))
        item_helmet.setTextAlignment(Qt.AlignCenter)

        item_head = QTableWidgetItem(str(head_cnt))
        item_head.setForeground(QColor("#EF4444"))
        item_head.setTextAlignment(Qt.AlignCenter)

        item_file = QTableWidgetItem(filename)
        item_file.setForeground(QColor("#38BDF8"))

        self.log_table.setItem(0, 0, item_time)
        self.log_table.setItem(0, 1, item_type)
        self.log_table.setItem(0, 2, item_helmet)
        self.log_table.setItem(0, 3, item_head)
        self.log_table.setItem(0, 4, item_file)

    def _take_manual_snapshot(self):
        """Operatörün manuel olarak anlık fotoğraf kaydetmesini sağlar."""
        if self.latest_frame is None:
            QMessageBox.warning(self, "Uyarı", "Aktif kamera karesi bulunamadı!")
            return

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp_str}.jpg"
        filepath = os.path.join(self.violation_dir, filename)

        try:
            cv2.imwrite(filepath, self.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            QMessageBox.information(self, "Fotoğraf Kaydedildi", f"Anlık görüntü başarıyla kaydedildi:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Görüntü kaydedilemedi: {str(e)}")

    # --------------------------------------------------------------------------
    # ALARM BANNER VE GÜVENLİK DURUMU YÖNETİMİ
    # --------------------------------------------------------------------------
    def _trigger_hazard_ui(self, count=1):
        """İhlal anında alarm bannerını anında kırmızıya çevirir ve yanıp sönme modunu başlatır."""
        self.current_hazard_count = count
        if not self.is_flashing:
            self.is_flashing = True
            self.flash_state = True
            self.status_bar_banner.setText(f"🚨 DİKKAT! BARETSİZ ÇALIŞAN TESPİT EDİLDİ! ({count} Kişi)")
            self.status_bar_banner.setObjectName("bannerViolation1")
            self.status_bar_banner.setStyle(self.status_bar_banner.style())
            self.blink_timer.start()

    def _set_safe_ui(self, count=1):
        """Güvenli durumda bannerı yeşile çeker."""
        if self.is_flashing:
            self.is_flashing = False
            self.blink_timer.stop()

        self.status_bar_banner.setText(f"✔ BARETLİ ÇALIŞAN MEVCUT — SAHA GÜVENLİ ({count} Kişi)")
        self.status_bar_banner.setObjectName("bannerSafe")
        self.status_bar_banner.setStyle(self.status_bar_banner.style())

    def _set_standby_ui(self):
        """Çalışan olmadığında bannerı bekleme moduna (mavi/gri) çeker."""
        if self.is_flashing:
            self.is_flashing = False
            self.blink_timer.stop()

        self.status_bar_banner.setText("👁 SAHA İZLENİYOR — AKTİF ÇALIŞAN TESPİT EDİLMEDİ")
        self.status_bar_banner.setObjectName("bannerStandby")
        self.status_bar_banner.setStyle(self.status_bar_banner.style())

    def _toggle_blink(self):
        """Yanıp sönen alarm animasyonu."""
        self.flash_state = not self.flash_state
        cnt = getattr(self, 'current_hazard_count', 1)
        if self.flash_state:
            self.status_bar_banner.setText(f"🚨 DİKKAT! BARETSİZ ÇALIŞAN TESPİT EDİLDİ! ({cnt} Kişi)")
            self.status_bar_banner.setObjectName("bannerViolation1")
        else:
            self.status_bar_banner.setText(f"⚠️ DİKKAT! BARETSİZ ÇALIŞAN TESPİT EDİLDİ! ({cnt} Kişi)")
            self.status_bar_banner.setObjectName("bannerViolation2")
        self.status_bar_banner.setStyle(self.status_bar_banner.style())

    # --------------------------------------------------------------------------
    # ARDUINO BAĞLANTISI VE SERİ PORT YÖNETİMİ
    # --------------------------------------------------------------------------
    def _refresh_serial_ports(self):
        """Sistemdeki aktif COM portlarını combobox'a doldurur."""
        self.combo_ports.clear()
        ports = ArduinoManager.get_available_ports()
        if ports:
            self.combo_ports.addItems(ports)
        else:
            self.combo_ports.addItem("Port Bulunamadı")

    def _toggle_arduino_connection(self):
        """Arduino bağlantısını açar veya kapatır."""
        if not self.arduino.is_connected:
            selected_port = self.combo_ports.currentText()
            if not selected_port or selected_port == "Port Bulunamadı":
                QMessageBox.warning(self, "Port Seçimi", "Lütfen geçerli bir COM portu seçiniz veya yenileyiniz.")
                return

            success, msg = self.arduino.connect(selected_port, baudrate=9600)
            if success:
                self.btn_arduino_connect.setText("🔌 Bağlantıyı Kes")
                self.arduino_status_badge.setText(f"Bağlı ({selected_port} - 9600)")
                self.arduino_status_badge.setObjectName("badgeArduinoOn")
                self.arduino_status_badge.setStyle(self.arduino_status_badge.style())
            else:
                self.arduino_status_badge.setText("Arduino Bağlı Değil (Yazılımsal Mod)")
                self.arduino_status_badge.setObjectName("badgeArduinoOff")
                self.arduino_status_badge.setStyle(self.arduino_status_badge.style())
                QMessageBox.warning(self, "Arduino Bağlantı Uyarısı", f"{msg}\nSistem yazılımsal modda çalışmaya devam edecektir.")
        else:
            self.arduino.disconnect()
            self.btn_arduino_connect.setText("🔌 Arduino'ya Bağlan")
            self.arduino_status_badge.setText("Arduino Bağlı Değil (Yazılımsal Mod)")
            self.arduino_status_badge.setObjectName("badgeArduinoOff")
            self.arduino_status_badge.setStyle(self.arduino_status_badge.style())

    # --------------------------------------------------------------------------
    # AYARLAR VE YARDIMCI METODLAR
    # --------------------------------------------------------------------------
    def _on_conf_changed(self, val):
        """Slider kaydırıldığında güven eşiğini günceller."""
        conf_float = val / 100.0
        self.lbl_conf.setText(f"Güven Eşiği (Conf): %{val}")
        if self.video_thread:
            self.video_thread.set_confidence(conf_float)

    def _update_clock(self):
        """Başlık alanındaki canlı saati günceller."""
        self.clock_badge.setText(QDateTime.currentDateTime().toString("dd.MM.yyyy HH:mm:ss"))

    def closeEvent(self, event):
        """Uygulama kapatılırken tüm iş parçacıklarını ve portları temiz sonlandırır."""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        if self.arduino.is_connected:
            self.arduino.disconnect()
        event.accept()


# ==============================================================================
# 4. UYGULAMA GİRİŞ NOKTASI (MAIN ENTRYPOINT)
# ==============================================================================
if __name__ == "__main__":
    # Yüksek DPI ekranlar için ölçeklendirme ayarları
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
