
import os
import cv2
import time
import math
import queue
import threading
import numpy as np
import requests
import streamlit as st
import mediapipe as mp
import paho.mqtt.client as mqtt
import streamlit.components.v1 as components

try:
    import av
    from streamlit_webrtc import (
        RTCConfiguration,
        VideoProcessorBase,
        WebRtcMode,
        webrtc_streamer,
    )
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

try:
    from playsound import playsound
except Exception:  # server cloud/headless biasanya tidak punya audio
    playsound = None

from streamlit_autorefresh import st_autorefresh


# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="Deteksi Kantuk Pengemudi",
    page_icon="🚗",
    layout="wide",
)

st.markdown("""
<style>
html, body, [class*="st-"] {
    color: #00FF00 !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stCaptionContainer"],
h1, h2, h3, p, label {
    color: #00FF00 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🚗 Sistem Deteksi Kantuk Pengemudi")
st.caption(
    "Kamera HP (realtime) / laptop · EAR · MAR · Alarm · Telegram · MQTT"
)


# =========================================================
# KONFIGURASI
# =========================================================
EAR_THRESHOLD = 0.15
MAR_THRESHOLD = 0.50
EYE_CLOSED_TIME_THRESHOLD = 1.0
TELEGRAM_COOLDOWN = 10

ALARM_FILE = "alarm.wav"

# Isi melalui environment variable.
# Jangan menuliskan token bot rahasia di dalam kode.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8872341923:AAGM9NRTY2evsZYsXZMHigXlk2OnvS-NOmA")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "6651250912")

MQTT_ENABLED = True
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC_STATUS = "deteksi/kantuk/status"


# =========================================================
# GEOMETRI
# =========================================================
def euclidean(p1, p2):
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1],
    )


def compute_ear(points):
    numerator = (
        euclidean(points[1], points[5])
        + euclidean(points[2], points[4])
    )
    denominator = 2.0 * euclidean(points[0], points[3])

    if denominator == 0:
        return 0.0

    return numerator / denominator


def get_mouth_points(landmarks, width, height):
    pairs_idx = [
        (13, 14),
        (78, 308),
        (81, 311),
        (82, 312),
    ]

    pairs = [
        (
            (int(landmarks[a].x * width),
             int(landmarks[a].y * height)),
            (int(landmarks[b].x * width),
             int(landmarks[b].y * height)),
        )
        for a, b in pairs_idx
    ]

    left = (
        int(landmarks[61].x * width),
        int(landmarks[61].y * height),
    )
    right = (
        int(landmarks[291].x * width),
        int(landmarks[291].y * height),
    )

    return pairs, left, right


def compute_mar(vertical_pairs, left, right):
    horizontal = euclidean(left, right)

    if horizontal == 0 or not vertical_pairs:
        return 0.0

    average_vertical = sum(
        euclidean(a, b) for a, b in vertical_pairs
    ) / len(vertical_pairs)

    return average_vertical / horizontal


# =========================================================
# STATUS BERSAMA ANTAR-THREAD
# =========================================================
def new_shared_state():
    return {
        "lock": threading.Lock(),
        "frame": None,
        "status": "Menunggu kamera",
        "ear": 0.0,
        "mar": 0.0,
        "eye": 0.0,
        "mouth": 0.0,
        "camera_ok": False,
        "error": "",
        "running": True,
    }


# =========================================================
# TELEGRAM
# =========================================================
def telegram_worker(alert_queue):
    while True:
        item = alert_queue.get()

        try:
            if item is None:
                return

            token, chat_id, image_bytes, caption = item

            url = (
                f"https://api.telegram.org/"
                f"bot{token}/sendPhoto"
            )

            requests.post(
                url,
                files={
                    "photo": (
                        "alert.jpg",
                        image_bytes,
                        "image/jpeg",
                    )
                },
                data={
                    "chat_id": chat_id,
                    "caption": caption,
                },
                timeout=10,
            ).raise_for_status()

        except Exception as exc:
            print("Telegram error:", exc)

        finally:
            alert_queue.task_done()


# =========================================================
# ALARM
# =========================================================
def alarm_worker(stop_event):
    while not stop_event.is_set():
        if not os.path.isfile(ALARM_FILE):
            stop_event.wait(1)
            continue

        try:
            playsound(ALARM_FILE)
        except Exception as exc:
            print("Alarm error:", exc)
            stop_event.wait(1)


# =========================================================
# MESIN DETEKSI KAMERA LAPTOP
# =========================================================
class DrowsinessEngine:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self, shared=None, server_alarm=True):
        self.shared = shared
        self.server_alarm = server_alarm
        self.eye_closed_start = None
        self.last_telegram_time = 0
        self.previous_mqtt_status = ""
        self.alarm_thread = None
        self.alarm_stop = threading.Event()
        self.alert_queue = queue.Queue(maxsize=2)

        self.alert_thread = threading.Thread(
            target=telegram_worker,
            args=(self.alert_queue,),
            daemon=True,
        )
        self.alert_thread.start()

        self.mqtt_client = None

        if MQTT_ENABLED:
            try:
                self.mqtt_client = mqtt.Client()
                self.mqtt_client.connect(
                    MQTT_BROKER,
                    MQTT_PORT,
                    60,
                )
                self.mqtt_client.loop_start()
            except Exception as exc:
                print("MQTT connection error:", exc)
                self.mqtt_client = None

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def set_status(self, status):
        if status == self.previous_mqtt_status:
            return

        if self.mqtt_client is not None:
            mqtt_status = None

            if "Mengantuk" in status:
                mqtt_status = "MENGANTUK"
            elif "Waspada" in status:
                mqtt_status = "WASPADA"
            elif status == "Normal":
                mqtt_status = "NORMAL"

            if mqtt_status:
                try:
                    self.mqtt_client.publish(
                        MQTT_TOPIC_STATUS,
                        mqtt_status,
                    )
                except Exception as exc:
                    print("MQTT publish error:", exc)

        self.previous_mqtt_status = status

    def start_alarm(self):
        # Pada mode kamera HP, suara alarm diputar di browser HP,
        # bukan di server.
        if not self.server_alarm or playsound is None:
            return

        if (
            self.alarm_thread is not None
            and self.alarm_thread.is_alive()
        ):
            return

        self.alarm_stop.clear()

        self.alarm_thread = threading.Thread(
            target=alarm_worker,
            args=(self.alarm_stop,),
            daemon=True,
        )
        self.alarm_thread.start()

    def stop_alarm(self):
        self.alarm_stop.set()

    def close(self):
        self.stop_alarm()

        try:
            self.face_mesh.close()
        except Exception:
            pass

        if self.mqtt_client is not None:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception:
                pass

        try:
            self.alert_queue.put_nowait(None)
        except queue.Full:
            pass

    def send_telegram(self, jpeg_bytes, caption):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return

        now = time.time()

        if now - self.last_telegram_time < TELEGRAM_COOLDOWN:
            return

        try:
            self.alert_queue.put_nowait((
                TELEGRAM_BOT_TOKEN,
                TELEGRAM_CHAT_ID,
                jpeg_bytes,
                caption,
            ))
            self.last_telegram_time = now
        except queue.Full:
            pass

    def process_frame(self, image):
        height, width = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        status = "Wajah tidak terdeteksi"
        ear = None
        mar = None
        eye_confidence = 0.0
        mouth_confidence = 0.0

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            landmarks = [
                (
                    int(point.x * width),
                    int(point.y * height),
                )
                for point in face.landmark
            ]

            left_eye = [
                landmarks[i] for i in self.LEFT_EYE
            ]
            right_eye = [
                landmarks[i] for i in self.RIGHT_EYE
            ]

            ear = (
                compute_ear(left_eye)
                + compute_ear(right_eye)
            ) / 2.0

            now = time.time()

            if ear < EAR_THRESHOLD:
                if self.eye_closed_start is None:
                    self.eye_closed_start = now

                eye_closed = (
                    now - self.eye_closed_start
                    >= EYE_CLOSED_TIME_THRESHOLD
                )
            else:
                self.eye_closed_start = None
                eye_closed = False

            mouth_pairs, mouth_left, mouth_right = (
                get_mouth_points(
                    face.landmark,
                    width,
                    height,
                )
            )

            mar = compute_mar(
                mouth_pairs,
                mouth_left,
                mouth_right,
            )

            yawning = mar > MAR_THRESHOLD

            if eye_closed:
                status = (
                    "Mengantuk (Mata Terpejam + Menguap) Diharapkan pengguna berhenti sejenak"
                    if yawning
                    else "Mengantuk (Mata Terpejam) Diharapkan pengguna berhenti sejenak"
                )
            elif yawning:
                status = "Waspada (Menguap)"
            else:
                status = "Normal"

            # Skor indikator berdasarkan ambang, bukan
            # probabilitas atau akurasi model.
            if ear < EAR_THRESHOLD:
                eye_confidence = min(
                    100.0,
                    max(
                        0.0,
                        (1 - ear / EAR_THRESHOLD) * 100,
                    ),
                )

            if mar > MAR_THRESHOLD:
                mouth_confidence = min(
                    100.0,
                    max(
                        0.0,
                        (mar / MAR_THRESHOLD - 1) * 100,
                    ),
                )

            # Titik mata berwarna hijau.
            for point in left_eye + right_eye:
                cv2.circle(
                    image, point, 2, (0, 255, 0), -1
                )

            # Garis indikator mulut.
            cv2.line(
                image,
                mouth_left,
                mouth_right,
                (255, 255, 255),
                2,
            )

            for p1, p2 in mouth_pairs:
                cv2.line(
                    image, p1, p2, (255, 0, 0), 1
                )

            if "Mengantuk" in status or "Waspada" in status:
                self.start_alarm()

                ok, buffer = cv2.imencode(".jpg", image)

                if ok:
                    self.send_telegram(
                        buffer.tobytes(),
                        (
                            f"PERINGATAN: {status}\n"
                            f"EAR: {ear:.3f} | MAR: {mar:.3f}"
                        ),
                    )
            else:
                self.stop_alarm()

            self.set_status(status)

        else:
            self.eye_closed_start = None
            self.stop_alarm()

        # Warna status: merah, oranye, atau hijau.
        color = (
            (0, 0, 255)
            if "Mengantuk" in status
            else (0, 165, 255)
            if "Waspada" in status
            else (0, 255, 0)
        )

        lines = [
            f"Status: {status}",
            f"EAR: {ear:.3f}" if ear is not None else "EAR: -",
            f"MAR: {mar:.3f}" if mar is not None else "MAR: -",
            f"Confidence Eye: {eye_confidence:.1f}%",
            f"Confidence Mouth: {mouth_confidence:.1f}%",
        ]

        for i, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color if i == 0 else (0, 255, 0),
                2,
            )

        return image, {
            "status": status,
            "ear": ear if ear is not None else 0.0,
            "mar": mar if mar is not None else 0.0,
            "eye": eye_confidence,
            "mouth": mouth_confidence,
        }

    def run(self):
        camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        # Fallback untuk lingkungan yang tidak mendukung
        # backend DirectShow, misalnya beberapa OS selain Windows.
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(1)
            
            
# Gunakan USB webcam eksternal
#         camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# 
# # Fallback jika backend DirectShow gagal
#         if not camera.isOpened():
#             camera.release()
#             camera = cv2.VideoCapture(1)
# 
#         if not camera.isOpened():
#             with self.shared["lock"]:
#                 self.shared["error"] = (
#                     "USB webcam tidak ditemukan. "
#                     "Periksa kabel USB dan indeks kamera."
#                 )
#                 self.shared["camera_ok"] = False
#                 self.shared["status"] = "Kamera gagal dibuka"
# 
#         self.face_mesh.close()
#         return


        if not camera.isOpened():
            with self.shared["lock"]:
                self.shared["error"] = (
                    "Kamera laptop gagal dibuka. "
                    "Periksa izin kamera atau indeks kamera."
                )
                self.shared["camera_ok"] = False
                self.shared["status"] = "Kamera gagal dibuka"
            self.face_mesh.close()
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        with self.shared["lock"]:
            self.shared["camera_ok"] = True
            self.shared["error"] = ""

        try:
            while self.shared["running"]:
                success, frame = camera.read()

                if not success:
                    time.sleep(0.05)
                    continue

                try:
                    processed, metrics = self.process_frame(frame)

                    ok, buffer = cv2.imencode(
                        ".jpg",
                        processed,
                        [cv2.IMWRITE_JPEG_QUALITY, 75],
                    )

                    if not ok:
                        continue

                    # Simpan frame dan semua metrik sekaligus
                    # supaya dashboard membaca data yang konsisten.
                    with self.shared["lock"]:
                        self.shared["frame"] = buffer.tobytes()
                        self.shared["status"] = metrics["status"]
                        self.shared["ear"] = metrics["ear"]
                        self.shared["mar"] = metrics["mar"]
                        self.shared["eye"] = metrics["eye"]
                        self.shared["mouth"] = metrics["mouth"]

                except Exception as exc:
                    print("Frame processing error:", exc)

        finally:
            camera.release()
            self.stop_alarm()
            self.face_mesh.close()

            if self.mqtt_client is not None:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()


# =========================================================
# MODE KAMERA HP / BROWSER (WebRTC)
# =========================================================
RTC_CONFIGURATION = None
if WEBRTC_AVAILABLE:
    RTC_CONFIGURATION = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
            ]
        }
    )

EMPTY_METRICS = {
    "status": "Menunggu kamera",
    "ear": 0.0,
    "mar": 0.0,
    "eye": 0.0,
    "mouth": 0.0,
}

if WEBRTC_AVAILABLE:

    class DrowsinessVideoProcessor(VideoProcessorBase):
        """Menerima frame video dari kamera HP lewat browser,
        memprosesnya, lalu mengirim balik frame beranotasi."""

        def __init__(self):
            self.engine = DrowsinessEngine(server_alarm=False)
            self._lock = threading.Lock()
            self._metrics = dict(EMPTY_METRICS)

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")

            try:
                image, metrics = self.engine.process_frame(image)

                with self._lock:
                    self._metrics = metrics
            except Exception as exc:
                print("Frame processing error:", exc)

            return av.VideoFrame.from_ndarray(image, format="bgr24")

        def get_metrics(self):
            with self._lock:
                return dict(self._metrics)

        def on_ended(self):
            self.engine.close()


# Alarm suara di browser (HP). Dibunyikan oleh JavaScript.
ALARM_HTML = """
<script>
(function () {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    function beep() {
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.type = "square";
      o.frequency.value = 880;
      g.gain.value = 0.3;
      o.connect(g);
      g.connect(ctx.destination);
      o.start();
      setTimeout(() => o.stop(), 400);
      try { navigator.vibrate && navigator.vibrate([300, 100, 300]); } catch (e) {}
    }
    ctx.resume();
    beep();
    setInterval(beep, 800);
  } catch (e) {}
})();
</script>
"""


def render_status(box, status):
    if "Mengantuk" in status:
        box.error(f"🚨 {status}")
    elif "Waspada" in status:
        box.warning(f"⚠️ {status}")
    elif status == "Normal":
        box.success("✅ Normal")
    else:
        box.info(status)


def run_phone_mode():
    st.subheader("📱 Kamera HP (Realtime)")

    if not WEBRTC_AVAILABLE:
        st.error(
            "Library belum terpasang. Jalankan: "
            "pip install streamlit-webrtc av"
        )
        return

    st.info(
        "Buka aplikasi ini dari browser HP (Chrome/Safari), tekan "
        "**START**, lalu izinkan akses kamera. Akses kamera di HP "
        "hanya berfungsi lewat **HTTPS** (atau localhost)."
    )

    facing_label = st.sidebar.radio(
        "Kamera HP",
        ["Depan (selfie)", "Belakang"],
        index=0,
    )
    facing = "user" if facing_label.startswith("Depan") else "environment"

    browser_alarm = st.sidebar.checkbox(
        "🔊 Alarm suara di HP",
        value=True,
        help="Centang/ketuk sekali agar browser mengizinkan suara.",
    )

    ctx = webrtc_streamer(
        key=f"drowsiness-phone-{facing}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=DrowsinessVideoProcessor,
        media_stream_constraints={
            "video": {
                "facingMode": facing,
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15},
            },
            "audio": False,
        },
        async_processing=True,
    )

    st.subheader("📊 Hasil Deteksi Real-Time")

    status_box = st.empty()
    col1, col2, col3 = st.columns(3)
    ear_box = col1.empty()
    mar_box = col2.empty()
    state_box = col3.empty()
    col4, col5 = st.columns(2)
    eye_box = col4.empty()
    mouth_box = col5.empty()
    alarm_slot = st.empty()

    def show(m):
        render_status(status_box, m["status"])
        ear_box.metric("EAR", f"{m['ear']:.3f}")
        mar_box.metric("MAR", f"{m['mar']:.3f}")
        state_box.metric("Status", m["status"])
        eye_box.metric("Confidence Eye", f"{m['eye']:.1f}%")
        mouth_box.metric("Confidence Mouth", f"{m['mouth']:.1f}%")

    show(EMPTY_METRICS)

    alarm_on = False

    while ctx.state.playing:
        processor = ctx.video_processor
        metrics = (
            processor.get_metrics()
            if processor is not None
            else EMPTY_METRICS
        )

        show(metrics)

        drowsy = (
            "Mengantuk" in metrics["status"]
            or "Waspada" in metrics["status"]
        )

        if drowsy and browser_alarm and not alarm_on:
            with alarm_slot.container():
                components.html(ALARM_HTML, height=0)
            alarm_on = True
        elif (not drowsy or not browser_alarm) and alarm_on:
            alarm_slot.empty()
            alarm_on = False

        time.sleep(0.3)

    alarm_slot.empty()

    st.caption(
        "Video diproses di server. Jika koneksi WebRTC gagal di "
        "jaringan seluler tertentu, biasanya diperlukan server TURN."
    )


# =========================================================
# MODE KAMERA LAPTOP (lokal, seperti sebelumnya)
# =========================================================
@st.cache_resource
def start_detection_engine():
    shared = new_shared_state()
    engine = DrowsinessEngine(shared)

    thread = threading.Thread(
        target=engine.run,
        daemon=True,
    )
    thread.start()

    return shared


def run_laptop_mode():
    shared = start_detection_engine()

    st.subheader("📷 Kamera Laptop dan Hasil Deteksi")

    st_autorefresh(
        interval=500,
        key="live_dashboard_refresh",
    )

    with shared["lock"]:
        frame_bytes = shared["frame"]
        status = shared["status"]
        ear = shared["ear"]
        mar = shared["mar"]
        eye = shared["eye"]
        mouth = shared["mouth"]
        camera_ok = shared["camera_ok"]
        error = shared["error"]

    if error:
        st.error(error)
    elif camera_ok:
        st.success("Kamera laptop aktif")
    else:
        st.info("Menghubungkan ke kamera laptop...")

    if frame_bytes:
        st.image(
            frame_bytes,
            caption="Video hasil deteksi dari kamera laptop",
            use_container_width=True,
        )
    else:
        st.warning(
            "Menunggu frame kamera pertama. "
            "Pastikan kamera tidak sedang dipakai aplikasi lain."
        )

    st.subheader("📊 Hasil Deteksi Real-Time")

    col1, col2, col3 = st.columns(3)
    col1.metric("EAR", f"{ear:.3f}")
    col2.metric("MAR", f"{mar:.3f}")
    col3.metric("Status", status)

    col4, col5 = st.columns(2)
    col4.metric("Confidence Eye", f"{eye:.1f}%")
    col5.metric("Confidence Mouth", f"{mouth:.1f}%")


# =========================================================
# PILIH MODE
# =========================================================
mode = st.sidebar.radio(
    "Sumber kamera",
    ["📱 Kamera HP (Realtime)", "💻 Kamera Laptop (lokal)"],
    index=0,
)

if mode.startswith("📱"):
    run_phone_mode()
else:
    run_laptop_mode()

st.caption(
    "EAR/MAR menggunakan ambang batas. Confidence Eye dan "
    "Confidence Mouth merupakan skor indikator, bukan probabilitas "
    "medis atau akurasi model."
)
