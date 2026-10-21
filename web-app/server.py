import atexit
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, send_from_directory

from camera import OUTPUT_DIR, CameraStream

app = Flask(__name__)
camera = CameraStream()


def _placeholder_frame():
    img = np.full((480, 640, 3), 24, dtype=np.uint8)
    cv2.putText(img, "Kamera belum aktif", (130, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 180), 2)
    cv2.putText(img, "Klik 'Mulai Kamera' di panel kanan", (80, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (130, 130, 130), 2)
    return img


PLACEHOLDER_JPEG = cv2.imencode(".jpg", _placeholder_frame())[1].tobytes()


@app.route("/")
def index():
    return render_template("index.html")


def gen_frames():
    while True:
        jpeg = camera.get_jpeg()
        if jpeg is None:
            jpeg = PLACEHOLDER_JPEG
            time.sleep(0.3)
        else:
            time.sleep(0.03)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify(camera.get_status())


@app.route("/start", methods=["POST"])
def start_camera():
    ok, error = camera.start()
    return jsonify({"ok": ok, "error": error})


@app.route("/stop", methods=["POST"])
def stop_camera():
    camera.stop()
    return jsonify({"ok": True})


@app.route("/save", methods=["POST"])
def save():
    ok, message, info = camera.save_frame()
    return jsonify({"ok": ok, "message": message, "info": info})


@app.route("/reset", methods=["POST"])
def reset():
    camera.reset()
    return jsonify({"ok": True})


@app.route("/output/<path:filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@atexit.register
def _cleanup():
    camera.stop()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, threaded=True, debug=False)
