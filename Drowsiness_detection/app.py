from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import numpy as np

new_model = keras.models.load_model('my_model.h5')
app = Flask(__name__)
camera = cv2.VideoCapture(0)


def generate_frames():
    while True:
        cbs = 0
        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            detector = cv2.CascadeClassifier(
                'haarcascade/haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(
                'haarcascade/haarcascade_eye.xml')
            faces = detector.detectMultiScale(frame, 1.1, 7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Draw rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                if len(eyes) == 0:
                    print('Eyes not detected')
                    cbs = 0
                    break

                for (ex, ey, ew, eh) in eyes:
                    cbs = 1
                    cv2.rectangle(roi_color, (ex, ey),
                                  (ex+ew, ey+eh), (0, 255, 0), 2)
                    eyes_roi = roi_color[ey:ey+eh, ex:ex+ew]

        if cbs == 1:
            final_image = cv2.resize(eyes_roi, (224, 224))
            final_image = np.expand_dims(final_image, axis=0)
            final_image = final_image/255.0

            predictions = new_model.predict(final_image)
            if (predictions[0][0] < 0.3):
                status = "Open Eyes"
            else:
                status = "Closed Eyes"

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, status, (50, 50), font,
                        3, (0, 0, 255), 2, cv2.LINE_4)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
