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
        ret, frame = camera.read()
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
        for x, y, w, h in eyes:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            eyess = eye_cascade.detectMultiScale(roi_gray)
            if len(eyess) == 0:
                print('eyes are not detected')
            else:
                for (ex, ey, ew, eh) in eyess:
                    eyes_roi = roi_color[ey:ey + eh, ex:ex+ew]

        final_image = cv2.resize(eyes_roi, (224, 224,))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image/255.0

        predictions = new_model.predict(final_image)
        if (predictions[0][0] < 0.3):
            status = "Open Eyes"
        else:
            status = "Closed Eyes"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, status, (50, 50), font, 3,
                    (0, 0, 255), 2, cv2.LINE_4)

        # cv2.imshow('Driver Drowsiness detection', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # if cv2.waitKey(2) & 0xFF == ord('q'):
        #   break


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
