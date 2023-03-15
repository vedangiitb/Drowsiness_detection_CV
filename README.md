# Real Time Drowsiness detection

This repo contains my project for real time Drowsiness detection using computer vision. 

## Training Dataset
I've used MRL eye dataset which contains over 15000 images of open and closed eyes. The model is trained on 3000 images(1,500 open eyes + 1,500 closed eyes).

## Working
We first detect the face using Haarcascade frontalface detection model and then detect eyes using eye cascade model. Finally we apply our trained model on eyes to check if the eyes are closed. The model for classifying closed and open eyes is built by performing transfer learning on the MobileNet model. I've changed last 3 layers to classify closed eyes and open eyes. 

