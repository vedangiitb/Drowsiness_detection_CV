# Drowsiness_detection_CV

This repo contains my project for real time Drowsiness detection using computer vision. 

## Training Dataset
The model is trained on MRL Eye dataset, which contains approx 15,000 images of closed and open eyes. The model is trained on 3000 images(1,500 open eyes + 1,500 closed eyes).

## Working
We first detect the face using Haarcascade frontalface detection model and then detect eyes using eye cascade model. Finally we apply our trained model on eyes to check if the eyes are closed. The model for classifying closed and open eyes is built by performing transfer learning on the MobileNet model. I've changed last 3 layers to classify closed eyes and open eyes. 

