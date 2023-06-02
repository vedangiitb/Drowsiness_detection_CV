# Real Time Drowsiness detection

This repo contains my project for real time Drowsiness detection using computer vision. 

## Training Dataset

This is the link for MRL eye dataset
http://mrl.cs.vsb.cz/eyedataset#:~:text=Therefore%2C%20we%20introduce%20the%20MRL%20Eye%20Dataset%2C%20the,suitable%20for%20testing%20several%20features%20or%20trainable%20classifiers.

## Working
We first detect the face using Haarcascade frontalface detection model and then detect eyes using eye cascade model. Finally we apply our trained model on eyes to check if the eyes are closed. The model for classifying closed and open eyes is built by performing transfer learning on the MobileNet model. I've changed last 3 layers to classify closed eyes and open eyes. 

