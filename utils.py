import cv2
import numpy as np

def preprocess_canvas(canvas):
    # Convert to grayscale
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Resize to MNIST size
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize
    normalized = resized / 255.0

    # Reshape to CNN format
    return normalized.reshape(1, 28, 28, 1)
