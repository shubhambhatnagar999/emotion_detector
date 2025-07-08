import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

# Load model
model = tf.keras.models.load_model('face_expression_model.h5')

# Load class indices (emotion labels)
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse the class_indices dict to map from index to label
emotion_map = {v: k for k, v in class_indices.items()}

# Load and preprocess image
img_path = 'test.jpg'  # your test image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48, 48))
img = img / 255.0
img = np.reshape(img, (1, 48, 48, 1))

# Predict
predictions = model.predict(img)[0]
predicted_index = np.argmax(predictions)
predicted_label = emotion_map[predicted_index]

# 🔍 Print prediction
print(f"Predicted Emotion: {predicted_label}")
print("Probabilities:", predictions)

# 🧠 Plot prediction probabilities
plt.figure(figsize=(8, 4))
plt.bar(range(len(predictions)), predictions, tick_label=list(emotion_map.values()))
plt.title("Emotion Prediction Probabilities")
plt.xlabel("Emotions")
plt.ylabel("Confidence")
plt.grid(True)
plt.tight_layout()
plt.show()

# 🖼️ Optional: Show original image with predicted label
img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()
