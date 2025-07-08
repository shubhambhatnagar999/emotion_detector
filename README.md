<<<<<<< HEAD
# Emotion Detection from Facial Images

This project uses a CNN model to classify facial expressions into 7 emotion categories using grayscale images.

## Folder Structure
- `src/train.py` - for training the model
- `src/predict.py` - for predicting from images
- `models/` - stores the trained `.h5` model
- `dataset/` - contains images organized by emotion labels

## How to Run
1. Install dependencies:
=======
# emotion_detector
>>>>>>> d11b0b9630e7c817cff51234505e921bc9f8e141
 dataset_path/
│
├── train/
│ ├── angry/
│ ├── happy/
│ ├── sad/
│ ├── surprise/
│ └── ... (other emotion folders)
│
└── test/
├── angry/
├── happy/
├── sad/
├── surprise/
└── ...

yaml
Copy
Edit

Each folder contains grayscale images of faces labeled with one of the 7 emotion categories.

---

## 🧠 Model Architecture

- Input size: `48x48` grayscale images
- Layers:
  - `Conv2D` + `MaxPooling`
  - `Conv2D` + `MaxPooling`
  - `Flatten`
  - `Dense(128)` + `Dropout`
  - `Dense(7)` + `Softmax`

The model is trained using:
- Optimizer: `Adam`
- Loss: `categorical_crossentropy`
- Metrics: `accuracy`

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/shubhambhatnagar999/emotion_detector.git
cd emotion_detector
2. Install requirements
bash
Copy
Edit
pip install -r requirements.txt
If you’re using pyenv or virtualenv, activate it first.

3. Train the model
Make sure your dataset is in the correct folder structure. Then:

bash
Copy
Edit
python train.py
4. Test prediction on a single image
bash
Copy
Edit
python predict.py
