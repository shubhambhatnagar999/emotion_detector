
# Emotion Detection from Facial Expressions using CNN

This project is a Deep Learning-based Emotion Detection system that classifies facial expressions into multiple emotion categories (like Happy, Sad, Angry, etc.) using a Convolutional Neural Network (CNN).

## Dataset Structure

The dataset used is structured into subfolders per emotion class and split into `train` and `test` directories:

```
dataset_path/
│
├── train/
│   ├── angry/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── ... (other emotion folders)
│
└── test/
    ├── angry/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── ...
```

Each folder contains grayscale images of faces labeled with one of the 7 emotion categories.

## Model Architecture

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

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/shubhambhatnagar999/emotion_detector.git
cd emotion_detector
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

> If you’re using `pyenv` or virtualenv, activate it first.

### 3. Train the model

Make sure your dataset is in the correct folder structure. Then:

```bash
python train.py
```

### 4. Test prediction on a single image

```bash
python predict.py
```

> Ensure your test image is provided inside `predict.py` or via argument.

## Results

Training and validation accuracy/loss are plotted after training using `matplotlib`.

- Model is saved as: `face_expression_model.h5`
- Class indices are saved in: `class_indices.json`

## Dependencies

- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`
- `opencv-python` *(optional for image preview)*

Install with:

```bash
pip install tensorflow keras numpy matplotlib opencv-python
```

## Future Improvements

- Use pre-trained CNN (like VGG or ResNet) for better accuracy
- Add data augmentation
- Convert to a web app using Flask or Streamlit
- Deploy the model via cloud or Docker

## Credits

This project was developed by [Shubham Bhatnagar](https://github.com/shubhambhatnagar999) as part of deep learning practice using TensorFlow/Keras.

## License

This project is open-source and available under the [MIT License](LICENSE).
