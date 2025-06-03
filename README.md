# Live Object Detection (LOD)

LOD is a simple collection of scripts that demonstrate how to create a small object detection pipeline with Python. The tools allow you to capture frames from a webcam, annotate regions of interest with text labels and train a lightweight model to recognise those objects.

## Features

- **Data Generation** (`dataGen.py`): capture frames from your webcam and draw bounding boxes around objects while assigning a text label to each box. The annotations are saved alongside the captured frame.
- **Model Training** (`modelMaker.py` / `trainer.py`): use the saved annotations to train either a classical SVM model (`modelMaker.py`) or a small neural network (`trainer.py`).
- **Live Detection** (`cam.py`): load the trained model and perform real-time predictions on webcam frames.
- **Face Detection Example** (`main.py`): a minimal script showcasing Haar-cascade based face detection.

## Requirements

 - Python 3.12+
- OpenCV
- TensorFlow
- NumPy
- scikit-learn

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

1. **Annotate Images**
   ```bash
   python dataGen.py
   ```
   Press the left mouse button to freeze the frame, draw a rectangle around the object and enter a label. Press `s` to save the frame with the annotations. Press `q` to quit.

2. **Train a Model**
   ```bash
   python modelMaker.py --data-dir path/to/annotated/images --model-path my_model.pkl
   ```
   The script trains an SVM on the labelled regions and saves the classifier.

   Alternatively, you can train a small CNN:
   ```bash
   python trainer.py
   ```

3. **Run Live Detection**
   ```bash
   python cam.py --model-path my_model.pkl
   ```

## Notes

- The provided scripts are minimal examples intended for experimentation. For better results you may need more data and more advanced models.
- Modify the paths in the scripts or pass them via command line arguments to suit your setup.

