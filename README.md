# HandGesture

## Introduction

This project reads hand gestures from video feeds. Because of COVID19, everyone uses Zoom ...

The hand gestures are: **one, two, three, thumb up, thumb down, hand raise**.  MediaPipe [[1]](#References) was used to figure the hand position. K-Nearest Neighbor classifer [[2]](#References) was used to determine the hand gesture for the complex method

---

## Requirements

- python3
- mediapipe
- sklearn
- matplotlib
- cv2 (`pip install opencv-python`)
- jupyter

---

## How to Run

### Simple Method

`python simple_method.py`

### Complex Method

`python complex_method.py`

---

## Results

### Accuracy

- simple method: 91.0%
- complex method: 95.0%

### Confusion Matrix

- simple method:
  - ![simple_confusion](simple_confusion.png)
- complex method:
  - ![complex_confusion](complex_confusion.png)

|class 1|class 2|class 3|class 4|class 5|
|-------|-------|-------|-------|-------|
|10.0   |0.0    |0.0    |0.0    |0.0    |
|0.0    |10.0   |0.0    |0.0    |0.0    |
|0.0    |0.0    |10.0   |0.0    |0.0    |

---

## Conclusion

RadiusNeighbor classifier is more accurate

---

## References

[1] [MediaPipe](https://google.github.io/mediapipe/)

[2] [KNNs](https://scikit-learn.org/stable/modules/neighbors.html)