# 🧠 Tumor Detection using Logistic Regression

This project implements a binary classifier to predict whether a tumor is malignant or benign using logistic regression.  
The model is trained on a dataset inspired by Andrew Ng’s Machine Learning course.

## deployed link -> https://tumour-detector.onrender.com

## 🔍 About the Project

- **Algorithm**: Logistic Regression (using scikit-learn)
- **Data**: 100 samples of tumor size and texture, labeled as benign (0) or malignant (1)
- **Goal**: Predict tumor type and visualize the decision boundary

## 📈 Model Performance

- **Training Accuracy**: 93.7%
- **Input Features**: Tumor size and texture
- **Output**: Probability of malignancy

## 📁 Files

- `tumor_classifier.ipynb`: Notebook with data loading, model training, and visualization
- `ex2data1.txt`: Dataset (adapted from Andrew Ng’s ML course)

## 🚀 How to Run

1. Clone this repository or download the files.
2. Open `tumor_classifier.ipynb` in Jupyter Notebook.
3. Make sure `ex2data1.txt` is in the same directory.
4. Run all the cells in the notebook.

## 📊 Example Output

- Predicted probability for tumor (size = 45, texture = 85): ~0.89
- Decision boundary plotted on training data

## 🛠️ Libraries Used

- Python 3.x
- NumPy
- Matplotlib
- scikit-learn

## 🧠 Inspiration

Inspired by [Andrew Ng's Machine Learning course on Coursera](https://www.coursera.org/learn/machine-learning)

---

Feel free to fork this project or contribute!
