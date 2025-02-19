# Automatic-Document-Classification-Project

## Introduction
This project focuses on automating the classification of scanned documents using machine learning techniques. It is built using Python and Jupyter Notebook, applying both classical machine learning methods and deep learning (CNNs) to compare their effectiveness. The dataset used is a subset of RVL-CDIP, a large collection of document images.

## Project Workflow

### Step 1: Creating a Mini Dataset
To streamline the training process and reduce computational requirements, a smaller dataset is created by selecting and copying a subset of the RVL-CDIP dataset. This step involves:
- **File manipulation**: Copying selected images into a new directory.
- **Updating labels**: Ensuring that label files correctly map to the newly created dataset.
- **Libraries used**: `os`, `shutil`, `random`.

### Step 2: Data Processing and Feature Extraction
Once the dataset is prepared, it undergoes preprocessing to ensure that the models receive well-structured input. The key steps include:
- **Label handling**: Using `pandas` to read and clean metadata.
- **Image preprocessing**:
  - Converting images to grayscale.
  - Resizing images to a uniform shape.
  - Normalizing pixel values for CNNs.
  - (Optional) Applying text extraction techniques if NLP-based classification is used.
- **Libraries used**: `Pandas`, `Pillow`, `OpenCV`, `numpy`.

### Step 3: Model Building and Evaluation
Multiple machine learning models are implemented and compared for classification accuracy:
- **Baseline ML models**:
  - Decision Trees
  - Support Vector Machines (SVM)
  - Logistic Regression
  - Hyperparameter tuning using `GridSearchCV`.
- **Deep Learning model**:
  - A Convolutional Neural Network (CNN) built with `TensorFlow/Keras`.
- **Evaluation metrics**:
  - Accuracy, precision, recall, F1-score.
  - K-Fold cross-validation for classical models.
- **Libraries used**: `Scikit-learn`, `TensorFlow/Keras`, `Matplotlib`.

## Technologies Used
- **Programming Language**: Python
- **Development Environment**: Jupyter Notebook
- **Data Handling**: Pandas, NumPy
- **Image Processing**: OpenCV, Pillow
- **Machine Learning**: Scikit-learn
- **Deep Learning**: TensorFlow, Keras
- **Visualization**: Matplotlib, Seaborn

## Results and Key Findings
The project compares classical ML approaches with deep learning, analyzing the trade-offs in terms of accuracy and computational efficiency. The CNN-based approach shows superior performance but requires significantly more computational resources.

## Future Improvements
- Expanding NLP techniques for text-based feature extraction.
- Implementing transfer learning with pre-trained CNN models.
- Optimizing dataset preparation for larger-scale deployment.
