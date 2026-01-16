<h1>Fake vs Real Video Detection</h1>

<h2>Project Overview</h2>

This project builds an end-to-end machine learning pipeline to classify videos as real or fake (deepfake-style detection) using preprocessed numerical features extracted from videos.

The work so far focuses on:
* Clean project structure
* Reproducible preprocessing
* Training a baseline ML/DL model
* Running inference and generating a CSV of predictions

<hr>

<h2>What Problem Are We Solving?</h2>

We are solving a binary classification problem:

* 0 → Real video
* 1 → Fake video

Each video is represented as a fixed-length numerical feature vector instead of raw video frames. The model learns patterns that separate real videos from fake ones.

<hr>

<h2>Dataset</h2>

The dataset consists of:

* Video metadata (video names)
* Extracted numerical features per video
* Binary labels indicating whether a video is real or fake

Instead of feeding raw videos into a deep CNN , I worked with pre-extracted features, for faster experimentation.

<hr>

<h2>Project Structure</h2>

```
ml-task/
│
├── preprocess.py        #Data loading & preprocessing
├── train.py             #Model training
├── predict.py           #Inference + CSV generation
│
├── X.npy                #Preprocessed feature matrix
├── y.npy                #Corresponding labels
│
├── models/
│   └── fake_real_classifier.keras   #Trained Keras model
│
├── predictions.csv      #Model predictions on test data
├── README.md
```

<hr>

<h2>Step by Step Implementation</h2>

<h3>1) Loading the Dataset</h3>

* Raw CSV files are read using pandas
* Video identifiers and labels are extracted
* Feature columns are selected and converted into NumPy arrays

---

<h3>2) Data Preprocessing (preprocess.py)</h3>

Preprocessing steps:

* Handling missing values
* Converting features to float32
* Normalizing / scaling feature values
* Separating features (X) and labels (y)

Final outputs:

X.npy  # shape: (num_samples, num_features)
y.npy  # shape: (num_samples,)

Saving preprocessed data avoids repeating expensive operations every time we train.

---

<h3>Model Training (train.py)</h3>

I trained a binary classifier using TensorFlow/Keras.

Model characteristics:

* Input layer matches feature vector size
* Fully connected layers
* Sigmoid output neuron

Technical details:

* Loss: Binary Crossentropy
* Optimizer: Adam
* Metric: Accuracy

The trained model is saved to disk:

<i>models/fake_real_classifier.keras</i>

This file contains:

* Network architecture
* Learned weights
* Optimizer state

---

<h3>Model Inference (predict.py)</h3>

This script:

1. Loads the trained model from disk
2. Loads test features
3. Runs `model.predict()`
4. Converts probabilities into class labels
5. Writes predictions to a CSV file

Output format:

```csv
Video_Name,Prediction,Probability
video_01.mp4,1,0.98
video_02.mp4,0,0.12
```

Where:

* Prediction: 0 = Real, 1 = Fake
* Probability: Confidence score for the predicted class

<hr>

<h2>How the Model Distinguishes Real and Fake</h2>

The model does not understand videos visually. Instead, it learns statistical patterns from numerical features such as:

* Temporal inconsistencies
* Compression artifacts
* Frequency-domain irregularities

During training, the model adjusts its weights to minimize classification error between real and fake samples.
This is supervised learning as the model learns by comparing predictions against known labels.

<hr>

<h2>What Has Been Implemented</h2>

<h3>Implemented</h3>

* Load dataset
* Preprocess data
* Train a machine learning / neural network model
* Run inference
* Generate predictions CSV


<hr>

<h3>Current Status</h3>

* End-to-end pipeline works
* Predictions are deterministic
* Model is trained and saved correctly
* Model accuracy: 
* CSV is generated using the trained model

This is the **baseline**.





