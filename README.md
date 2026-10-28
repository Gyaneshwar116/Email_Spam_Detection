# 📧 Email Spam Detection Using Machine Learning

A machine learning application that classifies text messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP), TF-IDF feature extraction, and supervised machine learning.

The project includes data preprocessing, model training, evaluation, and an interactive Streamlit application for real-time predictions.

---

## 📌 Project Overview

Spam messages are commonly used for unwanted advertising, scams, and phishing attempts. Automatically identifying these messages can help users recognize potentially unwanted or suspicious content.

This project builds a text classification system that learns patterns from labeled messages and predicts whether a new message belongs to the Spam or Ham category.

### Key Highlights

- Text preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- TF-IDF feature extraction
- Comparison of machine learning classification algorithms
- Model evaluation using classification metrics
- Interactive prediction interface using Streamlit
- Model persistence for application deployment

---

## 🎯 Business Problem

Manually identifying spam messages can be time-consuming and unreliable, particularly when users receive large volumes of messages.

The objective is to build a machine learning model that classifies messages automatically and evaluates its ability to distinguish spam from legitimate communication.

### Project Goals

1. Prepare and analyze a labeled text dataset.
2. Transform unstructured text into numerical features.
3. Train supervised machine learning models.
4. Evaluate classification performance on unseen test data.
5. Deploy the selected model through an interactive web application.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Regex | Text cleaning, if used |
| Scikit-learn | Feature extraction, model training, and evaluation |
| TF-IDF | Text feature representation |
| Streamlit | Interactive web application |
| Joblib / Pickle | Saving and loading trained models |

---

## 📂 Project Structure

```text
Email_Spam_Detection/
│
├── app.py
├── email_spam_detection.ipynb
├── Email_spam_model.pkl
├── spam.csv
├── requirements.txt
├── README.md
└── .gitignore
```

**File descriptions**

- `app.py` — Streamlit application for predicting Spam or Ham.
- `email_spam_detection.ipynb` — Notebook containing data analysis, model training, and evaluation.
- `Email_spam_model.pkl` — Saved trained model, if used by the application.
- `spam.csv` — Dataset used for model development.
- `requirements.txt` — Python dependencies.
- `README.md` — Project documentation.
- `.gitignore` — Files excluded from Git tracking.

*Adjust the structure if your application uses a separate TF-IDF vectorizer file or stores the vectorizer and classifier together.*

---

## 🔄 Machine Learning Workflow

```text
Raw Text Dataset
       |
       v
Data Cleaning and EDA
       |
       v
Text Preprocessing
       |
       v
Train-Test Split
       |
       v
TF-IDF Feature Extraction
       |
       v
Model Training
       |
       v
Model Evaluation
       |
       v
Save Trained Model
       |
       v
Streamlit Deployment
       |
       v
Spam / Ham Prediction
```

### 1. Data Collection

The project uses a labeled dataset containing text messages and their corresponding categories.

- **Spam:** Unwanted or suspicious messages.
- **Ham:** Legitimate, non-spam messages.

### 2. Exploratory Data Analysis

EDA helps understand the dataset before training the model.

Typical analysis includes:

- Dataset dimensions and column information
- Missing values and duplicate records
- Distribution of Spam and Ham messages
- Message length analysis
- Frequently occurring words
- Visualizations of class distribution and text patterns

### 3. Text Preprocessing

Text preprocessing prepares messages for machine learning.

Depending on the implementation, this can include:

- Converting text to a consistent case
- Removing HTML tags, if present
- Cleaning unwanted characters
- Handling extra whitespace
- Applying other text normalization techniques

Preprocessing should be consistent between model training and prediction.

### 4. TF-IDF Feature Extraction

Machine learning algorithms require numerical input rather than raw text.

TF-IDF converts text into numerical features based on the importance of words in individual messages relative to the dataset.

It helps represent useful word patterns while reducing the influence of terms that occur frequently across many messages.

The vectorizer should be fitted on training data only and then reused to transform validation, test, and incoming messages.

### 5. Model Training

The notebook imports several classification algorithms:

- Multinomial Naive Bayes
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

These algorithms are compared to determine which performs best for the dataset..

### 6. Model Evaluation

The notebook reports the following test accuracy:

**Accuracy: 98.06%**

The reported classification report is:

| Metric | Class 0 | Class 1 |
|---|---:|---:|
| Precision | 0.98 | 0.98 |
| Recall | 1.00 | 0.88 |
| F1-score | 0.99 | 0.92 |
| Support | 893 | 139 |

Additional reported metrics:

| Metric | Score |
|---|---:|
| Accuracy | 98.06% |
| Macro-average F1-score | 0.96 |
| Weighted-average F1-score | 0.98 |

These results are taken from the displayed notebook output and should be reverified if the model or test split changes.

**Interpretation**

- **Accuracy:** Approximately 98% of the test messages were classified correctly.
- **Precision:** Approximately 98% of predictions for each class were correct.
- **Recall for class 0:** The model identified almost all class 0 messages.
- **Recall for class 1:** The model identified approximately 88% of class 1 messages.
- **F1-score:** Balances precision and recall.

Confirm whether class `0` represents Ham and class `1` represents Spam in the dataset before interpreting these results by category.

Accuracy alone is not sufficient to assess a spam classifier, especially when the dataset contains more Ham messages than Spam messages.

---

## 🧠 How the Application Works

The application follows this prediction process:

```text
User Enters a Message
          |
          v
Text Preprocessing
          |
          v
TF-IDF Transformation
          |
          v
Trained Classification Model
          |
          v
Predicted Category
          |
     +----+----+
     |         |
     v         v
   Spam       Ham
```

The Streamlit application accepts a message, processes it using the same feature-extraction process used during training, and displays the model's predicted category.

---

## 🧪 Example Predictions

The following messages can be used to demonstrate the application.

| Example message | Expected category |
|---|---|
| Congratulations! You have won a free prize. Claim your reward now! | Spam |
| Claim your cash reward today by clicking this link! | Spam |
| Hey, are we meeting at 5 PM today? | Ham |
| Please send me the assignment notes. | Ham |

These are illustrative test cases. The actual output depends on the trained model.

### Prediction Screenshots

Added screenshots of actual Streamlit application of above text.

## 🚀 Installation and Setup

### Prerequisites

- Python installed on your system
- Git installed on your system
- A copy of this repository

### Step 1: Clone the repository

```bash
git clone https://github.com/Gyaneshwar116/Email_Spam_Detection.git
```

### Step 2: Navigate to the project directory

```bash
cd Email_Spam_Detection
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

### Step 4: Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### Step 5: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

---

## 📦 Dependencies

The project uses Python libraries for data analysis, machine learning, visualization, and application deployment.

Example `requirements.txt` entries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
joblib

---
