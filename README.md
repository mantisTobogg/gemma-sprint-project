
# Gemma-Sprint-Project
### Google for Developer: Machine Learning Bootcamp 2024

- #### Special thanks to the bootcamp organizers for the support throughout the bootcamp as well as for the opportunity to take part in this journey. - Kihun Choi 

- For further documentation regarding the overview and scope of the project, they are available in the **docs/** folder!

---

## Table of Contents
1. [About the Dataset](#about-the-dataset)
   - [1. Original Dataset](#original-dataset)
   - [2. Refined Dataset](#refined-dataset)
2. [Sentiment Analysis & Flagging](#sentiment-analysis-flagging)
3. [Model Card for Offensive and Sarcasm Detection](#model-card-for-offensive-and-sarcasm-detection)
   - [1. Model Description](#model-description)
   - [2. Applications](#applications)
   - [3. Downstream Use](#downstream-use)
   - [4. Out-of-Scope Use](#out-of-scope-use)
4. [Bias, Risks, and Limitations](#bias-risks-and-limitations)
5. [Training Details](#training-details)
6. [Evaluation](#evaluation)
7. [How to Get Started](#how-to-get-started)

---

## About the Dataset
### Dataset Information
- Datasets also available in **Kaggle's Datasets Library:  
  [US YouTube Comments Dataset on Kaggle](https://www.kaggle.com/datasets/samkihunchoi/us-youtube-comments-dataset)

### 1. Original Dataset: `UScomments.csv`
The **UScomments.csv** dataset was initially a raw collection of user-generated comments from various sources. It contained multiple columns, such as user IDs, timestamps, and additional metadata. However, it was quite noisy, making it challenging to extract meaningful insights for sentiment analysis directly. The primary goal was to convert this dataset into a refined form that could serve as a strong foundation for a binary sentiment classification model.

**Initial Processing:**  
The main reason for refining this dataset was to create a structured and standardized format suitable for training a sentiment analysis model. By filtering out irrelevant metadata and focusing solely on the text and its associated sentiment, we aimed to improve model performance and reduce complexity.

### 2. Refined Dataset: `UScomments_final_cleaned.csv`
The refined version, **UScomments_final_cleaned.csv**, contains only two columns:
- **Comment**: The cleaned text data, representing user comments.
- **Sentiment**: A binary label indicating the sentiment:
  - `1`: **Positive Sentiment**
  - `0`: **Negative Sentiment**

**Seconday Processing:**  
The purpose of creating this refined dataset was to enable a more focused sentiment analysis task, eliminating unnecessary information and ensuring that the model could effectively learn patterns based solely on the sentiment labels.

---

## Sentiment Analysis & Flagging

The project is based on a transformer architecture with multiple stages of comment classification:

1. **Sentiment Analysis**
2. **Offensive Language Detection**
3. **Sarcasm Detection**
4. **Weighted Reclassification**

By integrating these stages, the model provides a comprehensive sentiment overview that factors in multiple nuanced aspects of user comments, making it suitable for advanced NLP applications requiring fine-grained content analysis.

---

## Model Card for Offensive and Sarcasm Detection 📢💬
![image](https://github.com/user-attachments/assets/37cd3a29-df6f-4202-aa48-3f0b145013c0)

### 1. Model Description
This model is designed to enhance the sentiment analysis pipeline by incorporating both offensive language detection and sarcasm detection to provide a refined sentiment classification. It leverages pre-trained language models from Hugging Face, along with custom weighting and reclassification rules for nuanced sentiment analysis.

- **Model type**: Transformer-based model for sentiment analysis and offensive language detection
- **Languages Supported**: English
- **Fine-tuned models used**: 
		`distilbert-base-uncased-finetuned-sst-2-english`, `helinivan/english-sarcasm-detector`, `google/gemma-2-2b-it`

### 2. Applications
1. **Sentiment Analysis**: Determining whether user comments are positive, negative, or neutral.
2. **Offensive Language Detection**: Identifying and categorizing offensive comments.
3. **Sarcasm Detection**: Detecting sarcastic content to ensure accurate interpretation of user sentiment.

### 3. Downstream Use
- **Context-Aware Content Moderation**
- **Toxicity Filtering in Social Media**
- **Refinement of Sentiment Models**

### 4. Out-of-Scope Use
- This model is not intended for high-stakes scenarios or clinical decisions without human supervision.

---

## Bias, Risks, and Limitations
1. **Racial, Gender, and Cultural Bias**  
   Certain offensive terms may not be equally represented across all demographics, leading to uneven detection rates.

2. **Sarcasm Sensitivity**  
   Detecting sarcasm is inherently subjective and may result in false positives/negatives.

**Recommendations:**  
- Use in conjunction with human moderation.
- Reassess and retrain regularly.
- Avoid using for critical tasks.

---

## Training Details

### 1. Training Data
- Pre-trained on a variety of sentiment analysis and sarcasm detection datasets.
- Additional datasets were used to detect and categorize offensive language.

### 2. Training Procedure
- **Sentiment Model**: Fine-tuned using supervised learning on labeled data.
- **Offensive Detection**: Enhanced through regular expressions and weighted reclassification strategies.
- **Sarcasm Detection**: Integrated using a Hugging Face model fine-tuned for sarcasm in English comments.

### 3. Training Hyperparameters
- **Sentiment Analysis Model**:
  - Batch size: 32
  - Learning rate: 2e-5
  - Epochs: 3
- **Sarcasm Detection**:
  - Batch size: 16
  - Learning rate: 1e-5
  - Epochs: 5

---

## Evaluation

### 1. Results
The model demonstrated strong performance in distinguishing between offensive and non-offensive comments. It was able to detect sarcasm in typical English language patterns with reasonable accuracy. The final weighted reclassification significantly improved the handling of complex comments.

### 2. Technical Specifications
The model was trained and evaluated on Apple Silicon (M2) using PyTorch.

- **Hardware**: Apple M2 (32GB RAM)
- **Training Time**: ~6 hours for the sarcasm model and ~4 hours for offensive language integration.

---

## How to Get Started

### Install the Required Libraries:
Refer to `requirements.txt` for installation instructions.

### Example Code:

```python
import pandas as pd
from transformers import pipeline

# Load the Pre-trained Sentiment Analysis Model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)

# Example Comment List
comments = ["Great job!", "This is terrible.", "You did it! 🙃"]

# Perform Sentiment Analysis
results = sentiment_analyzer(comments)
print(results)

# For sarcasm detection, use the following code snippet:
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load Sarcasm Detection Model
sarcasm_model_name = "helinivan/english-sarcasm-detector"
sarcasm_tokenizer = AutoTokenizer.from_pretrained(sarcasm_model_name)
sarcasm_model = AutoModelForSequenceClassification.from_pretrained(sarcasm_model_name).to("mps")

# Perform Sarcasm Detection on a Single Example
comment = "Wow, you really did a great job. 🙄"
inputs = sarcasm_tokenizer(comment, return_tensors="pt").to("mps")
outputs = sarcasm_model(**inputs)
sarcasm_score = outputs.logits.softmax(dim=1)[0][1].item()
is_sarcastic = sarcasm_score >= 0.6
print(f"Is the comment sarcastic? {'Yes' if is_sarcastic else 'No'}")
```

---
