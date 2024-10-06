# gemma-sprint-project


#### Being my first repo, it is in need of some organization improvements. I'll continue to "fine-tune :)" the visuals and overall repository organization. 

  - Special thanks to the bootcamp organizers for the support all throughout the bootcamp as well as for the opportunity to take part in this journey. -Kihun Choi -


  - For documentation regarding the overview and scope of the project, I urge you to take a look in the docs/ folder!
  - > Datasets are also available in **Kaggle's Datasets Libray**:
    > https://www.kaggle.com/datasets/samkihunchoi/us-youtube-comments-dataset
---

# About the Dataset: 

## 1. Original Dataset: `UScomments.csv`

The **UScomments.csv** dataset was initially a raw collection of user-generated comments from various sources. It contained multiple columns, such as user IDs, timestamps, and additional metadata. However, it was quite noisy, making it challenging to extract meaningful insights for sentiment analysis directly. The primary goal was to convert this dataset into a refined form that could serve as a strong foundation for a binary sentiment classification model.

### Purpose:
The main reason for refining this dataset was to create a structured and standardized format suitable for training a sentiment analysis model. By filtering out irrelevant metadata and focusing solely on the text and its associated sentiment, we aimed to improve model performance and reduce complexity.


## 2. Refined Dataset: `UScomments_final_cleaned.csv`

The refined version, **UScomments_final_cleaned.csv**, contains only two columns:

1. **Comment**: The cleaned text data, representing user comments.
2. **Sentiment**: A binary label indicating the sentiment:
   - `1`: **Positive Sentiment**
   - `0`: **Negative Sentiment**

### Purpose: Gemma-Sprint Project
The purpose of creating this refined dataset was to enable a more focused sentiment analysis task, eliminating unnecessary information and ensuring that the model could effectively learn patterns based solely on the sentiment labels. This step was crucial in allowing us to integrate more complex models, such as the `gemma-2b-it` text generation model, for downstream analysis and contextual response generation.


## A Quick Note on Dataset Attribution
I apologize for not being able to provide the original creators or references for the dataset, as it was provided without detailed attribution. If anyone recognizes this dataset and has more information on its origin, I would be happy to update the documentation and give credit where it's due.

 ---
 ---
 
# Sentiment Analysis &amp; Subsequent Flagging of Inappropriate Language


### The project is based on a transformer architecture with multiple stages of comment classification:

	1.	Sentiment Analysis.
	2.	Offensive Language Detection.
	3.	Sarcasm Detection.
	4.	Weighted Reclassification.

    By integrating these stages, the model provides a comprehensive sentiment overview that factors in multiple nuanced aspects of user comments, making it suitable for advanced NLP applications requiring fine-grained content analysis.
     

![Gemma  Final_Architecture_Overview](https://github.com/user-attachments/assets/ed02ac3c-35f4-4a6b-a6e1-57bef0530617)



# A: Model Card for Offensive and Sarcasm Detection 📢💬

### 1. Model Description
This model is designed to enhance the sentiment analysis pipeline by incorporating both offensive language detection and sarcasm detection to provide a refined sentiment classification. It leverages pre-trained language models from Hugging Face, along with custom weighting and reclassification rules for nuanced sentiment analysis. The model can be applied to various NLP applications requiring content moderation or toxicity analysis.

	•	Model type: 
	        Transformer-based model for sentiment analysis and offensive language detection
	•	Languages Supported: 
	        English
	•	Fine-tuned models used: 
	        distilbert-base-uncased-finetuned-sst-2-english, helinivan/english-sarcasm-detector, google/gemma-2-2b-it

### 2. This model can be directly used for:

	1.	Sentiment Analysis: Determining whether user comments are positive, negative, or neutral.
	2.	Offensive Language Detection: Identifying and categorizing offensive comments based on varying levels of severity.
	3.	Sarcasm Detection: Detecting sarcastic content to ensure accurate interpretation of user sentiment.

### 3. Downstream Use

	•	Context-Aware Content Moderation: Integrate into comment moderation tools to flag and categorize harmful or misleading content.
	•	Toxicity Filtering in Social Media: Enhance existing systems that filter out harmful language by adding sarcasm detection.
	•	Refinement of Sentiment Models: Fine-tuning for more complex datasets or extended languages.

### 4. Out-of-Scope Use

	•	This model is not intended for critical decisions in safety-critical environments (e.g., predicting crime or assessing threats).
	•	Should not be used to make clinical decisions or for any high-stakes scenarios without human supervision.

# B: Bias, Risks, and Limitations

#### 1. The model’s performance is highly influenced by the training data, which may have inherent biases. For example:

	•	Racial, Gender, and Cultural Bias: Certain offensive terms may not be equally represented across all demographics, leading to uneven detection rates.
	•	Sarcasm Sensitivity: Detecting sarcasm is inherently subjective and may result in both false positives and false negatives.
	•	Contextual Understanding: The model lacks deep contextual understanding and may misinterpret certain phrases, particularly in the presence of irony or humor.

### 2. Recommendations

	1.	Use in Conjunction with Human Moderation: Always use the model’s output as a supplementary tool, with final decisions made by human moderators.
	2.	Reassess and Retrain Regularly: Given the evolving nature of offensive language, it’s recommended to fine-tune the model periodically with updated datasets.
	3.	Avoid Using for Critical Tasks: The model should not be solely relied upon for sensitive content filtering where real-world consequences are at stake.


# C: Training Details
### 1. Training Data

	•	The model was pre-trained on a variety of sentiment analysis and sarcasm detection datasets, including the SST-2 dataset for sentiment and a custom sarcasm dataset.
	•	Additional datasets were used to detect and categorize offensive language, ensuring diverse coverage across multiple categories of severity.

### 2. Training Procedure

	•	Sentiment Model: Fine-tuned using supervised learning on labeled data.
	•	Offensive Detection: Enhanced through regular expressions and weighted reclassification strategies.
	•	Sarcasm Detection: Integrated using a Hugging Face model fine-tuned specifically for sarcasm in English comments.

### 3. Training Hyperparameters

	•	Sentiment Analysis Model:
	•	Batch size: 32
	•	Learning rate: 2e-5
	•	Epochs: 3
	•	Mixed Precision: fp16
	•	Sarcasm Detection:
	•	Batch size: 16
	•	Learning rate: 1e-5
	•	Epochs: 5


# D: Evaluation
### 1. Results

    The model demonstrated strong performance in distinguishing between offensive and non-offensive comments. It was able to detect sarcasm in typical English language patterns with a reasonable accuracy. The final weighted reclassification significantly improved the handling of complex comments that mix positivity and negativity.

### 2. Technical Specifications
The model was trained and evaluated on Apple Silicon (M2) using PyTorch. Resource consumption and training time were optimized for local development.

	•	Hardware: Apple M2 (32GB RAM)
	•	Training Time: ~6 hours for the sarcasm model and ~4 hours for offensive language integration.



# - temp -
# E: How to Get Started with the Model
### Install the Required Libraries:
    Refer to requirements.txt 

### Use the Code Below to Get Started:
    import pandas as pd
    from transformers import pipeline

### Load the Pre-trained Sentiment Analysis Model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)

### Example Comment List
    comments = ["Great job!", "This is terrible.", "You did it! 🙃"]

### Perform Sentiment Analysis
    results = sentiment_analyzer(comments)
    print(results)

#### For sarcasm detection, use the following code snippet:

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

#### Load Sarcasm Detection Model
    sarcasm_model_name = "helinivan/english-sarcasm-detector"
    sarcasm_tokenizer = AutoTokenizer.from_pretrained(sarcasm_model_name)
    sarcasm_model = AutoModelForSequenceClassification.from_pretrained(sarcasm_model_name).to("mps")

#### Perform Sarcasm Detection on a Single Example
    comment = "Wow, you really did a great job. 🙄"
    inputs = sarcasm_tokenizer(comment, return_tensors="pt").to("mps")
    outputs = sarcasm_model(**inputs)
    sarcasm_score = outputs.logits.softmax(dim=1)[0][1].item()
    is_sarcastic = sarcasm_score >= 0.6
    print(f"Is the comment sarcastic? {'Yes' if is_sarcastic else 'No'}")

