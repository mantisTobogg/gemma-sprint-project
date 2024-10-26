# 🎓 **Google for Developers: Machine Learning Bootcamp Korea 2024**

# 🛠️ **Gemma-Sprint-Project**

### Acknowledgments:  
> 🏅 **Special thanks to the bootcamp organizers** for their support throughout the journey and for this amazing opportunity!  
> **- Kihun (Sam) Choi**  
>
> 📑 For detailed **project documentation**, refer to the `docs/` folder!
> 
> 📜 Historical Reference  
>For earlier versions of the project and documentation, refer to the [OLD_README.md](archive/OLD_README.md).
  

---

## 📑 **Table of Contents**

1. [Overview](#overview)  
2. [About the Dataset](#about-the-dataset)  
3. [Updated Architecture](#updated-architecture)  
4. [Training and Checkpoints](#training-and-checkpoints)  
5. [Installation and Quickstart](#installation-and-quickstart)  
6. [How to Run](#how-to-run)  
7. [Model Evaluation](#model-evaluation)  
8. [Limitations and Future Work](#limitations-and-future-work)  

---

# 📌 **Overview**  

### **Project Overview:**  

This project aims to **detect and classify user comments** across various sentiment categories, including:  
1. **Sentiment Analysis Pipeline**: Classifies comments as Positive or Negative.  
2. **Offensive Language Detection**: Identifies and flags offensive comments based on severity.  
3. **Sarcasm Detection Integration**: Enhances sentiment accuracy by identifying sarcasm.  
4. **Response Generation**: Leverages the `gemma-2b-it` model to create relevant text-based responses.  

> The goal is to develop a powerful **sentiment and sarcasm detection pipeline** integrated with a **GEMMA-2B-IT** text generation model. This project utilizes **fine-tuned NLP models** to classify comments and generate meaningful responses with **context awareness**.  

---

## 📦 **About the Dataset**

> [!TIP]  
> Datasets also available in **Kaggle's Datasets Library**: [US YouTube Comments Dataset on Kaggle](https://www.kaggle.com/datasets/samkihunchoi/us-youtube-comments-dataset)  

> [!IMPORTANT]  
> ***Be advised, the dataset contains some extremely vulgar language.***  

### 1. **Original Dataset**: `UScomments.csv`  
- Raw user comments with metadata from multiple sources.  
- This version was **noisy** and contained unnecessary metadata.

### 2. **Refined Dataset**: `UScomments_final_cleaned.csv`  
- Contains **cleaned text data** with two columns:  
  1. **`cleaned_comment`**: Text data representing user comments.  
  2. **`sentiment`**: Binary labels:  
     - `1`: Positive  
     - `0`: Negative  

---

## 🚀 **Recent Improvements:** **2024.10.26**

> - **50% of the dataset** processed to reduce runtime while retaining sufficient insights.  
> - **Batch size** set to **128** and **8 workers** for optimal performance on **M2 hardware**.  
> - **Checkpoints and backups** implemented to prevent data loss during long computations.  
> - Enhanced **logging** for better tracking and bottleneck identification.  

---

# 🎯 **Summary of Changes**  

With these improvements, the **GEMMA-2B-IT integration** is now faster, more responsive, and sentiment-aware. The project provides **actionable insights** for **social media moderation**, **customer interaction**, and **sentiment-based content filtering**.  

---

## 🛠️ **Enhancements**  

### 1️⃣ **Enhanced Contextual Response Generation**  
> **Dynamic Prompts** ensure GEMMA’s responses align with the sentiment:
>   - **Positive Sentiments** → Encouraging, uplifting messages.  
>   - **Negative Sentiments** → Constructive feedback or empathy.  
>   - **Sarcastic Comments** → Witty and humorous replies.  

💡 **Example Prompt**:  
```
Provide a witty and sarcastic response to: "<user comment>"
```

---

### 2️⃣ **Increased Token Limit for More Meaningful Output**  
> - **Max tokens set to 40**: More detailed and coherent responses.  
> - **Before**: Short, incomplete replies.  
> - **After**: Well-rounded messages that feel complete.  

💡 **Example Output for Positive Sentiment**:  
```
Comment: "Your videos always brighten my day!"
GEMMA Response: "I'm glad to hear that! Stay tuned for more exciting content. Your support means a lot!"
```

---

### 3️⃣ **Parallel Processing for Speed Optimization**  
> - **ProcessPoolExecutor**: Parallelized response generation to handle comments faster.  
> - **Result**:  
   - Faster execution: Handles large datasets efficiently.  
   - Trackable progress via real-time logs and tqdm.  

---

### 4️⃣ **Graceful Error Handling with Fallbacks**  
> - **Default Responses**: If GEMMA encounters an error, it provides fallback responses.  

💡 **Example Placeholder Response**:  
```
Comment: "Why is this trending?"
GEMMA Response: "We couldn’t generate a response. Please try again later!"
```

---

### 5️⃣ **Optimized for Apple Silicon (M2) Hardware**  
> - **MPS Backend**: Faster inference on Apple hardware.  
> - **Impact**: Reduced response time to 1-2 seconds.  

---

## ⚙️ **Updated Architecture**

1. **Sentiment Analysis**:  
   Uses `distilbert-base-uncased-finetuned-sst-2-english` to classify comments into positive or negative sentiment.  

2. **Sarcasm Detection**:  
   Integrated `helinivan/english-sarcasm-detector` to enhance classification accuracy by identifying subtle sarcastic cues.  

3. **Response Generation with GEMMA**:  
   Uses `GEMMA-2B-IT` to generate context-aware responses tailored to the detected sentiment.  

4. **Batch Processing**:  
   - **Batch size**: 128  
   - **Workers**: 8  
   Optimized for **Apple Silicon (M2)** hardware.

---

## 📋 **Training and Checkpoints**

- **Checkpoints are saved** after processing each batch to prevent loss in case of a crash.  
- Backup files stored in the `checkpoints/` directory:
  - Example: `sarcasm_labels.pkl`, `sarcasm_labels_backup.pkl`  
- **Real-time logging** with `main.log` for monitoring progress.

---

## 📊 **Evaluation & Results**

- **50% of the dataset** processed (~340,864 comments).  
- **GEMMA-2B-IT** generated sentiment-aligned responses.  
- **Improved runtime performance** through optimized batch processing.

---

## 🔮 **Limitations and Future Work**

1. **Bias and False Positives**:  
   Sarcasm detection is subjective and prone to inaccuracies.

2. **Long Processing Times**:  
   Large datasets still require significant time to process, despite optimizations.

---

## 🚀 **Installation and Quickstart**

1. **Install Dependencies**:  
```bash
pip install transformers torch pandas tqdm
```

2. **Run the Pipeline**:  
```bash
python src/run_pipeline.py
```

3. **Monitor Logs**:  
```bash
tail -f logs/main.log
```

---

## 🎯 **Future Plans**

1. **Fine-tune models** for better sentiment classification.  
2. **Deploy the model** via a REST API for public use.  
3. **Create an interactive UI** for real-time response generation.

---

## 📚 **Documentation & Logs**

- Refer to the `docs/` folder for project-related documentation.
- Monitor the **logs/** folder for tracking execution progress.

---
