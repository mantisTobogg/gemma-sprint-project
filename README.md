# 🎓 **Google for Developers: Machine Learning Bootcamp Korea 2024**

<div align="center">
    <img src="./images/image.png" alt="Google for Developers: Machine Learning Bootcamp" width="1000">
</div>

---
### Acknowledgments:
> 🏅 **Special thanks to the bootcamp organizers** for their support and mentorship throughout this project journey!  
> **- Kihun (Sam) Choi**

> 📑 For detailed **project documentation**, refer to the `docs/` folder!
> 
> 📜 Historical Reference  
> For earlier versions of the project and documentation, refer to the [OLD_README.md](archive/OLD_README.md).

---

## 📑 **Table of Contents**

1. [Overview](#overview)  
2. [Enterprise Use Cases](#enterprise-use-cases)  
3. [About the Dataset](#about-the-dataset)  
4. [Model Architecture](#model-architecture)  
5. [Training and Checkpoints](#training-and-checkpoints)  
6. [Installation and Quickstart](#installation-and-quickstart)  
7. [Sample Output](#sample-output)  
8. [⚙️ Mac ARM64 Optimization Efforts](#mac-arm64-optimization-efforts)  
9. [Limitations and Future Work](#limitations-and-future-work)

---

# 📌 **1. Overview**
## A **Secure**, multi-LLM Enterprise Workflow/Productivity Empowering A.I. Solution
### **Key Capabilities:**
> [!IMPORTANT]
> This project aims to build a sophisticated multi-LLM solution tailored for **enterprise consulting** use-cases, providing sentiment-aware, contextually enriched responses. The solution supports consultants in capturing user sentiment, detecting sarcasm, and generating impactful responses, enhancing client engagement and customer satisfaction.
>- **Customer Sentiment Analysis**: Classifies feedback as positive, negative, or sarcastic to prioritize responses.
>- **B2B Support**: Automates responses, saving time while maintaining professionalism.
>- **HR Feedback Analysis**: Detects trends in employee feedback, identifying areas for improvement.

---

## 🏢 **2. Enterprise Use Cases**

1. **Customer Support for SaaS and B2B Companies**  
   In fast-paced environments, **Gemma-Sprint-Project** flags high-risk tickets (negative or sarcastic tone) for prioritized handling, generating empathetic responses that increase customer satisfaction.

2. **Real-Time Sentiment Analysis for Product Consultants**  
   This tool enables consultants to gain insights into customer sentiment across social media and reviews, identifying positive, negative, or sarcastic tones to support product improvement efforts.

---

## 📦 **3. About the Dataset**

### **Primary Dataset**: `UScomments_final_cleaned.csv`

- **Core Columns**:
  - **`cleaned_comment`**: User comments, cleaned and pre-processed.
  - **`sentiment`**: Binary sentiment labels (1: Positive, 0: Negative).

### **Supplementary IT Glossary Dataset**: `IT_glossory_final_v4.csv`

- **Purpose**: Provides a glossary of IT terms to enhance technical accuracy and relevance in responses, particularly for enterprise and IT consultancy.

---

## ⚙️ **4. Model Architecture**

### **Model Stack**:

1. **GPT-J**: Generates empathetic, context-rich responses.
2. **MiniLM**: Recognizes domain-specific IT terms efficiently.
3. **DistilBERT**: Detects sentiment for classification.
4. **Sarcasm Detection Model**: Identifies sarcasm for nuanced communication.
5. **GEMMA-2B-IT**: Delivers sentiment-aware, industry-specific responses.

---

## 📋 **5. Training and Checkpoints**

- **Checkpoints**: Stored in the `checkpoints/` directory, minimizing data loss during lengthy computations.
- **Logging**: Real-time logging in `logs/` allows progress tracking and troubleshooting.

Optimized for **Apple Silicon (M2)** using the **MPS backend**, balancing high-performance with efficient resource usage.

---

## 🚀 **6. Installation and Quickstart**

### **Pre-requisites**:

1. **Docker** and **Colima** for containerized setup.
2. **Python 3.10** with dependencies in `requirements.txt`.

### **Installation Steps**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dockerdoge12/gemma-sprint-prj.git
   ```
   ```bash
   cd gemma-sprint-prj
   ```

2. **Pull Docker Image**:
   ```bash
   docker pull dockerdoge12/gemma-sprint-prj:latest
   ```

3. **Run Docker Container**:
   ```bash
   docker run -p 8080:8080 dockerdoge12/gemma-sprint-prj:latest
   ```

---

## 📊 **7. Sample Output**

 ```zsh
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.38it/s]

Sample Query 1:
User Comment: "We’re using legacy Oracle databases. How challenging would it be to migrate to a modern data warehouse solution?"

Response: Thank you for bringing this to our attention. Let's work through the compatibility concerns step by step.
**Understanding the Challenges**
- **Different Database Technologies:** Oracle Database is a relational database management system (RDBMS) based on the SQL standard. Fundamental difference in architecture can lead to compatibility issues.
- **Data Mapping and Transformation:**
    * Identify the data elements that need to be transferred between the two systems (e.g., customer master data, sales orders, inventory levels).
    * Determine the mapping between data types, structures, and formats in both databases. Specialized tools or custom scripts may be needed for data conversion.
- Would you like me to elaborate further?
 ```

---

## ⚙️ **8. Mac ARM64 Optimization Efforts**

Optimizations were implemented to ensure model performance on **Apple Silicon**:

1. **PyTorch over TensorFlow**: Enabled compatibility with MPS (Metal Performance Shaders) for faster computation.
2. **Batch Size Adjustment**: Fine-tuned batch sizes to balance memory and computation on the M2 chip.
3. **Reduced Computation Complexity**: Limited training epochs and iteratively adjusted parameters to minimize resource usage.
4. **Efficient Logging and Checkpointing**: Detailed logs and checkpoints track performance without requiring re-runs.

---

## 🎯 **9. Limitations and Future Work**

### **Limitations**

- **Bias and Context Sensitivity**: Sarcasm detection accuracy remains affected by contextual subtleties; further retraining with specific data is needed to enhance precision.
- **Computational Constraints**: Optimized for local M2 hardware, limiting training complexity and token length for efficient operation.
- **Incomplete Responses**: Implemented continuation prompts to address response truncation due to token limitations.

### **Future Work**

1. **Model Enhancements**: Refine GEMMA-2B-IT prompts for better contextual accuracy.
2. **Interactive UI Development**: Build a real-time interface for improved user engagement and feedback integration.

---

##  Acknowledgments-GEMMA-Sprint Project:

1.	**DistilBERT** for Sentiment Analysis
    Model: distilbert-base-uncased-finetuned-sst-2-english
    > Source: "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english"
	- Developed by Hugging Face, this model is fine-tuned for sentiment analysis and provides a compact version of BERT for efficient classification tasks.

2.	**Sarcasm Detection Model**
    Model: helinivan/english-sarcasm-detector
    > Source: Hugging Face
  	> Link: "https://huggingface.co/helinivan/english-sarcasm-detector"
	- Created by Helini Ivan on Hugging Face, this model is designed to detect sarcasm in English text, helping to capture nuances in sentiment.

3.	**GPT-J for Contextual Response Generation**
    Model: EleutherAI/gpt-j-6B
    > Source: Hugging Face
    > Link: "https://huggingface.co/EleutherAI/gpt-j-6B"
	- Developed by EleutherAI, GPT-J is a powerful language model suitable for generating context-aware, coherent responses.

5.	**MiniLM for IT Domain-Specific Recognition**
    Model: microsoft/MiniLM-L12-H384-uncased
    > Source: Hugging Face
    > Link: "https://huggingface.co/microsoft/MiniLM-L12-H384-uncased"
	- Created by Microsoft, this model provides efficient natural language understanding, especially for identifying and handling IT-specific terms in
context.

6.	**GEMMA-2B-IT**
    > Source: Hugging Face   
--- 

