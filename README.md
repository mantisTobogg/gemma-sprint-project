# 🎓 **Google for Developers: Machine Learning Bootcamp Korea 2024**

<div align="center">
    <img src="./images/image.png" alt="Google for Developers: Machine Learning Bootcamp" width="1000">
</div>

---

### Acknowledgments:

> 🏅 **Special thanks to the bootcamp organizers** for their support and mentorship throughout this project journey!  
> **- Kihun (Sam) Choi**

> 📑 For detailed **project documentation**, refer to the `docs/` folder!

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

# 📌 **Overview**

### **Project Overview:**

The objective was to develop a sophisticated multi-stack model solution tailored to **enterprise consulting** use-cases. By integrating a five-model stack, this project generates sentiment-aware, contextually enriched responses to streamline interactions in tech, B2B, and customer service environments. This project’s unique design aids consultants in capturing user sentiment, detecting sarcasm, and automatically generating meaningful responses, significantly enhancing client engagement and customer satisfaction.

### **Key Capabilities:**

- **Customer Sentiment Analysis**: Classify feedback as positive, negative, or sarcastic to prioritize responses.
- **B2B Support**: Automate responses to save time while maintaining professionalism.
- **HR Feedback Analysis**: Detect employee feedback trends, identifying areas for workplace improvement.

---

## 🏢 **Enterprise Use Cases**

1. **Customer Support for SaaS and B2B Companies**:  
   In fast-paced environments, efficiently handling large volumes of feedback is critical. **Gemma-Sprint-Project** automatically flags high-risk tickets (negative or sarcastic tone) for prioritized attention and generates empathetic responses, increasing customer satisfaction.

2. **Real-Time Sentiment Analysis for Product Consultants**:  
   **Gemma-Sprint-Project** enables consultants to gain deeper insights into customer sentiment across social media and reviews. Positive, negative, or sarcastic tones are identified, aiding in product improvement by quickly pinpointing areas of concern.

---

## 📦 **About the Dataset**

### **1. Primary Dataset**: `UScomments_final_cleaned.csv`

- **Core Columns**:
  - **`cleaned_comment`**: User comments, cleaned and pre-processed.
  - **`sentiment`**: Binary sentiment labels (1: Positive, 0: Negative).

### **2. Supplementary IT Glossary Dataset**: `IT_glossory_final_v4.csv`

- **Purpose**: Provides a glossary of IT terms to enhance technical accuracy and relevance in responses, particularly for enterprise and IT consultancy.

---

## ⚙️ **Model Architecture**

### **Model Stack**:

1. **GPT-J**: Generates empathetic, context-rich responses.
2. **MiniLM**: Recognizes domain-specific IT terms efficiently.
3. **DistilBERT**: Detects sentiment for classification.
4. **Sarcasm Detection Model**: Identifies sarcasm for nuanced communication.
5. **GEMMA-2B-IT**: Delivers sentiment-aware, industry-specific responses.

---

## 📋 **Training and Checkpoints**

1. **Checkpoints**: Saved in the `checkpoints/` directory, minimizing risk of data loss during lengthy computations.
2. **Logging**: Real-time logging in `logs/` allows progress tracking and aids in troubleshooting.

The pipeline is optimized for **Apple Silicon (M2)** hardware using the **MPS backend**, allowing high-performance without sacrificing response quality.

---

## 🚀 **Installation and Quickstart**

### **Pre-requisites**:

1. **Docker** and **Colima** for efficient setup.
2. **Python 3.10** with dependencies listed in `requirements.txt`.

### **Installation Steps**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dockerdoge12/gemma-sprint-prj.git
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
## 📊 Sample Output

 ```zsh
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.38it/s]

Sample Query 1:
User Comment: "We’re using legacy Oracle databases. How challenging would it be to migrate to a modern data warehouse solution?"

Response: Thank you for bringing this to our attention. Let's work through the compatibility concerns step by step.
**Understanding the Challenges**
***Different Database Technologies:** Oracle Database is a relational database management system (RDBMS) based on the SQL standard. Fundamental difference in architecture can lead to compatibility issues.
1. **Data Mapping and Transformation:**
    * Identify the data elements that need to be transferred between the two systems (e.g., customer master data, sales orders, inventory levels).
   * Determine the mapping between data types, structures, and formats in both databases. You might need specialized tools or custom scripts for data conversion and transformation.
     * Ensure you have the appropriate database drivers installed for both Oracle and Would you like me to elaborate further?
   ```
---

## ⚙️ Mac ARM64 Optimization Efforts

Given the limited compatibility of some deep learning libraries with ARM64 architecture, a series of critical adaptations were implemented to ensure the model performed efficiently on Apple Silicon:

1. **PyTorch over TensorFlow**:  
   Replacing TensorFlow with PyTorch enabled compatibility with MPS (Metal Performance Shaders) for faster computation on Apple Silicon.

2. **Batch Size Optimization**:  
   Adjusted batch sizes and restricted dataset iterations to balance memory use and prevent overloads during training on an M2 chip.

3. **Reduced Computation Complexity**:  
   Limited training to one epoch initially, while running inference cycles to monitor and iteratively adjust parameters without straining system resources.

4. **Efficient Logging and Checkpointing**:  
   Enabled detailed logging and periodic checkpointing to track performance without re-running entire training processes.

---

## 🎯 Limitations and Future Work

### **Limitations**

1. **Bias and Context Sensitivity**:  
   Sarcasm detection remains challenging due to subtle contextual nuances. Model retraining with more context-specific data is needed to reduce false positives in professional settings.

2. **Computational Constraints**:  
   The model was optimized for local M2 hardware, necessitating simplifications like single-epoch training and prompt-specific responses with GEMMA-2B-IT to generate robust outputs.

3. **Incomplete Responses**:  
   To address potential truncation of responses due to token limitations, we added continuation prompts to enhance user experience without overloading computation.

---

