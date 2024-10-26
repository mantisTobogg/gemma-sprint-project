# GEMMA-Sprint Project: Model Flow and Architecture Overview

---
**Author**: Kihun (Sam) Choi  
**Revision**: 5.0  
**Date**: 2024-10-26  

---
## Overview  
The GEMMA-Sprint Project is an advanced solution integrating **sentiment analysis**, **sarcasm detection**, and **context-aware text generation**. This pipeline classifies and generates accurate responses to user comments with seamless processing.

---

## 1. System Architecture Diagram

<div align="center">
    <img src="/images/GEMMA-Sprint Prj. Model Architecture.png" alt="Model Architecture" width="1000">
</div>

---

## 2. Process Flow Description  

1. **Data Input**: Load comments from `datasets/UScomments_final_cleaned.csv`.
2. **Sentiment Analysis**:
   - Use `distilbert-base-uncased-finetuned-sst-2-english` to classify comments as **positive** or **negative**.
   - Generate logs for each processed batch.
3. **Sarcasm Detection**:
   - Detect sarcasm using `helinivan/english-sarcasm-detector`.
   - Modify response prompts for sarcastic content.
4. **Text Generation with GEMMA-2B-IT**:
   - Generate contextual responses aligned with sentiment and sarcasm.
   - Positive → Encouraging | Negative → Constructive | Sarcasm → Witty.
5. **Checkpointing and Backup**: Save progress to `/checkpoints/` to prevent data loss.
6. **Logging**: Monitor logs to detect bottlenecks or errors.
7. **CSV Output**: Save results in `outputs/50_percent_processed_comments.csv`.

---

## 3. Key Enhancements  

### 3.1 Contextual Response Generation  
> GEMMA generates **tailored responses**:
- **Positive**: Encouraging  
- **Negative**: Constructive feedback  
- **Sarcasm**: Witty and humorous
  
**Example Prompt:**
```bash
Provide a witty response to: "<user comment>"
```

### 3.2 Token Limit Increase  
- **Max tokens** set to 40 for coherent responses.  
- **Before**: Short, incomplete outputs.  
- **After**: Detailed responses.  

💡 **Example Output for Positive Sentiment**:
```
Comment: "Your videos always brighten my day!"
GEMMA Response: "I’m glad to hear that! Stay tuned for more exciting content. Your support means a lot!"
```

### 3.3 Parallel Processing Optimization  
- **ProcessPoolExecutor** ensures faster response generation.  
- **8 workers** handle concurrent batches, reducing computation time.

### 3.4 Graceful Error Handling with Fallbacks  
- **Default Responses**: Provides fallback responses when GEMMA encounters an error or exception.  

- 💡 **Example Placeholder Response**:
Comment: "Why is this trending?"
GEMMA Response: "We couldn’t generate a response. Please try again later!"


### 3.5 Optimized for Apple Silicon (M2) Hardware  
- **MPS Backend**: Uses Metal Performance Shaders for accelerated processing.  
- **Performance Boost**: Achieved reduced inference time to 1-2 seconds per comment.

---
## 4. Dataset Overview  

### Original Dataset: `UComments.csv`  
- Contains raw comments with metadata from multiple sources.  
- Preprocessing was required to remove noise.

### Refined Dataset: `UComments_final_cleaned.csv`  
- **cleaned_comment**: Preprocessed text data.  
- **sentiment**: Binary label (`1` = Positive, `0` = Negative).  

💡 **TIP**:  
Datasets are available on Kaggle: [US YouTube Comments Dataset](https://www.kaggle.com/datasets/samkihunchoi/us-youtube-comments-dataset)

---
## 5. Batch Processing  
- **Batch size**: 128  
- **Workers**: 8  
- **Checkpointing**: Save each batch to prevent loss.  
- **Output File**: `outputs/50_percent_processed_comments.csv`

---
## 6. Training and Checkpoints  

### Checkpoint System  
- Save checkpoints to `/checkpoints/`.  
- Backup files include:
  - `sarcasm_labels.pkl`  
  - `sarcasm_labels_backup_corrected.pkl`

---

## 7. Model Evaluation  
### Improvements  
- **50% dataset processed** to balance performance and accuracy.  
- Accurate sarcasm detection with refined prompts.  
- GEMMA-2B-IT integration enhances response quality.

### Future Plans  
- Fine-tune sarcasm detection for fewer false positives.  
- Deploy model via **REST API** for public access.

---
## 8. Limitations and Future Work  

### Bias  
- Cultural or linguistic bias may affect sarcasm detection accuracy.

### Processing Time  
- Large datasets remain resource-intensive despite optimizations.

### Future Improvements  
- **GEMMA-2B-IT Enhancements**: Further refine prompts for better context.  
- **Interactive UI**: Develop a real-time web interface for direct user interaction.

---

## 9. Conclusion  
The GEMMA-Sprint Project delivers an **integrated sentiment analysis and response generation pipeline** with **scalable** and **high-performance processing**.  

With enhanced checkpointing and GEMMA-2B-IT’s advanced model, it supports:  
- **Social Media Moderation**  
- **Customer Interaction**  
- **Content Filtering Based on Sentiment**
