# 📚 Empirical Study on Paraphrases

**Authors:**  
- Bidisha Mukherjea (bmukh051@uottawa.ca)  
- Vrishab Davey (vdave048@uottawa.ca)  
**Affiliation:** Dept of Electrical Engineering & Computer Science, University of Ottawa  

---

## 📌 Overview

This project investigates paraphrase detection performance across both real-world and synthetic datasets using various algorithms. It is motivated by the increasing need for accurate intent recognition in virtual assistants.

---

## 📁 Datasets

### 🔹 D1: Microsoft Research Paraphrase Corpus (MRPC)
- Real-world dataset of 5,801 sentence pairs
- Labeled as paraphrase (1) or non-paraphrase (0)
- Used to benchmark sentence similarity models

### 🔹 D2: Synthetic Virtual Assistant Dataset
- Created using GPT-4 to generate task-based paraphrases
- Five intents: Set a Timer, Play Music, Check the Weather, Send a Message, Turn on the Lights
- Each intent contains 30 paraphrased sentence pairs

---

## 🔍 Models Evaluated

### 1. **Baseline Model**
- Exact string match classifier
- Labels sentences as paraphrases only if they are identical

### 2. **Transformer-Based Model**
- Uses `sentence-transformers/paraphrase-mpnet-base-v2`
- Embeds sentence pairs and computes cosine similarity for paraphrase classification

### 3. **Cross-Encoder Model**
- `cross-encoder/stsb-roberta-base`
- Trained on:
  - D1
  - D2
  - Combined D1 + D2

---

## ⚙️ Experimental Setup

### Training Configurations
- Within-domain: Train and test on same dataset
- Cross-domain: Train on one, test on the other
- Combined-domain: Train on both datasets, test on each

### Metrics Used
- Accuracy
- Precision
- Recall
- F1-Score

---

## 🧪 Key Results

| Model             | Training | Testing | Accuracy | Precision | Recall | F1-Score |
|------------------|----------|---------|----------|-----------|--------|----------|
| Baseline          | D2       | D2      | 0.52     | -         | -      | -        |
| Baseline          | D1       | D1      | 0.34     | -         | -      | -        |
| Transformer       | D1       | D1      | 0.74     | 0.74      | 0.95   | 0.83     |
| Transformer       | D2       | D2      | 0.49     | 0.48      | 0.81   | 0.60     |
| Cross-Encoder     | D1+D2    | D1      | 0.87     | 0.90      | 0.91   | 0.90     |
| Cross-Encoder     | D1       | D1      | 0.87     | 0.90      | 0.91   | 0.90     |
| Cross-Encoder     | D2       | D1      | 0.77     | 0.76      | 0.95   | 0.85     |
| Cross-Encoder     | D2       | D2      | 0.52     | 0.49      | 0.42   | 0.45     |
| Cross-Encoder     | D1+D2    | D2      | 0.51     | 0.48      | 0.41   | 0.44     |
| Cross-Encoder     | D1       | D2      | 0.52     | 0.50      | 0.37   | 0.43     |

---

## 📈 Analysis

- Cross-Encoder trained on D1 performs best on MRPC
- Transformer model generalizes better across domains than baseline
- D2-trained models generalize to D1 better than the reverse
- Combining datasets doesn't guarantee better performance
- VA-based data (D2) is essential for task-based NLP applications

---

## ⚠️ Challenges Observed

1. **Intent Confusion**  
   Sentences with same intent but different targets classified as paraphrases

2. **Semantic vs. Functional Similarity**  
   Similar words with different task outcomes

3. **Contextual Dependency**  
   Requires understanding world knowledge and contextual links

4. **Question vs. Statement Pairs**  
   Difficulty aligning different syntactic forms

---

## 🧩 Future Work

- Domain-specific fine-tuning for virtual assistant data  
- Context-aware modeling for better functional paraphrase understanding  
- More robust datasets to enhance generalization

---

## 📚 References

1. https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall  
2. https://en.wikipedia.org/wiki/F-score  
3. https://encord.com/glossary/f1-score-definition/  
4. https://www.kaggle.com/datasets/doctri/microsoft-research-paraphrase-corpus/data  
5. https://chatgpt.com/  
6. https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2
