# JAX or PyTorch: Financial News Sentiment Analysis

## Author
**Kerem Haktan Kurt**  
**September 2024**

---

## 📖 Overview
This project focuses on developing a **financial sentiment analysis model** to classify Turkish financial news as either positive or negative. The task was approached using **two frameworks**: PyTorch and JAX, leveraging various models like BERT, XLM-R, and Llama.

---

## 📊 Data Description
- **Format**: `.txt` files in `positive` and `negative` folders.  
- **Language**: Turkish financial news.  
- **Source**: Financial institutions (e.g., Yapı Kredi, Garanti) and news websites.  
- **Processing**:
  - Data shuffled and split into an 80/20 train-test ratio.
  - Tokenized inputs with padding for uniformity.

---

## 🤖 Models
### 1. **BERTurk**
- **Model**: [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased).
- **Why BERTurk?**  
  Pre-trained on Turkish literature, ideal for adding financial knowledge.

### 2. **XLM-R Large**
- Pre-trained multilingual RoBERTa, tuned for better understanding of Turkish.

### 3. **Llama 3.1 (8B)**  
- Large-scale model used with quantization for GPU memory efficiency.

---

## 🛠 Training
### **PyTorch**
- **Epochs**: 2-3  
- **Batch Size**: 2-16 (experimented).  
- **Optimizer**: AdamW.  
- **Performance**:
  - Final Accuracy: **90.35%**.
  - Significant boosts in **precision**, **recall**, and **F1-score** after two epochs.
- **Challenges**:
  - Overfitting mitigated with gradient clipping, data augmentation, and weight balancing.

### **JAX**
- **Epochs**: 3  
- **Framework**: Functional programming with `optax` for optimization.  
- **Performance**:
  - Final Accuracy: **86.31%**.  
  - Improvements seen in balanced predictions after 3 epochs.  

---

## 📈 Results
### **PyTorch Metrics**
- **Accuracy**: 42.69% → **90.35%**  
- **Training Loss**: 0.6945 → 0.5170  
- **Validation Loss**: 0.4634 → 0.4569  

### **JAX Metrics**
- **Accuracy**: 45.54% → **86.31%**  
- **Confusion Matrix**: Improved significantly over 3 epochs with reduced bias.

---

## ⚔️ PyTorch vs JAX
| **Aspect**         | **PyTorch**                        | **JAX**                                    |
|---------------------|------------------------------------|--------------------------------------------|
| **Graph Type**      | Dynamic computation graphs.       | Static computation graphs (XLA optimized). |
| **Ease of Use**     | Intuitive and flexible.           | Functional style, requires explicit states.|
| **Performance**     | Fast, mature environment.         | Faster with TPUs and XLA optimization.     |
| **Community**       | Large support and documentation.  | Growing, niche audience.                   |

---

## 📚 References
- [BERTurk Model Documentation](https://huggingface.co/dbmdz/bert-base-turkish-cased)  
- [JAX Official Documentation](https://jax.readthedocs.io/en/latest/quickstart.html)  
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/tasks/sequence_classification)  
- Additional Resources on **Hyper-Parameter Optimization** and JAX-PyTorch comparisons.

---

For a Full report, check the pdf in the repository!
