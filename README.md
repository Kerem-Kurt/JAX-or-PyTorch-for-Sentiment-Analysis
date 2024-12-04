# JAX or PyTorch: Financial News Sentiment Analysis

## Author
**Kerem Haktan Kurt**  
**September 2024**

---

## Overview
This project focuses on developing a **financial sentiment analysis model** to classify Turkish financial news as either positive or negative. The task was approached using **two frameworks**: PyTorch and JAX, leveraging various models like BERT, XLM-R, and Llama.

---

## Data Description
- **Format**: `.txt` files in `positive` and `negative` folders.  
- **Language**: Turkish financial news.  
- **Source**: Financial institutions (e.g., Yapı Kredi, Garanti) and news websites.  
- **Processing**:
  - Data shuffled and split into an 80/20 train-test ratio.
  - Tokenized inputs with padding for uniformity.

---

## Training
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

## Improvements Achieved

### PyTorch
- **Accuracy Increase**: From **42.69%** to **90.35%** in just 2 epochs.
- **Precision**: Improved from **0.7563** to **0.9039**.
- **F1 Score**: Boosted from **0.2588** to **0.9036**.
- **Training Loss**: Decreased from **0.6945** to **0.5170**.
- **Evaluation Loss**: Reduced from **0.7299** to **0.4569**.

The model demonstrated significant improvements in performance metrics over two epochs, achieving a nearly perfect balance of precision and recall, with optimized training and evaluation losses.

### JAX
- **Accuracy Increase**: From **45.54%** to **86.31%** after 3 epochs.
- **Precision**: Peaked at **0.9355** during the second epoch.
- **Recall**: Improved from **0.0162** to **0.8649**.
- Successfully overcame initial bias in predictions, as seen in confusion matrix progressions.

Despite a steeper learning curve with JAX, the model's performance steadily improved with careful state management and custom training functions.

---

## PyTorch vs JAX
| **Aspect**         | **PyTorch**                        | **JAX**                                    |
|---------------------|------------------------------------|--------------------------------------------|
| **Graph Type**      | Dynamic computation graphs.       | Static computation graphs (XLA optimized). |
| **Ease of Use**     | Intuitive and flexible.           | Functional style, requires explicit states.|
| **Performance**     | Fast, mature environment.         | Faster with TPUs and XLA optimization.     |
| **Community**       | Large support and documentation.  | Growing, niche audience.                   |

---

## Key Actions Taken
1. **Framework-Specific Implementations**:
   - PyTorch: Leveraged built-in `Trainer` for seamless training and evaluation.
   - JAX: Implemented custom training and evaluation loops to handle functional programming constraints.
   
2. **Optimization Techniques**:
   - Data augmentation.
   - Gradient clipping (PyTorch: 1.0).
   - Learning rate adjustments (2e-5).
   - Customized batch sizes and epochs.

3. **Handling Overfitting**:
   - Balanced dataset weights for sentiment classes.
   - Early stopping in PyTorch to prevent unnecessary epochs.
   - Careful parameter initialization to avoid biased predictions.

4. **Experimentation**:
   - Explored multiple models like **BERT**, **XLM-R**, and **Llama3.1** before finalizing **BERTurk** for its compatibility with Turkish text.

---

## 📚 References
- [BERTurk Model Documentation](https://huggingface.co/dbmdz/bert-base-turkish-cased)  
- [JAX Official Documentation](https://jax.readthedocs.io/en/latest/quickstart.html)  
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/tasks/sequence_classification)  
- Additional Resources on **Hyper-Parameter Optimization** and JAX-PyTorch comparisons.

---

For a Full report, check the pdf in the repository!
