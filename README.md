# 📝 Auto Tagging Support Tickets Using LLM

## 📋 Task Overview

This project implements an automated support ticket tagging system using **Large Language Models (LLMs)**. The system categorizes customer support tickets into predefined classes using three distinct AI strategies: **Zero-shot learning**, **Few-shot learning**, and a **Fine-tuned BERT model**.

## 🎯 Objective

* Automatically categorize support tickets into 6 predefined classes.
* Compare the performance of Transformer-based architectures.
* Provide the top 3 most probable tags per ticket to assist human agents.

---

## 📊 Dataset & Categories

* **Type:** Synthetic Support Ticket Dataset
* **Size:** 500 tickets
* **Classes:** 1. Technical Issue
2. Billing/Account
3. Feature Request
4. Product Inquiry
5. Complaint
6. Other

---

## 🛠️ Technologies Used

* **Core:** Python 3.8+, PyTorch
* **Models:** BERT (base-uncased), BART (large-MNLI), GPT-2
* **Libraries:** Hugging Face Transformers, Scikit-learn, NLTK
* **Visualizations:** Matplotlib, Seaborn
* **Utility:** TQDM, Pandas, NumPy

---

## 📁 Project Structure

```text
Auto Tagging Support Tickets/
├── code.ipynb                            # Main development notebook
├── README.md                             # Project documentation
│
├── fine_tuned_bert/                      # Local storage for weights
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer_config.json
│
├── results/                              # Evaluation plots & CSVs
│   ├── category_distribution.png
│   ├── bert_confusion_matrix.png
│   ├── model_comparison_detailed.png
│   └── radar_comparison.png


```

---

## 🚀 Implementation Highlights

### 1. Zero-Shot Learning (BART)

Utilizes the `BART-large-MNLI` model to classify text without any prior training on the specific dataset. It treats the categories as "hypotheses" in a Natural Language Inference (NLI) task.

### 2. Few-Shot Learning (Prompt Engineering)

Implements GPT-2 with structured prompts. By providing 2-3 examples (shots) per category within the prompt, the model learns the context and nuance of the support tickets dynamically.

### 3. Fine-tuned BERT

A `BERT-base-uncased` model was fine-tuned over 3 epochs using a custom PyTorch Dataset. This approach yields the highest domain-specific accuracy.

---

## 📈 Results Comparison

| Approach | Top-1 Accuracy | Top-3 Accuracy | Training Required | Inference Speed |
| --- | --- | --- | --- | --- |
| **Zero-shot** | ~65% | ~80% | No | Fast |
| **Few-shot** | ~72% | ~85% | No | Slow |
| **Fine-tuned** | **~85%** | **~94%** | **Yes** | **Fast** |

---

## 🏆 Key Features

* **Top-K Analysis:** Measures how often the correct tag appears in the top 3 predictions.
* **Production-Ready Class:** Includes a `SupportTicketTagger` wrapper for easy deployment.
* **Hybrid Fallbacks:** The system can fall back from Fine-tuned to Zero-shot if a new, unseen category is introduced.
* **Visualization Suite:** Detailed radar charts and confusion matrices for performance auditing.

---

## 💻 Installation & Usage

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/auto-tagging-support-tickets.git
cd auto-tagging-support-tickets

# Install dependencies
pip install -r requirements.txt

```

### Quick Inference

```python
from auto_tagger import SupportTicketTagger

# Initialize with the finee-tuned model
tagger = SupportTicketTagger(model_type='fine-tuned')

ticket = "The app crashes every time I try to upload a file."
result = tagger.tag_ticket(ticket)

print(f"Primary Tag: {result['top_1']}")
print(f"Confidence: {result['confidence']:.2f}")

```

---

## 👨‍💻 Author

**AI/ML Engineering Intern** DevelopersHub Corporation

*March 2026*
