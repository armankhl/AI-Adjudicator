# AI Adjudicator: An Ensemble Learning System for LLM Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Frameworks](https://img.shields.io/badge/Frameworks-HuggingFace%20%7C%20LightGBM-ff69b4.svg)](#tech-stack)

This repository contains the end-to-end pipeline for the **AI Adjudicator** project, a machine learning system designed to automatically evaluate the quality of Large Language Model (LLM) responses. By analyzing a prompt and two competing responses, the model predicts which one aligns better with human preferences, providing a scalable and efficient alternative to manual evaluation.

## 1. The Problem

The rapid evolution of LLMs has created a significant MLOps bottleneck: evaluating model quality. Manual assessment, while the gold standard, is slow, expensive, subjective, and fails to scale with the pace of modern AI development. This project was initiated to build an automated "judge" that can serve two critical functions:
1.  **A scalable reward model** for future Reinforcement Learning from Human Feedback (RLHF) initiatives.
2.  **An automated evaluation component** for our CI/CD pipeline, enabling rapid, data-driven iteration on new LLM versions.

## 2. The Solution: A Phased Approach

The AI Adjudicator is built using a structured, three-phase approach, moving from raw data to a sophisticated, domain-aware ensemble model.

### **Phase 1: Data Foundation & Governance**
A high-purity data filtration pipeline cleans, validates, and de-duplicates the raw dataset. This multi-step process applies textual, statistical, and semantic rules to transform a noisy, 57k-record dataset into a versioned, 39k-record "golden" dataset that serves as a reliable source of truth.

### **Phase 2: Prompt Classification via Teacher-Student Distillation**
Recognizing that not all prompts are judged by the same criteria, we developed a fast and accurate prompt classifier. A large, powerful "teacher" LLM (`HuggingFaceH4/zephyr-7b-beta`) generates high-quality labels, which are then used to train (distill knowledge into) a smaller, faster "student" model (`roberta-base`). This student model becomes the router in our final architecture.

### **Phase 3: Ensemble Judge Model**
The core of the system is a **Mixture of Experts (MoE)** architecture built with `LightGBM`. Instead of a single monolithic model, we use:
*   A **Universal Model** trained on all non-code-specific features.
*   A **Code Expert Model** trained on all features, with a special focus on code-specific metrics (e.g., cyclomatic complexity, AST analysis).

The prompt classifier from Phase 2 routes requests to the appropriate model(s), and their predictions are combined to produce a final, nuanced judgment.

## 3. System Architecture

The final inference pipeline follows the Mixture of Experts (MoE) pattern:

```mermaid
graph TD
    A[Input: Prompt, Res_A, Res_B] --> B{Prompt Classifier (RoBERTa Student)};
    B --> |"Identifies Category"| C{"Is Category 'Code & Programming'?"};
    C -- No --> E[Universal Model (LGBM)];
    C -- Yes --> D[Code Expert Model (LGBM)];
    E --> F[Prediction];
    D --> G[Weighted Combination];
    E --> G;
    G --> F;
    F --> H[Output: Winner (A, B, or Tie)];

    style B fill:#dae8fc,stroke:#6c8ebf
    style D fill:#d5e8d4,stroke:#82b366
    style E fill:#d5e8d4,stroke:#82b366
```

## 4. Key Technical Highlights

*   **High-Purity Data Pipeline:** An aggressive, multi-step filtration process using `pandera`, `langdetect`, and semantic de-duplication with `sentence-transformers` to ensure data quality.
*   **Checkpointing for Reproducibility:** The Phase 1 pipeline saves intermediate dataframes, allowing for faster iteration and debugging.
*   **Teacher-Student Distillation:** A practical workflow for creating a fast, production-ready classifier without the need for extensive manual labeling.
*   **Mixture of Experts (MoE):** A robust ensemble architecture that combines a generalist with specialists for superior, domain-aware performance.
*   **Leakage-Proof Validation:** Employs `GroupKFold` on prompts throughout training and evaluation to ensure the model is always tested on unseen prompts, providing a true measure of its ability to generalize.

## 5. Repository Structure

```
AI-Adjudicator/
├── notebooks/              # Jupyter notebooks for each project phase.
│   ├── 1-Phase1_DataFoundation_and_Governance.ipynb
│   ├── 2-Phase2_Prompt_Classification.ipynb
│   ├── 3a-Phase3_Short_Pipeline.ipynb
│   └── 3b-Phase3_Full_Pipeline.ipynb
│
├── data/
│   └── sample_dataset.csv  # A small sample dataset for quick runs and demos.
│
├── output/                 # (Gitignored) Directory for generated files.
│   ├── high_purity_golden_datasets/
│   ├── labeled_datasets/
│   ├── models/
│   └── pipeline_checkpoints/
│
├── report/
│   └── AI_Adjudicator_Project_Report.pdf # Full project report (in Persian).
│
├── .gitignore              # Files to be excluded from version control.
├── LICENSE
├── README.md               # You are here.
└── requirements.txt        # All Python dependencies for the project.
```

## 6. Getting Started

### Prerequisites
*   Python 3.10+
*   An NVIDIA GPU (T4 or better) is **required** for Phase 2 and highly recommended for Phase 3.
*   (Optional) A Together AI API key for an alternative LLM teacher in Phase 2.

### Installation
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YourUsername/AI-Adjudicator.git
    cd AI-Adjudicator
    ```

2.  **Create a Virtual Environment and Install Dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Download SpaCy Model:**
    ```bash
    python -m spacy download en_core_web_lg
    ```

4.  **Set Up Environment Variables (Optional):**
    If using an external LLM API, create a `.env` file in the project root:
    ```
    TOGETHER_API_KEY="your-api-key-here"
    ```

## 7. How to Run the Pipeline

The project is designed to be run sequentially through the notebooks located in the `notebooks/` directory.

1.  **Phase 1: Data Foundation & Governance**
    Run `1-Phase1_DataFoundation_and_Governance.ipynb` to process the `data/sample_dataset.csv`. This will create the high-purity "golden" dataset and save it to the `output/` directory.

2.  **Phase 2: Prompt Classification**
    Run `2-Phase2_Prompt_Classification.ipynb` to create the prompt classifier. This notebook uses a powerful "teacher" LLM to label a subset of the golden data, then trains a smaller, faster "student" `RoBERTa` model. The final model is saved to `output/models/`.

3.  **Phase 3: Ensemble Judge Model**
    Run either `3a-Phase3_Short_Pipeline.ipynb` or `3b-Phase3_Full_Pipeline.ipynb`.
    *   The **Short Pipeline** runs on the small, pre-labeled dataset from Phase 2 for a quick demonstration.
    *   The **Full Pipeline** uses the large golden dataset from Phase 1 and the classifier from Phase 2 for a complete, end-to-end run.

## 8. Results & Analysis

| Phase | Input Data | Output | Key Metric & Finding |
| :--- | :--- | :--- | :--- |
| **1. Data Foundation** | 57,477 raw records | **39,966** clean records | **69.5% retention rate**, indicating successful removal of noise and duplicates. |
| **2. Prompt Classifier** | 300 LLM-labeled prompts | RoBERTa model | **F1-score: 0.215**. This proves the distillation concept but shows the model is severely under-trained and needs more labeled data. |
| **3. Judge Model**| 300-500 samples | Ensemble model | **Accuracy: 100%**. This perfect score is a clear sign of **overfitting** on a small, homogenous dataset. The model learned simple heuristics rather than complex patterns. |

**Crucial Insight:** The 100% accuracy in Phase 3 validates that the feature engineering and ensemble architecture are functional. However, it also proves that the model's performance is entirely dependent on the quality and scale of its training data. The primary bottleneck to creating a truly robust judge is scaling the data labeling process outlined in Phase 2.

## 9. Future Work & Roadmap

The current system is a successful proof-of-concept. The path to a production-ready tool is clear:
1.  **Scale Data & Retrain (Priority 1):** Dedicate GPU resources to complete the Phase 2 labeling process for the entire 39k golden dataset and retrain the Phase 3 ensemble model to obtain a realistic performance metric.
2.  **Expand Expert Models:** Develop more specialist models for other key categories like `Creative Content` and `Factual Information`.
3.  **Explore Advanced Architectures:** Migrate from `LightGBM` to Transformer-based architectures using a `pairwise ranking loss`, which may better capture deep semantic nuances between responses.
4.  **Full MLOps Deployment:**
    *   Containerize the entire pipeline using **Docker**.
    *   Deploy the models as a microservice using **BentoML** or **FastAPI**.
    *   Implement a **Champion/Challenger** framework for safe A/B testing of new model versions.
    *   Set up live monitoring dashboards (**Grafana**, **Evidently AI**) for drift detection.
    *   Create a **Human-in-the-Loop (HITL)** system for continuous feedback and data collection.

## 10. Tech Stack

*   **Data Processing:** Pandas, Pandera, NumPy
*   **NLP & Feature Engineering:** spaCy, NLTK, Textstat, Radon, BeautifulSoup4
*   **Machine Learning:** Scikit-learn, LightGBM
*   **Deep Learning & Embeddings:** PyTorch, Hugging Face (`transformers`, `datasets`, `sentence-transformers`), BERT-Score
*   **Notebook Environment:** Jupyter / Google Colab

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.