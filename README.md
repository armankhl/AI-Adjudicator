# AI Adjudicator: An Ensemble Model for Evaluating LLM Responses

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

This repository contains the end-to-end pipeline for the **AI Adjudicator** project, an automated system designed to predict human preferences between two competing LLM responses. This system serves as a powerful tool to accelerate the LLM development lifecycle by providing a scalable, efficient, and reliable alternative to manual evaluation.

## Project Overview

Evaluating the quality of LLM outputs is a slow, expensive, and subjective process that bottlenecks rapid iteration. The AI Adjudicator addresses this by creating a sophisticated "judge" model that analyzes a `prompt` and two responses (`response_a`, `response_b`) to determine the winner with high fidelity to human judgment.

This project is implemented in three distinct phases:
1.  **Data Foundation & Governance:** A high-purity data filtration pipeline that cleans, validates, and de-duplicates the raw dataset, creating a "golden" source of truth.
2.  **Prompt Classification:** A Teacher-Student distillation approach where a large "teacher" LLM (`Zephyr-7B-Beta`) labels prompts, and a smaller, faster "student" model (`RoBERTa`) is trained to perform real-time classification.
3.  **Ensemble Judge Model:** A Mixture of Experts (MoE) architecture where a `Universal Model` handles general prompts and a specialized `Code Expert` model judges programming-related queries, with their predictions combined for a final, accurate verdict.

## Key Features

*   **High-Purity Data Pipeline:** An aggressive, multi-step filtration process that retains only ~70% of the highest-quality data by applying textual, statistical, and semantic validation rules.
*   **Teacher-Student Distillation:** An efficient workflow for creating a fast, production-ready prompt classifier without extensive manual labeling.
*   **Mixture of Experts (MoE) Architecture:** A robust ensemble model using `LightGBM` that combines a general-purpose judge with domain-specific experts for nuanced and accurate evaluation.
*   **Leakage-Proof Validation:** Employs `GroupKFold` on prompts to ensure the model is always evaluated on unseen prompts, providing a true measure of its generalization capability.

## Project Structure

```
AI-Adjudicator/
├── notebooks/              # Jupyter notebooks for each project phase.
│   ├── 1-Phase1_DataFoundation_and_Governance.ipynb
│   ├── 2-Phase2_Prompt_Classification.ipynb
│   └── 3-Phase3_Ensemble_Modeling.ipynb
├── data/
│   └── sample_dataset.csv  # A small sample dataset for quick runs.
├── report/
│   └── AI_Adjudicator_Project_Report.pdf # Full project report (in Persian).
├── .gitignore              # Files to be excluded from version control.
├── LICENSE
├── README.md
└── requirements.txt        # Python dependencies.
```

## Getting Started

Follow these steps to set up the environment and run the project.

### 1. Prerequisites

*   Python 3.10+
*   An NVIDIA GPU with at least 16GB VRAM is recommended for Phase 2.
*   A Together AI API key (for an alternative LLM teacher in Phase 2).

### 2. Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/AI-Adjudicator.git
    cd AI-Adjudicator
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Download SpaCy Model:**
    ```bash
    python -m spacy download en_core_web_lg
    ```

4.  **Set up Environment Variables (Optional):**
    If you wish to use an external LLM in Phase 2, create a file named `.env` in the project root and add your API key:
    ```
    TOGETHER_API_KEY="your-api-key-here"
    ```

### 3. Data

A small `sample_dataset.csv` is included in the `/data` directory to allow for a quick run of the pipelines. For full-scale training, you will need to replace this with your complete dataset.

## How to Run the Pipeline

The project is structured across several notebooks that should be run in order.

1.  **Phase 1: Data Foundation & Governance**
    Run the `notebooks/1-Phase1_DataFoundation_and_Governance.ipynb` notebook to process the raw data and generate the high-purity "golden" dataset. This notebook applies all cleaning and validation steps.

2.  **Phase 2: Prompt Classification**
    Run `notebooks/2-Phase2_Prompt_Classification.ipynb` to create the prompt classifier. This notebook demonstrates the Teacher-Student distillation process, where a large LLM generates labels to train a smaller `RoBERTa` model.

3.  **Phase 3: Ensemble Judge Model**
    Run `notebooks/3-Phase3_Ensemble_Modeling.ipynb` to perform feature engineering and train the final judge model. This notebook contains two modes:
    *   **Short Pipeline:** Uses a small, pre-labeled dataset for rapid development and testing.
    *   **Full Pipeline:** Uses the golden dataset from Phase 1 and the classifier from Phase 2 for a complete, end-to-end run.

## Results & Analysis

*   **Data Pipeline:** The data foundation pipeline successfully filtered an initial dataset of 57,477 records down to a high-purity set of **39,966 records (69.5% retention)**, removing duplicates, non-English text, and semantically similar prompts.
*   **Prompt Classifier:** The student `RoBERTa` model, trained on only 300 LLM-generated labels, achieved a baseline **F1-score of 0.215**. This proves the viability of the distillation approach but highlights the need for a larger labeled dataset to improve performance.
*   **Judge Model:** The ensemble model achieved **100% accuracy** on the small, held-out test sets (60 samples in the short pipeline, 100 in the full pipeline).
    *   **Insight:** This perfect score is a classic sign of **overfitting** on a small, homogenous dataset. The model has likely found simple, non-generalizable shortcuts (e.g., `delta_word_count`). While the architectural concept is validated, the model must be retrained on the full 39k+ dataset to obtain a realistic performance metric.

## Future Work

The current system serves as a robust proof of concept. The strategic roadmap for evolving this into a production-ready tool includes:
1.  **Full-Scale Data Labeling & Retraining:** Complete the Phase 2 labeling process for the entire 39k golden dataset and retrain the Phase 3 ensemble model to get a true measure of its performance.
2.  **Expand Expert Models:** Develop more specialist models for other key categories like `Creative Content` and `Factual Information`.
3.  **Explore Advanced Architectures:** Move from `LightGBM` to Transformer-based architectures using a `pairwise ranking loss` to better capture deep semantic nuances.
4.  **Full Operationalization (MLOps):**
    *   Containerize the entire pipeline using Docker.
    *   Deploy the models as a microservice using a framework like BentoML or FastAPI.
    *   Implement a Champion/Challenger framework for A/B testing.
    *   Set up live monitoring dashboards (Grafana, Evidently AI) for drift detection and operational health.
    *   Create a Human-in-the-Loop (HITL) system for continuous feedback and data collection.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.