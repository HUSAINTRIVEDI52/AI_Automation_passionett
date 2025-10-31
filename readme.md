# 👶 Weighted Nanny-Parent Matching Engine

## 🌟 1. Project Overview

This repository contains the complete Python pipeline for the **Weighted Nanny-Parent Matching Engine** designed for **The Urban Parents Club**. The goal is to intelligently pair parent requirements with nanny profiles using a **rule-based weighted scoring algorithm** to maximize compatibility.

The project incorporates best practices from Data Science, including a thorough data cleaning stage and a **Train/Test evaluation** phase to rigorously check the algorithm's performance on **unseen data** and prevent overfitting.

---

## 🚀 2. Getting Started

Follow these steps to set up the project locally and run the matching and evaluation pipeline.

### Prerequisites

* **Python 3.8+**
* **Git**
* **Source Data Files** (Must be in the repository root):
    * `Urban_Parents_Club_Data.xlsx`
    * `Parent.data.xlsx`
    * `Nanny.data.xlsx`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Husaintrivedi52/nanny-match-engine.git](https://github.com/YourUsername/nanny-match-engine.git)
    cd nanny-match-engine
    ```

2.  **Install dependencies:**
    The project relies on core data science and string matching libraries.
    ```bash
    pip install pandas numpy scikit-learn fuzzywuzzy[speedup]
    ```

---

## 💻 4. Pipeline Stages & Execution

The system runs in three distinct stages, producing intermediate and final output files.

### Stage 1: Data Cleaning and Standardization

* **Goal:** Ensure data quality, handle missing values, validate against business rules, and standardize text (locations, languages) using fuzzy matching.
* **Action:** Execute your primary data cleaning script.
* **Output:** Cleaned data files (e.g., `nanny_details_cleaned.csv`, `parent_details_cleaned.csv`).

### Stage 2: Data Splitting (Train/Test)

This stage partitions the clean data to simulate real-world evaluation on unseen data.

* **Goal:** Split the Parent and Nanny datasets into reproducible Training (75%) and Testing (25%) sets. 
* **Input:** Cleaned data from Stage 1.
* **Output Files:** `parents_train.csv`, `nannies_train.csv`, `parents_test.csv`, `nannies_test.csv`.

### Stage 3: Model Evaluation and Overfitting Check

This final stage calculates the weighted compatibility score on both data sets, reports metrics, and checks for model stability.

* **Action:** Execute the main evaluation script (which contains the matching algorithm).
* **Process:** The algorithm is run on the **Training set** to establish a baseline score, and then on the **Testing set** for true validation.
* **Key Output:** The script performs an **Overfitting Check**: if the Test Average Score drops significantly (e.g., >10 points) compared to the Training Average Score, it flags a potential stability issue.
* **Output Files:**
    * `day7_testing_results.csv`: Detailed top 3 matches for all parents in the testing set.
    * `day7_metrics.csv`: Summary DataFrame showing the final average scores and the overfitting warning status.

---

## ⚖️ 5. Weighted Matching Algorithm

The core of the system is the `calculate_match_score` function, which dictates the compatibility score (0-100) using the following weights:

| Factor | Description | Weight | Scoring Detail |
| :--- | :--- | :--- | :--- |
| **`Location_Match`** | Proximity/Preferred Area | 30% | **100:** Exact Match |
| **`Language_Match`** | Communication | 20% | **100:** Nanny speaks **ANY** required language. |
| **`Experience_Match`**| Child Age Need | 20% | **100:** Meets required experience (0+ years for `<3`, 3+ years for `>=3`). |
| **`Availability_Match`**| Hours Alignment | 15% | **100:** Nanny hours meet FT/PT/24HR request thresholds. |
| **`Travel_Willingness`**| Commute Flexibility | 15% | **100:** Willing to travel. **50:** Not willing (partial constraint). |

---

## 🤝 6. Contributing

Contributions are welcome! If you find a way to improve the accuracy, efficiency, or maintainability of the matching rules, please submit a pull request.

* **Refine Rules:** Improve the thresholds or logic in `score_experience` or `score_availability`.
* **Enhance Data Quality:** Suggest additions to the fuzzy matching master lists.
* **Documentation:** Improve clarity or detail in any project documentation.