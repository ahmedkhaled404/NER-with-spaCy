<h1>Named Entity Recognition with spaCy<h1/>

**1. Introduction**

**Problem Statement:** Named Entity Recognition (NER) is a subtask of information extraction that seeks to identify and classify named entities in text into predefined categories such as persons, organizations, and locations. This project aims to develop a custom NER pipeline using spaCy, a popular NLP library. The objective is to build a model that can accurately identify and classify entities in text data, optimize the pipeline for efficiency.

**Objectives:**

-   Develop a pipeline for NER using spaCy.
-   Implement and classify named entitie.
-   Optimize the pipeline for efficient entity recognition.

**2.Data Description**

The dataset for this project is sourced from Kaggle and can be accessed [here](https://www.kaggle.com/datasets/naseralqaydeh/named-entity-recognition-ner-corpus/). It consists of:

-   **Sentence**: The complete sentence in text format.
-   **POS**: List of Part-of-Speech (POS) tags for each word in the sentence.
-   **Tag**: List of Named Entity Recognition (NER) tags for each word in the sentence.

**Example Data**

-   **Sentence**: "Thousands of demonstrators have marched through London to protest the war in Iraq and demand the withdrawal of British troops from that country."
-   **POS**: ['NNS', 'IN', 'NNS', 'VBP', 'VBN', 'IN', 'NNP', 'TO', 'VB', 'DT', 'NN', 'IN', 'NNP', 'CC', 'VB', 'DT', 'NN', 'IN', 'JJ', 'NNS', 'IN', 'DT', 'NN', '.']
-   **Tag**: ['O', 'O', 'O', 'O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'B-gpe', 'O', 'O', 'O', 'O', 'O']

**3. Baseline Experiments**

**Goal:** Establish a baseline performance for the NER model using spaCy's pre-trained models and evaluate its effectiveness on the custom dataset.

**Methodology:**

-   Data Preprocessing
-   Loaded the pre-trained English model en_core_web_sm from spaCy.
-   Created and trained an NER component using the custom dataset.
-   Custom NER Model Training, Evaluation, and Visualization of results.

**Results & Conclusion:**

-   The baseline model performance was evaluated, and initial results indicated the effectiveness of spaCy's pre-trained models for the custom dataset, despite that, the model performance might be biased due to data imbalance.
-   Model needs more generalization.
-   These problems were not encountered due to the time limitations.

**4. Methodology breakdown**

1.  **Data Import and Inspection:**
    -   Loaded data from CSV file.
    -   Checked for null values and data types.
2.  **Data Exploration (EDA):**
    -   Verified the shape and distribution of data.
    -   Checked for duplicates and removed them.
    -   Checked distribution of labels.

![image](https://github.com/user-attachments/assets/8d957b50-ff81-49c6-b270-612e785723d3)


(fig shows the distribution of labels on the data)

-   Label Distribution Analysis across datasets

    To analyze the distribution of labels across training, validation, and test datasets.

![image](https://github.com/user-attachments/assets/f43690a3-d414-405f-b40f-65d9ce7d6f88)

1.  **Data Preprocessing:**
    -   Converted sentences and annotations into spaCy's format
    -   split the data into training, validation, and test sets.
2.  **Model Training:**
    -   Fine-tuned the NER model using spaCy’s training pipeline.
    -   Trained for 20 Iterations (epochs)
3.  **Model Evaluation:**
    -   Evaluated the model on validation and test sets using classification metrics.
4.  **Results:**
    -   **Validation Classification Report**

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| B-eve | 1.00      | 0.00   | 0.00     | 1       |
| B-org | 1.00      | 0.00   | 0.00     | 3       |
| B-per | 1.00      | 0.00   | 0.00     | 1       |
| B-tim | 1.00      | 0.50   | 0.67     | 8       |
| I-art | 0.00      | 0.00   | 1.00     | 2       |
| I-eve | 0.50      | 1.00   | 0.67     | 1       |
| I-geo | 0.00      | 0.00   | 1.00     | 7       |
| I-org | 0.62      | 0.40   | 0.49     | 25      |
| I-per | 0.67      | 0.33   | 0.44     | 6       |
| I-tim | 0.87      | 0.85   | 0.86     | 65      |
| O     | 1.00      | 1.00   | 1.00     | 16451   |

-   

| Accuracy | Macro Avg | Weighted Avg |
|----------|-----------|--------------|
| 99.00    | 0.70      | 1.00         |

-   **Test Classification Report:**

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| B-eve | 1.00      | 0.00   | 0.00     | 1       |
| B-geo | 1.00      | 0.00   | 0.00     | 1       |
| B-org | 1.00      | 0.33   | 0.50     | 3       |
| B-tim | 0.60      | 0.43   | 0.50     | 7       |
| I-art | 1.00      | 0.00   | 0.00     | 1       |
| I-eve | 0.75      | 1.00   | 0.86     | 3       |
| I-geo | 0.40      | 0.40   | 0.40     | 5       |
| I-org | 0.45      | 0.36   | 0.40     | 14      |
| I-per | 1.00      | 0.14   | 0.25     | 7       |
| I-tim | 0.91      | 0.92   | 0.91     | 75      |
| O     | 1.00      | 1.00   | 1.00     | 16446   |

-   

| Accuracy | Macro Avg | Weighted Avg |
|----------|-----------|--------------|
| 99.00    | 0.83      | 1.00         |

-   **Visualization of Entities**

    provided visual examples showing how the model's predictions compare to true annotations.

![image](https://github.com/user-attachments/assets/b93b6c2b-b25e-46e3-b50f-8b1736e84866)

**5. Overall Conclusion**

The project developed and fine-tuned a Named Entity Recognition pipeline using spaCy. The custom model demonstrated improved performance on the custom dataset compared to the baseline pre-trained model. Advanced experiments revealed the effectiveness of deep learning approaches for NER tasks and provided valuable insights into entity classification and pipeline optimization.

**Key Findings:**

-   Fine-tuning a pre-trained model can significantly improve performance on domain-specific datasets.
-   Label distribution analysis and visualization of entities are essential for understanding model performance and accuracy.

**Additional Information**

**1. Libraries and Tools Used**

| Library/Tool | Description                                                                                                          | Link                                                    |
|--------------|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| spaCy        | A popular NLP library used for developing and training NER models.                                                   | [spaCy Documentation](https://spacy.io/)                |
| Pandas       | A data manipulation library used for loading and processing the dataset.                                             | Pandas Documentation                                    |
| Matplotlib   | A plotting library used for visualizing label distributions and model results.                                       | [Matplotlib Documentation](https://matplotlib.org/)     |
| Seaborn      | A data visualization library based on Matplotlib, used for creating attractive and informative statistical graphics. | Seaborn Documentation                                   |
| Scikit-learn | A machine learning library used for generating classification reports and calculating performance metrics.           | [Scikit-learn Documentation](https://scikit-learn.org/) |

**2. External Resources**

| Resource          | Description                                                                   | Link                                                                                                 |
|-------------------|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Dataset           | Named Entity Recognition Corpus, including sentences, POS tags, and NER tags. | [Kaggle Dataset](https://www.kaggle.com/datasets/naseralqaydeh/named-entity-recognition-ner-corpus/) |
| Pre-trained Model | spaCy's en_core_web_sm Model used for initial NER training and evaluation.    | Model Details                                                                                        |
