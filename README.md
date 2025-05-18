# 🧠 Autism Spectrum Disorder (ASD) Detection Using Machine Learning

This project explores the detection of Autism Spectrum Disorder (ASD) through machine learning models applied to both behavioral questionnaire data and neuroimaging (fMRI) data from the ABIDE dataset. It aims to demonstrate the potential of AI-driven tools in supporting early and accurate ASD diagnosis using multimodal data.

---

## 📂 Project Structure

- `models on the questionnaire dataset/`: ML models applied to structured questionnaire data (CSV format)
- `models on the ABIDE dataset/`: Feature extraction and classification on ABIDE fMRI data using Nilearn
- `datasets/`: datasets used

---

## 🔍 Objective

To compare the performance of multiple machine learning algorithms on:
- Questionnaire-based structured data
- Preprocessed ABIDE neuroimaging data

and to evaluate their potential for real-world clinical support in ASD diagnosis.

---

## 📊 Key Results

### Questionnaire Dataset
| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    | 0.992    | 1.000     | 0.975  | 0.987    |
| SVM                    | 0.978    | 1.000     | 0.925  | 0.961    |
| Decision Tree          | 0.957    | 0.925     | 0.925  | 0.925    |

### ABIDE fMRI Dataset
| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.807    | 0.772     | 0.772  | 0.772    |
| SVM                 | 0.677    | 0.666     | 0.736  | 0.700    |
| k-NN                | 0.673    | 0.692     | 0.409  | 0.514    |

---

## 🛠️ Tools & Technologies Used

- **Python**: Core language
- **Pandas, NumPy**: Data processing
- **Scikit-learn**: Machine learning modeling
- **Nilearn**: fMRI preprocessing and analysis
- **Matplotlib, Seaborn**: Visualization
- **Jupyter Notebook / VS Code**: Development environments

---

## 📌 Highlights

- Applied classical ML models like Logistic Regression, SVM, Decision Tree, and k-NN
- Demonstrated logistic regression's generalizability across structured and unstructured data
- Emphasized the impact of preprocessing and PCA for high-dimensional neuroimaging data
- Discussed clinical relevance and ethical considerations in medical AI

---

## 🧪 Future Work

- Integrate deep learning models (e.g., CNNs) for raw fMRI data
- Combine questionnaire and imaging data in a unified pipeline
- Develop lightweight diagnostic tools for real-time clinical deployment

---

## 🧠 Keywords

Autism Spectrum Disorder (ASD), Machine Learning, Neuroimaging, ABIDE Dataset, Diagnostic Questionnaires, Logistic Regression, Support Vector Machines, Dimensionality Reduction, Principal Component Analysis (PCA), Multimodal Analysis, fMRI, Feature Selection, Clinical Decision Support, Medical AI, Early Diagnosis, Data Preprocessing, Model Interpretability, Computational Psychiatry

---

## 📄 License

This project is for academic and educational purposes only. Please cite the repository if you use or adapt it in your work.

---

## 📬 Contact

For questions or collaborations, please reach out via GitHub issues or email: `youremail@example.com`
