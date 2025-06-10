
# Physical Activity Classification with Mobile Sensors

This repository contains the full development of a classification system for physical activities using data collected from mobile sensors (accelerometer and gyroscope) and machine learning models. It includes scripts, notebooks, collected data, serialized objects, an online classification prototype, and the final project report.

---

## 📁 Repository Structure

```
.
├── 📁 data/
│   ├── activity_data_Govea.csv
│   ├── activity_data_Govea.txt
│   ├── activity_data_Angel.txt
│   ├── activity_data_Augusto.txt
│   ├── Angel_data.obj
│   ├── Augusto_data.obj
│   └── Govea_data.obj
│
├── 📁 notebooks/
│   ├── Modelos_proyecto.ipynb
│   └── Mejor_modelo_proyecto.ipynb
│
├── 📁 scripts/
│   ├── data_acquisition.py
│   ├── data_processing.py
│   ├── data_plot.py
│   ├── communication_test.py
│   └── online_prototype.py
│
├── 📄 borrador-9.pdf
└── README.md
```

---

## 🧠 Project Overview

The goal of this project was to develop a model capable of identifying specific physical activities based on signals captured by mobile sensors. Activities include walking, running, shadow boxing, among others.

Project highlights:
- Data collected via a mobile app.
- Key statistical features extracted.
- Multiple classification models trained and evaluated.
- A functional online classification prototype implemented.

---

## 🚀 How to Run

1. Clone this repository:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. To run the online classification:

```bash
python scripts/online_prototype.py
```

4. To explore model training and evaluation:

Open the notebooks in Jupyter or VS Code:
- `notebooks/Modelos_proyecto.ipynb`
- `notebooks/Mejor_modelo_proyecto.ipynb`

---

## 📊 Models and Metrics

Used models:
- XGBoost
- Random Forest
- Extra Trees
- Gradient Boosting
- Naive Bayes

Techniques applied:
- Feature selection (`SelectKBest`)
- Hyperparameter optimization (`GridSearchCV`)
- Nested cross-validation

Accuracy achieved: **up to 98%** with XGBoost and Extra Trees.

---

## 📄 Final Report

The full development and results are detailed in [`borrador-9.pdf`](borrador-9.pdf).

---

## 📚 Credits

This project was developed by [Your Name] as part of the machine learning course at [Your University or Course].

---

## 📜 License

[MIT License](LICENSE) – Free to use with attribution.
