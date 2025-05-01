# 🌊 Titanic Survival Predictor
An end-to-end machine learning project showcasing data exploration, preprocessing, model training and evaluation using the classic Titanic dataset. I implemented and compared three classifiers—K-Nearest Neighbors, Random Forest and Support Vector Machine—to predict passenger survival.

## 🚀 Features

- **Data Exploration & Visualization**  
  - Inspect raw data, plot distributions, boxplots and pairwise relationships.  
  - Generate a heatmap of feature correlations.  
- **Data Cleaning & Preprocessing**  
  - Handle missing values (imputation by mean and forward/backward fill).  
  - Encode categorical variables and scale numerical features.  
- **Model Training & Evaluation**  
  - Train KNN, SVM and Random Forest classifiers.  
  - Compute training vs. validation accuracy, confusion matrices and classification reports.  
- **Cross-Validation**  
  - Perform stratified 10-fold CV to assess performance and stability.  
  - Compare models with combined boxplots and bar charts (mean ± std).  
- **Visualization Outputs**  
  - All charts (exploratory analysis + model comparison) are saved to the `plots/` directory.

## ⚙️ Installation & Usage
1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/titanic-survival-predictor.git
   cd titanic-survival-predictor
    ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the prediction script**
   ```bash
   python3 prediction.py
   ```
4. **Explore the results**
- Check the console output for accuracy scores and classification reports.
- Open the plots/ folder to review all generated visualizations.

## 🎯 Skills Demonstrated
- Python & pandas for data manipulation
- Matplotlib & Seaborn for visualization
- scikit-learn for model building, evaluation and cross-validation