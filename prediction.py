import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from data_titanic import features_train, target_train, save_plot, features_validation, target_validation

output_dir = "plots"
pd.set_option('display.max_columns', None)
seed = 44

# ---------------- CROSS VALIDATION FUNCTION ----------------   
def cross_val(model, name):
    cv_results = cross_val_score(
    model, features_train, target_train, cv=kfold, scoring="accuracy"
    )
    print("\n")
    print("%"*150)
    print("\n")
    print(f"{name} Cross Validation: {cv_results.mean()} {cv_results.std()}")
    plt.figure() 
    plt.subplots(figsize=(15,10))
    plt.boxplot(cv_results, tick_labels=["{name} Cross Validation"])
    save_plot(f"{name}_cv_boxplot.png")

# ------------------------------ 1) MODEL PREDICTION -----------------------------
# ------------------------------ 1) MODEL PREDICTION -----------------------------
# --------------- 1.1) K Neighbors ---------------
knn = KNeighborsClassifier()

knn.fit(features_train, target_train)
knn_pred = knn.predict(features_validation)

# --------------- 1.2) Support Vector Machines ---------------
svc = SVC()
svc.fit(features_train, target_train)
svc_pred = svc.predict(features_validation)

# --------------- 1.3) Random Forest ---------------
r_forest = RandomForestClassifier(n_estimators=100,  max_depth=8)
r_forest.fit(features_train, target_train)
r_forest_pred = r_forest.predict(features_validation)

# ---------------------------------- 2) RESULTS ----------------------------------
# ---------------------------------- 2) RESULTS ----------------------------------
print("\n\n")
print("*"*79)
print("Accuracy training: \n")

print(f"knn score = {knn.score(features_train, target_train)}")
print(f"svc score = {svc.score(features_train, target_train)}")
print(f"random forest score = {r_forest.score(features_train, target_train)}")

print("\n\n")
print("*"*79)
print("Accuracy prediction: \n")

print(f"knn score = {accuracy_score(target_validation, knn_pred)}")
print(f"svc score = {accuracy_score(target_validation, svc_pred)}")
print(f"random forest score = {accuracy_score(target_validation, r_forest_pred)}")

print("\n")
print("*"*79)
print("Confusion matrix: \n")

print("Report knn: \n")
print(confusion_matrix(target_validation, knn_pred))

print("Report svc: \n")
print(confusion_matrix(target_validation, svc_pred))

print("Report random forest: \n")
print(confusion_matrix(target_validation, r_forest_pred))

print("\n")
print("*"*79)
print("Report knn: \n")
print(classification_report(target_validation, knn_pred))

print("Report svc: \n")
print(classification_report(target_validation, svc_pred))

print("Report random forest: \n")
print(classification_report(target_validation, r_forest_pred))

# --------------------------- 3) CROSS VALIDATION ----------------------------
# --------------------------- 3) CROSS VALIDATION ----------------------------
kfold = StratifiedKFold(n_splits=10, random_state = seed, shuffle=True)

cross_val(knn, "knn")
cross_val(svc, "svc")
cross_val(r_forest, "random_forest")


# --------------------------- 3) GRID SEARCH CV ----------------------------
# --------------------------- 3) GRID SEARCH CV ----------------------------
def OptimizeModel(model, param_grid, kfold, name):
    grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=kfold,
    n_jobs=-1,
    )
    
    grid.fit(features_train, target_train)

    best_model = grid.best_estimator_
    print(f"Best parameters for {name}:", grid.best_params_)
    print(f"Best accuracy CV for {name}:", best_model)
    
    val_score = best_model.score(features_validation, target_validation)
    print(f"Accuracy of the best {name} model on validation :", val_score)
    
# -------- RANDOM FOREST GRID & MODEL ----------
param_forest = {
    "n_estimators":    [100, 200],
    "max_depth":       [4, 6, 8],
    "min_samples_leaf":[1, 3, 5],
    "max_features":    ["sqrt", 0.5]
}
r_forest_base = RandomForestClassifier()

# -------- SVC GRID ----------
param_SVC = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", "auto", 0.1, 1],
    "kernel": ["rbf", "poly"],
}

OptimizeModel(r_forest_base, param_forest, kfold, "random forest")
OptimizeModel(svc, param_SVC, kfold, "svc")


    
    