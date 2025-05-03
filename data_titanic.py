import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

output_dir = "plots"
pd.set_option('display.max_columns', None)
seed = 1

# ---------------- SAVE PLOT FUNCTION ----------------
def save_plot(name):
    plt.savefig(os.path.join(output_dir, name))
    plt.close()

# -------------------------------- 1) IMPORT DATA --------------------------------
# -------------------------------- 1) IMPORT DATA --------------------------------
X, Y = fetch_openml("titanic", version=1, as_frame=True,
return_X_y=True)
print('---------------- BASE DATA ----------------')
print(X.head(10))
columns_int = ['parch', 'sibsp', 'age', 'fare' ]
columns_category = ['sex', 'pclass']

# -------------------------------- 2) CLEAN DATA --------------------------------
# -------------------------------- 2) CLEAN DATA --------------------------------

# 1) DROP USELESS & INCOMPLETE COLUMNS
X.drop(['boat', 'cabin', 'name', 'embarked', 'body', 'ticket', 'home.dest'], axis=1, inplace=True)

dataset = X.copy(deep=True)
dataset["survived"] = Y

# 2) REPLACE NA BY MEAN
for i in columns_int:
    dataset[i] = dataset[i].fillna(
        dataset[i].mean()
    )

# 3) REPLACE NA BY PREVIOUS VALUE
for i in columns_category:
    dataset[i] = dataset[i].fillna(method='bfill')
    print(dataset.isnull().sum())
    
print('---------------- CLEAN DATA ----------------')
print(dataset.head())

# 4) REPLACE STRING CATEGORIES BY INT
dataset = pd.get_dummies(
    dataset,
    columns=['sex'],
    drop_first=True
)

# 5) STANDARD SCALING FOR THE TRAIN & VALIDATION DATA
features = dataset.drop(['survived'], axis=1)
target   = dataset["survived"]

features_train, features_validation, target_train, target_validation = train_test_split(
    features, target, test_size=0.30, random_state=seed, shuffle=True
)

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_validation = scaler.transform(features_validation)
features_train = pd.DataFrame(
    features_train,
    columns=features.columns
)
features_validation = pd.DataFrame(
    features_validation,
    columns=features.columns
)

# ---------------------------- 3) VISUALIZE & ANALYSE ------------------------------
# ---------------------------- 3) VISUALIZE & ANALYSE ------------------------------
print('---------------- FEATURES DATA ----------------')
print(features.head(5))
print('---------------- TARGET DATA ----------------')
print(target.head(5))

# ----------------- BOX PLOT ----------------
plt.figure()
dataset.plot(kind='box', subplots=True, sharex=False, sharey=False)
save_plot("boxplots.png")

# ----------------- HEATMAP -----------------
corr = dataset.corr(numeric_only=True)
plt.subplots(figsize=(15,10))
sns.heatmap(
    corr,
    xticklabels=corr.columns,
    yticklabels=corr.columns,
    annot=True,
    cmap=sns.diverging_palette(220, 20, as_cmap=True)
)
save_plot("heatmap.png")

# ---------------- PAIRPLOT -----------------
sns.pairplot(dataset, hue="survived")
save_plot("pairplot.png")
