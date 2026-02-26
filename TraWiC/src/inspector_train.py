import os
import pickle
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, train_test_split


class Colors:
    GREEN = "\033[92m"  # GREEN
    YELLOW = "\033[93m"  # YELLOW
    BLUE = "\033[94m"  # BLUE
    END = "\033[0m"  # reset to the default color


arg_parse = ArgumentParser()
arg_parse.add_argument(
    "--input_dir",
    type=str,
    default=None,
    help="Directory containing train.csv (defaults to rf_data/syn{syn}_sem{sem})",
)
arg_parse.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to save model and plots (defaults to current directory)",
)
arg_parse.add_argument(
    "--syntactic_threshold",
    type=int,
    default=100,
)
arg_parse.add_argument(
    "--semantic_threshold",
    type=int,
    default=20,
)
arg_parse.add_argument(
    "--visualisation",
    type=bool,
    default=True,
)
args = arg_parse.parse_args()

# Determine input directory
if args.input_dir:
    train_csv_path = os.path.join(args.input_dir, "train.csv")
else:
    train_csv_path = os.path.join(
        os.getcwd(),
        "rf_data",
        f"syn{args.syntactic_threshold}_sem{args.semantic_threshold}",
        "train.csv",
    )

# Determine output directory
if args.output_dir:
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
else:
    output_dir = os.getcwd()

print(f"Reading training data from: {train_csv_path}")
print(f"Saving outputs to: {output_dir}")

combined_ds = pd.read_csv(train_csv_path)

# Split the dataset into training and testing datasets
train_ds, test_ds = train_test_split(
    combined_ds,
    test_size=0.2,
    random_state=42,
    stratify=combined_ds["trained_on"],
)

# drop the index column
train_ds.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
test_ds.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')

# split the training and testing datasets into x and y (skip first column which is filename)
x, y = train_ds.iloc[:, 1:-1].values, train_ds.iloc[:, -1].values
print(f"Features shape: {x.shape}")
print(f"Target shape: {y.shape}")
print(f"Features Snippet: {x[:1]}")
print(f"Target Snippet: {y[:1]}")

# classifier: Random Forest
clf = RandomForestClassifier()
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_features": ["sqrt", "log2"],
    "max_depth": [10, 20, 30],
    "criterion": ["gini", "entropy"],
}

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring="f1",
)
grid_search.fit(x, y)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")
clf = grid_search.best_estimator_

print(
    "Number of 1s and 0s in the train dataset:", train_ds["trained_on"].value_counts()
)
print("Number of 1s and 0s in the test dataset:", test_ds["trained_on"].value_counts())

# create a confusion matrix and print it
tn, fp, fn, tp = confusion_matrix(
    test_ds.iloc[:, -1].values,
    clf.predict(test_ds.iloc[:, 1:-1].values),
).ravel()

print(
    f"{Colors.GREEN}True Negatives:{Colors.END} {Colors.YELLOW}{tn/(tn+fp+fn+tp)}{Colors.END}",
    f"{Colors.BLUE}False Positives:{Colors.END} {Colors.YELLOW}{fp/(tn+fp+fn+tp)}{Colors.END}",
    f"{Colors.BLUE}False Negatives:{Colors.END} {Colors.YELLOW}{fn/(tn+fp+fn+tp)}{Colors.END}",
    f"{Colors.GREEN}True Positives:{Colors.END} {Colors.YELLOW}{tp/(tn+fp+fn+tp)}{Colors.END}",
)
# print the accuracy
accuracy = accuracy_score(
    test_ds.iloc[:, -1].values,
    clf.predict(test_ds.iloc[:, 1:-1].values),
)
# calcualte the precision and recall
precision, recall, fscore, _ = precision_recall_fscore_support(
    test_ds.iloc[:, -1].values,
    clf.predict(test_ds.iloc[:, 1:-1].values),
    average="weighted",
)

print(f"{Colors.GREEN}Precision:{Colors.END} {Colors.YELLOW}{precision}{Colors.END}")
print(f"{Colors.GREEN}Accuracy:{Colors.END} {Colors.YELLOW}{accuracy}{Colors.END}")
print(f"{Colors.GREEN}F-score:{Colors.END} {Colors.YELLOW}{fscore}{Colors.END}")
print(f"{Colors.GREEN}Sensitivity:{Colors.END} {Colors.YELLOW}{recall}{Colors.END}")
print(
    f"{Colors.GREEN}Specificity:{Colors.END} {Colors.YELLOW}{tn / (tn + fp)}{Colors.END}"
)

pickle.dump(
    clf,
    open(
        os.path.join(
            output_dir,
            f"rf_model__syn{args.syntactic_threshold}_sem{args.semantic_threshold}.sav",
        ),
        "wb",
    ),
)

if args.visualisation:
    sns.set_theme(style="dark")

    fig, ax = plt.subplots(figsize=(12, 12))

    # Adjust font sizes
    ax.tick_params(labelsize=27)
    ax.set_xlabel("Importance", fontsize=20, fontdict={"weight": "bold"})
    # ax.set_ylabel("Features", fontsize=10)

    # Horizontal bar chart with feature importances
    ax.barh(train_ds.columns[1:-1], clf.feature_importances_)

    # Rotate y-axis labels to fit
    plt.yticks(rotation=0)
    plt.tight_layout()
    # Save the figure with a descriptive filename
    plt.savefig(
        os.path.join(
            output_dir,
            f"feature_importance__syn{args.syntactic_threshold}_sem{args.semantic_threshold}.png",
        ),
        dpi=300,
    )
    # calcualte the distriutino of each feature
    # Use only numeric columns (exclude filename)
    train_ds_numeric = train_ds.select_dtypes(include=['number'])
    train_ds_numeric.hist(figsize=(20, 20))
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f"feature_distribution__syn{args.syntactic_threshold}_sem{args.semantic_threshold}.png",
        ),
        dpi=300,
    )
    #### Correlation Matrix ####
    fig, ax = plt.subplots(figsize=(12, 12))
    # Create the heatmap, ensuring square cells and other configurations
    sns.heatmap(
        train_ds_numeric.corr(method="spearman"), annot=True, fmt=".2f", ax=ax, square=True
    )
    # Adjusting the Y-axis limit
    ax.set_ylim(len(train_ds_numeric.columns), 0)
    ax.tick_params(labelsize=12)
    plt.tight_layout(pad=2)
    plt.savefig(
        os.path.join(
            output_dir,
            f"correlation_matrix__syn{args.syntactic_threshold}_sem{args.semantic_threshold}.png",
        ),
        dpi=300,
    )
