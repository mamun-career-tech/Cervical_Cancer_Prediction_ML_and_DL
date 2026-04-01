"""
Part-by-part Python script converted from `Cervical_Cancer_Voting.ipynb`.

Notes:
- This script keeps the original notebook flow, but organizes it into clear sections.
- A few notebook entries were written in markdown instead of executable code
  (feature dropping, PCA, and SMOTE blocks). Those are preserved here as
  optional sections so nothing important is lost.
- Update `DATA_PATH` before running if your CSV is stored somewhere else.
"""

import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
# PART 1 — CONFIGURATION
# =============================================================================
DATA_PATH = "risk_factors_cervical_cancer.csv"
TEST_SIZE = 0.35
CV_FOLDS = 15

ENCODE_COLUMNS = [
    "Age",
    "Number_of_sexual_partners",
    "First_sexual_intercourse",
    "Num_of_pregnancies",
    "Smokes",
    "Smokes_years",
    "Smokes_packs_per_year",
    "Hormonal_Contraceptives",
    "Hormonal_Contraceptives_years",
    "IUD",
    "IUD_years",
    "STDs",
    "STDs_number",
    "STDs_condylomatosis",
    "STDs_cervical_condylomatosis",
    "STDs_vaginal_condylomatosis",
    "STDs_vulvo_perineal_condylomatosis",
    "STDs_syphilis",
    "STDs_pelvic_inflammatory_disease",
    "STDs_genital_herpes",
    "STDs_molluscum_contagiosum",
    "STDs_AIDS",
    "STDs_HIV",
    "STDs_Hepatitis_B",
    "STDs_HPV",
    "STDs_Number_of_diagnosis",
    "STDs_Time_since_first_diagnosis",
    "STDs_Time_since_last_diagnosis",
    "Dx_Cancer",
    "Dx_CIN",
    "Dx_HPV",
    "Dx",
    "Hinselmann",
    "Schiller",
    "Citology",
    "Biopsy",
]


# =============================================================================
# PART 2 — DATA LOADING
# =============================================================================
def load_dataset(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    pd.options.mode.chained_assignment = None
    return df


# =============================================================================
# PART 3 — PREPROCESSING
# =============================================================================
def label_encode_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    encoder = LabelEncoder()

    for column in ENCODE_COLUMNS:
        if column in df.columns:
            df[column] = encoder.fit_transform(df[column])

    return df


def split_features_target(df: pd.DataFrame):
    x = df.drop(["Biopsy"], axis=1)
    y = df["Biopsy"]
    return x, y


def compute_mutual_information(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    mi = mutual_info_classif(x, y)
    mi = pd.Series(mi, index=x.columns).sort_values(ascending=False)
    return mi


# =============================================================================
# PART 4 — OPTIONAL CLEANING / FEATURE REDUCTION FROM NOTEBOOK MARKDOWN
# =============================================================================
def optional_drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    These lines existed in the notebook as markdown, not executable code.
    Enable this function if you want to actually apply them.
    """
    drop_cols = ["STDs_cervical_condylomatosis", "STDs_AIDS"]
    existing = [col for col in drop_cols if col in df.columns]
    if existing:
        df = df.drop(existing, axis=1)
    return df


def optional_scale_and_pca(x: pd.DataFrame, n_components: int = 4) -> np.ndarray:
    """
    Optional PCA block reconstructed from notebook markdown.
    """
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x)

    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(scaled_x)
    return x_pca


def optional_smote_train_test(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """
    Optional SMOTE block reconstructed from notebook markdown.
    The original notebook showed this idea in markdown only.
    """
    smote = SMOTE()
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
    x_test_smote, y_test_smote = smote.fit_resample(x_test, y_test)
    return x_train_smote, x_test_smote, y_train_smote, y_test_smote


# =============================================================================
# PART 5 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
def chart_function(col: pd.DataFrame) -> None:
    plt.figure(figsize=(20, 15))
    plot_number = 1

    for column in col:
        if plot_number <= 9:
            plt.subplot(3, 3, plot_number)
            sns.histplot(col[column])
            plt.xlabel(column, fontsize=20)
        plot_number += 1

    plt.tight_layout()
    plt.show()


def plot_feature_distributions(df: pd.DataFrame) -> None:
    cols1 = df[
        [
            "Age",
            "Number_of_sexual_partners",
            "First_sexual_intercourse",
            "Num_of_pregnancies",
            "Smokes",
            "Smokes_years",
            "Smokes_packs_per_year",
            "Hormonal_Contraceptives",
            "Hormonal_Contraceptives_years",
        ]
    ]

    cols2 = df[
        [
            "IUD",
            "IUD_years",
            "STDs",
            "STDs_number",
            "STDs_condylomatosis",
            "STDs_cervical_condylomatosis",
            "STDs_vaginal_condylomatosis",
            "STDs_vulvo_perineal_condylomatosis",
            "STDs_syphilis",
        ]
    ]

    cols3 = df[
        [
            "STDs_pelvic_inflammatory_disease",
            "STDs_genital_herpes",
            "STDs_molluscum_contagiosum",
            "STDs_AIDS",
            "STDs_HIV",
            "STDs_Hepatitis_B",
            "STDs_HPV",
            "STDs_Number_of_diagnosis",
            "STDs_Time_since_first_diagnosis",
        ]
    ]

    cols4 = df[
        [
            "STDs_Time_since_last_diagnosis",
            "Dx_Cancer",
            "Dx_CIN",
            "Dx_HPV",
            "Dx",
            "Hinselmann",
            "Schiller",
            "Citology",
        ]
    ]

    chart_function(cols1)
    chart_function(cols2)
    chart_function(cols3)
    chart_function(cols4)


def plot_biopsy_distribution(df: pd.DataFrame, title: str = "Biopsy Distribution") -> None:
    plt.figure(figsize=(4, 4))
    names = ["0", "1"]
    count = [(df.Biopsy.values == 0).sum(), (df.Biopsy.values == 1).sum()]
    plt.bar(names, count, color=["green", "red"])
    plt.xlabel("Biopsy", fontsize=20)
    plt.ylabel("Count", fontsize=20)
    plt.ylim(0, 1000)
    plt.title(title, fontsize=16)
    for i in range(len(names)):
        plt.text(i, count[i], count[i], ha="center", va="bottom", fontsize=15)
    plt.show()


def plot_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(35, 15))
    sns.heatmap(df.corr(), annot=True, cmap="BuPu")
    plt.show()


# =============================================================================
# PART 6 — TRAIN / TEST SPLIT
# =============================================================================
def create_train_test_split(x: pd.DataFrame, y: pd.Series):
    return train_test_split(x, y, test_size=TEST_SIZE)


# =============================================================================
# PART 7 — MACHINE LEARNING MODELS WITHOUT OPTIMIZATION
# =============================================================================
def train_baseline_models(x_train: pd.DataFrame, y_train: pd.Series):
    models = {
        "SVM": SVC().fit(x_train, y_train),
        "RF": RandomForestClassifier().fit(x_train, y_train),
        "KNN": KNeighborsClassifier().fit(x_train, y_train),
        "DT": DecisionTreeClassifier().fit(x_train, y_train),
        "NB": GaussianNB().fit(x_train, y_train),
        "LR": LogisticRegression(solver="liblinear", max_iter=3000).fit(x_train, y_train),
        "AB": AdaBoostClassifier().fit(x_train, y_train),
        "GB": GradientBoostingClassifier().fit(x_train, y_train),
        "MLP": MLPClassifier().fit(x_train, y_train),
        "NCC": NearestCentroid().fit(x_train, y_train),
    }

    voting_estimators = [("svm", models["SVM"]), ("lr", models["LR"])]
    models["VC"] = VotingClassifier(estimators=voting_estimators, voting="hard").fit(
        x_train, y_train
    )
    return models


def evaluate_holdout_models(models, x_test: pd.DataFrame, y_test: pd.Series):
    predictions = {}
    scores = {}

    for model_name, model in models.items():
        y_pred = model.predict(x_test)
        score = model.score(x_test, y_test) * 100
        predictions[model_name] = y_pred
        scores[model_name] = f"{score:.2f}"

    return predictions, scores


def plot_accuracy(scores: dict, title: str) -> None:
    algo = list(scores.keys())
    pert = list(scores.values())
    acc = [math.floor(int(float(value))) for value in pert]

    plt.figure(figsize=(15, 10))
    plt.bar(algo, acc, width=0.6)
    plt.title(title, color="black", fontsize=25)
    plt.xlabel("Classifier", fontsize=18)
    plt.ylabel("Accuracy (%)", fontsize=18)
    plt.ylim(0, 110)

    for i in range(len(algo)):
        plt.text(i, acc[i], pert[i], ha="center", va="bottom", fontsize=16)

    plt.show()


# =============================================================================
# PART 8 — RANDOM OVERSAMPLING
# =============================================================================
def apply_random_oversampling(x: pd.DataFrame, y: pd.Series):
    ros = RandomOverSampler(sampling_strategy=1)
    x_os, y_os = ros.fit_resample(x, y)
    return x_os, y_os


def plot_resampled_distribution(y_resampled: pd.Series) -> None:
    plt.figure(figsize=(4, 4))
    names = ["0", "1"]
    count = [(y_resampled.values == 0).sum(), (y_resampled.values == 1).sum()]
    plt.bar(names, count, color=["green", "red"])
    plt.xlabel("Biopsy", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.ylim(0, 1000)
    for i in range(len(names)):
        plt.text(i, count[i], count[i], ha="center", va="bottom", fontsize=16)
    plt.show()


# =============================================================================
# PART 9 — MACHINE LEARNING MODELS WITH OPTIMIZATION / CV
# =============================================================================
def train_optimized_models_cv(x_os: pd.DataFrame, y_os: pd.Series, cv_folds: int = CV_FOLDS):
    cv_predictions = {}
    cv_scores = {}
    fitted_model_defs = {}

    fitted_model_defs["SVM"] = SVC(C=1000, gamma=0.0001, kernel="rbf")
    fitted_model_defs["RF"] = RandomForestClassifier()
    fitted_model_defs["KNN"] = KNeighborsClassifier(n_neighbors=1, weights="uniform")
    fitted_model_defs["DT"] = DecisionTreeClassifier(
        criterion="gini", max_depth=20, min_samples_leaf=5
    )
    fitted_model_defs["NB"] = GaussianNB()
    fitted_model_defs["LR"] = LogisticRegression(
        max_iter=3000, C=1.0, penalty="l2", solver="liblinear"
    )
    fitted_model_defs["AB"] = AdaBoostClassifier()
    fitted_model_defs["GB"] = GradientBoostingClassifier()
    fitted_model_defs["MLP"] = MLPClassifier(max_iter=3000)
    fitted_model_defs["NCC"] = NearestCentroid()

    for model_name, model in fitted_model_defs.items():
        use_cv = 5 if model_name in {"NB", "LR"} else cv_folds
        y_pred = cross_val_predict(model, x_os, y_os, cv=use_cv)
        score = accuracy_score(y_os, y_pred) * 100

        cv_predictions[model_name] = y_pred
        cv_scores[model_name] = f"{score:.2f}"

    return fitted_model_defs, cv_predictions, cv_scores


# =============================================================================
# PART 10 — VOTING CLASSIFIER
# =============================================================================
def voting_classifier_method(estimators, vote_type: str, x_os: pd.DataFrame, y_os: pd.Series):
    vot = VotingClassifier(estimators=estimators, voting=vote_type)
    vc_y_pred = cross_val_predict(vot, x_os, y_os, cv=CV_FOLDS)
    vc_score = accuracy_score(y_os, vc_y_pred) * 100
    return f"{vc_score:.2f}", vc_y_pred


def evaluate_voting_classifiers(model_defs: dict, x_os: pd.DataFrame, y_os: pd.Series):
    est1 = [("gb", model_defs["GB"]), ("rf", model_defs["RF"])]
    est2 = [("gb", model_defs["GB"]), ("mlp", model_defs["MLP"])]
    est3 = [("rf", model_defs["RF"]), ("mlp", model_defs["MLP"])]
    est4 = [("gb", model_defs["GB"]), ("rf", model_defs["RF"]), ("mlp", model_defs["MLP"])]

    voting_scores = {}
    voting_predictions = {}

    for label, estimators in {
        "VC-hard(GB+RF)": (est1, "hard"),
        "VC-soft(GB+RF)": (est1, "soft"),
        "VC-hard(GB+MLP)": (est2, "hard"),
        "VC-soft(GB+MLP)": (est2, "soft"),
        "VC-hard(RF+MLP)": (est3, "hard"),
        "VC-soft(RF+MLP)": (est3, "soft"),
        "VC-hard(GB+RF+MLP)": (est4, "hard"),
        "VC-soft(GB+RF+MLP)": (est4, "soft"),
    }.items():
        score, y_pred = voting_classifier_method(
            estimators[0], estimators[1], x_os, y_os
        )
        voting_scores[label] = score
        voting_predictions[label] = y_pred

    return voting_scores, voting_predictions


def plot_voting_accuracy(voting_scores: dict) -> None:
    algo = list(voting_scores.keys())
    pert = list(voting_scores.values())
    acc = [math.floor(int(float(value))) for value in pert]

    plt.figure(figsize=(20, 10))
    plt.bar(algo, acc, width=0.6)
    plt.title("Voting", color="black", fontsize=25)
    plt.xlabel("Voting Type & Classifier", fontsize=18)
    plt.ylabel("Accuracy (%)", fontsize=18)
    plt.ylim(0, 110)

    for i in range(len(algo)):
        plt.text(i, acc[i], pert[i], ha="center", va="bottom", fontsize=16)

    plt.show()


# =============================================================================
# PART 11 — CONFUSION MATRIX & CLASSIFICATION REPORT
# =============================================================================
def print_confusion_matrix(
    cm: np.ndarray,
    class_names,
    figsize=(7, 4),
    fontsize=14,
    title="Confusion Matrix",
) -> None:
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="cividis")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=90, ha="right", fontsize=fontsize
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=0, ha="center", fontsize=fontsize
    )
    plt.ylabel("Actual", fontsize=18, color="blue")
    plt.xlabel("Prediction", fontsize=18, color="blue")
    plt.title(title, fontsize=22, color="blue")
    plt.show()


def show_reports(y_true: pd.Series, predictions: dict) -> None:
    for model_name, y_pred in predictions.items():
        cm = confusion_matrix(y_true, y_pred)
        print_confusion_matrix(cm, ["0", "1"], title=model_name)
        report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))
        print(f"\nClassification Report: {model_name}")
        print(report_df)


# =============================================================================
# PART 12 — MAIN PIPELINE
# =============================================================================
def main():
    # Load data
    df = load_dataset(DATA_PATH)

    # Encode data
    df = label_encode_dataset(df)

    # Optional notebook-markdown feature drops:
    # df = optional_drop_columns(df)

    # Split features / target
    x, y = split_features_target(df)

    # Mutual information
    mutual_info = compute_mutual_information(x, y)
    print("\nTop Mutual Information Features:")
    print(mutual_info.head(10))

    # EDA
    plot_feature_distributions(df)
    plot_biopsy_distribution(df, title="Biopsy Distribution (Original)")
    plot_heatmap(df)

    # Optional PCA reconstructed from notebook markdown:
    # x = optional_scale_and_pca(x, n_components=4)

    # Train / test split
    x_train, x_test, y_train, y_test = create_train_test_split(x, y)

    # Optional SMOTE reconstructed from notebook markdown:
    # x_train, x_test, y_train, y_test = optional_smote_train_test(
    #     x_train, x_test, y_train, y_test
    # )

    # Baseline models
    baseline_models = train_baseline_models(x_train, y_train)
    _, baseline_scores = evaluate_holdout_models(baseline_models, x_test, y_test)
    print("\nAccuracy without Optimization:")
    print(baseline_scores)
    plot_accuracy({k: v for k, v in baseline_scores.items() if k != "VC"}, "Accuracy without Optimization")

    # Random oversampling
    x_os, y_os = apply_random_oversampling(x, y)
    print("\nClass balance after RandomOverSampler:")
    print(Counter(y_os))
    plot_resampled_distribution(y_os)

    # Optimized models with cross-validation
    model_defs, cv_predictions, cv_scores = train_optimized_models_cv(x_os, y_os, cv_folds=CV_FOLDS)
    print("\nAccuracy with Optimization:")
    print(cv_scores)
    plot_accuracy(cv_scores, "Accuracy with Optimization")

    # Voting classifier
    voting_scores, voting_predictions = evaluate_voting_classifiers(model_defs, x_os, y_os)
    print("\nVoting Classifier Scores:")
    print(voting_scores)
    print(f"Max Voting Accuracy = {max(map(float, voting_scores.values())):.2f}")
    plot_voting_accuracy(voting_scores)

    # Reports for optimized models
    show_reports(y_os, cv_predictions)

    # Report for best voting model
    if voting_scores:
        best_voting_model = max(voting_scores, key=lambda k: float(voting_scores[k]))
        print(f"\nBest Voting Classifier: {best_voting_model} -> {voting_scores[best_voting_model]}")
        show_reports(y_os, {best_voting_model: voting_predictions[best_voting_model]})


if __name__ == "__main__":
    main()
