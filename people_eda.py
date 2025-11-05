# ==========================================
# PEOPLE DATA ANALYSIS & MACHINE LEARNING PIPELINE
# ==========================================
# Loads, cleans, explores, clusters, and predicts from a large people dataset.
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import dedent
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# -----------------------------------
# 🧩 STEP 1 — LOAD DATA
# -----------------------------------
df = pd.read_csv("AgeDataset-V1.csv")
print("✅ Data loaded successfully!")
print(df.head())
print("\nData shape:", df.shape)

# -----------------------------------
# 🔤 STEP 1.1 — STANDARDIZE COLUMN NAMES
# -----------------------------------
# Map any original names to clean snake_case (robust to spaces/case)
def to_snake(s: str) -> str:
    return (
        s.strip()
         .lower()
         .replace("/", " ")
         .replace("-", " ")
         .replace(".", "")
         .replace("(", "")
         .replace(")", "")
         .replace("  ", " ")
         .replace(" ", "_")
    )

df.columns = [to_snake(c) for c in df.columns]

# If the file contains already-computed age_of_death as "age_of_death" or "age_of_death"/"age_of_death"
# the normalization above will already set it to "age_of_death".
# For reference, expected canonical columns now:
# id, name, short_description, gender, country, occupation,
# birth_year, death_year, manner_of_death, age_of_death

print("\n🧾 Columns after normalization:\n", df.columns.tolist())

# -----------------------------------
# 🧹 STEP 2 — CLEAN & PREPARE DATA
# -----------------------------------
print("\n--- Cleaning and preparing data ---\n")

# Replace obvious blanks with NaN
df = df.replace(["", " ", "NaN", "nan", "None"], np.nan)

# Standardize text columns (values) — keep as lowercase trimmed strings
text_cols = df.select_dtypes(include="object").columns.tolist()
for col in text_cols:
    df[col] = df[col].astype("string").str.strip().str.lower()

# Coerce numeric columns when they look like years/ages
for col in df.columns:
    if "year" in col or "age" in col:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Compute age_of_death if missing and birth/death available
if "age_of_death" not in df.columns:
    df["age_of_death"] = pd.NA
if "birth_year" in df.columns and "death_year" in df.columns:
    mask = df["birth_year"].notna() & df["death_year"].notna()
    df.loc[mask, "age_of_death"] = (df.loc[mask, "death_year"] - df.loc[mask, "birth_year"]).astype("Int64")

# Remove duplicates
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"Removed {before - after} duplicate rows.\n")

# Quick stats
numeric_preview_cols = [c for c in ["birth_year", "death_year", "age_of_death"] if c in df.columns]
if numeric_preview_cols:
    print("Summary statistics (key numeric columns):\n", df[numeric_preview_cols].describe(), "\n")

# Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)
print("✅ Cleaned dataset saved as 'cleaned_dataset.csv'")

# -----------------------------------
# 🎨 MINIMAL, CLEAN VISUALS
# -----------------------------------
print("\n--- Minimal visuals ---\n")

# Age of Death distribution
if "age_of_death" in df.columns and df["age_of_death"].notna().any():
    plt.figure(figsize=(7,4))
    sns.histplot(df["age_of_death"].dropna(), bins=25, edgecolor="black")
    plt.title("Age of Death Distribution", fontsize=13)
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("clean_age_distribution.png")
    plt.close()
    print("📊 Saved 'clean_age_distribution.png'")

# Top 10 occupations
if "occupation" in df.columns and df["occupation"].notna().any():
    top_occ = df["occupation"].value_counts().head(10)
    plt.figure(figsize=(8,4))
    sns.barplot(y=top_occ.index, x=top_occ.values)
    plt.title("Top 10 Occupations", fontsize=13)
    plt.xlabel("Count")
    plt.ylabel("Occupation")
    plt.tight_layout()
    plt.savefig("clean_top_occupations.png")
    plt.close()
    print("📊 Saved 'clean_top_occupations.png'")

print("\n✅ Data cleaning and preparation completed!")

# -----------------------------------
# 📊 STEP 3 — EXPLORATORY DATA ANALYSIS (TEXT-FOCUSED)
# -----------------------------------
print("\n--- EXPLORATORY DATA ANALYSIS ---\n")
print(f"Dataset shape: {df.shape}")
missing = df.isnull().sum()
print("\nMissing values (top 10):\n", missing.sort_values(ascending=False).head(10))

if "age_of_death" in df.columns:
    print("\nAge of Death (describe):\n", df["age_of_death"].describe())

# -----------------------------------
# 🧾 STEP 4 — AUTO SUMMARY REPORT
# -----------------------------------
summary_lines = []
summary_lines.append(f"Total entries: {len(df)}")
summary_lines.append(f"Columns: {', '.join(df.columns)}")
if "age_of_death" in df.columns:
    try:
        avg_age = float(pd.to_numeric(df["age_of_death"], errors="coerce").mean())
        summary_lines.append(f"Average age of death: {avg_age:.2f}")
    except Exception:
        pass
if "gender" in df.columns:
    gender_counts = df["gender"].value_counts(dropna=True).to_dict()
    summary_lines.append(f"Gender distribution (non-null): {gender_counts}")

report_text = dedent(f"""
===============================
📊 PEOPLE DATASET ANALYSIS SUMMARY
===============================

{chr(10).join(summary_lines)}

Charts generated (if available):
- clean_age_distribution.png
- clean_top_occupations.png

✅ Ready for presentation!
""")
with open("analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
print("\n✅ Summary report saved as 'analysis_summary.txt'")

# -----------------------------------
# 🤖 STEP 5 — MACHINE LEARNING: CLUSTERING
# -----------------------------------
print("\n--- MACHINE LEARNING: CLUSTERING ---\n")

# Select features: numeric + selected categoricals if present
feature_candidates = []
feature_candidates += [c for c in ["birth_year", "death_year", "age_of_death"] if c in df.columns]
feature_candidates += [c for c in ["gender", "occupation", "country", "manner_of_death"] if c in df.columns]
# Keep existing and non-empty
features = [c for c in feature_candidates if c in df.columns]

if not features:
    print("⚠️ No suitable features found for clustering. Skipping.")
else:
    ml_df = df[features].copy()

    # Fill categorical NaNs with 'unknown' then label-encode
    for col in ml_df.select_dtypes(include=["object", "string"]).columns:
        ml_df[col] = ml_df[col].fillna("unknown").astype(str)
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col])

    # Numeric: fill NaN with column means
    ml_df = ml_df.fillna(ml_df.mean(numeric_only=True))

    # Scale
    scaler = StandardScaler()
    scaled = scaler.fit_transform(ml_df)

    # KMeans
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaled)
    print(f"✅ K-Means clustering complete! Created {k} clusters.")

    # Quick numeric summary per cluster
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        print("\nCluster Summary (numeric means):\n", df.groupby("cluster")[num_cols].mean())

    # Optional 2D view if birth_year & age_of_death exist
    if "birth_year" in df.columns and "age_of_death" in df.columns:
        plt.figure(figsize=(8,5))
        sns.scatterplot(data=df.sample(min(50000, len(df)), random_state=42),  # sample for speed
                        x="birth_year", y="age_of_death", hue="cluster", palette="tab10", s=10, linewidth=0)
        plt.title("K-Means Clusters by Birth Year & Age of Death")
        plt.tight_layout()
        plt.savefig("clusters_birth_vs_age.png")
        plt.close()
        print("📊 Saved 'clusters_birth_vs_age.png'")

    df.to_csv("clustered_dataset.csv", index=False)
    print("✅ Clustered dataset saved as 'clustered_dataset.csv'")

# -----------------------------------
# 🧠 STEP 6 — PREDICTIVE MODELING
# -----------------------------------
print("\n--- MACHINE LEARNING: PREDICTION ---\n")

# Choose the target — make sure it exists; options: "manner_of_death" or "age_of_death"
# If your dataset lacks 'manner_of_death' values for most rows, you may prefer 'age_of_death' (regression).
target = "manner_of_death"  # or: target = "age_of_death"

if target not in df.columns:
    print(f"⚠️ Column '{target}' not found. Available columns:")
    print(df.columns.tolist())
else:
    # Exclude leakage/ID-like columns
    exclude = {target, "cluster", "id", "wikidata_id", "name", "full_name", "short_description"}
    X_cols = [c for c in df.columns if c not in exclude]

    # Build X, y
    X = df[X_cols].copy()
    y = df[target].copy()

    # Handle categoricals
    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].fillna("unknown").astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Fill numerics
    X = X.fillna(X.mean(numeric_only=True))

    # If target is categorical, encode
    is_regression = target == "age_of_death"
    if not is_regression:
        y = y.fillna("unknown").astype(str)
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

    # Train/test split
    # For massive datasets, a smaller test size speeds up training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=None
    )


    # Model
    model = RandomForestRegressor(random_state=42) if is_regression else RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model training complete.")

    # Evaluate
    if is_regression:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"\n🔍 Regression results:\nMAE = {mae:.2f}\nR²  = {r2:.2f}")
    else:
        y_pred = model.predict(X_test)
        print("\n🔍 Classification Report:\n")
        print(classification_report(y_test, y_pred))

    # Feature importance
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:10]
    plt.figure(figsize=(8,5))
    plt.barh(np.array(X.columns)[order][::-1], importances[order][::-1])
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    print("📊 Saved 'feature_importance.png'")

    print("\n✅ Predictive modeling finished!")
