import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ==============================================================================
# STEP 1: LOAD & CONTEXT-AWARE DATA CLEANING
# ==============================================================================
# Load the dataset from your local folder
file_path = r"c:\Users\sumuk\OneDrive\Desktop\coding\student_placement_500_analysis.csv"
df = pd.read_csv(file_path)

# Drop 'Student_ID' because unique IDs have zero predictive pattern for ML models
df = df.drop(columns=["Student_ID"])

# Remove exact duplicate rows (STU021, STU146, STU321 appeared twice in the raw file)
df = df.drop_duplicates().reset_index(drop=True)

# Drop rows where 2 or more values are missing (e.g., if both CGPA and a score are missing)
df = df[df.isnull().sum(axis=1) < 2].reset_index(drop=True)

# If CGPA is missing, fill it with the mean CGPA of students in the same Department & Placement outcome
df["CGPA"] = df["CGPA"].fillna(
    df.groupby(["Department", "Placed"])["CGPA"].transform("mean")
)

# Group students into 3 CGPA performance tiers so missing scores match similar peers
cgpa_tier = pd.cut(df["CGPA"], bins=[0, 7.0, 8.0, 10.1], labels=["Low", "Mid", "High"])

# Fill missing continuous scores using the mean of peers in the same CGPA tier & Placement status
for col in ["Attendance", "Technical_Score", "Communication_Score"]:
    df[col] = df[col].fillna(
        df.groupby([cgpa_tier, "Placed"], observed=False)[col].transform("mean")
    )

# Fill missing discrete count (Internships) using the peer-group median (keeps whole numbers)
df["Internships"] = df["Internships"].fillna(
    df.groupby([cgpa_tier, "Placed"], observed=False)["Internships"].transform("median")
)

print(f"Data cleaned successfully! Total clean records: {len(df)}\n")

# ==============================================================================
# STEP 2: EXPLORATORY VISUALIZATIONS (2 CHARTS)
# ==============================================================================
# Note: Close each chart window to move to the next chart and then to model training.

# --- Visualization 1: Feature Correlation Heatmap ---
df_corr = df.copy()
# Convert 'Yes'/'No' to 1/0 temporarily so we can measure correlation with placement
df_corr["Placed_Num"] = df_corr["Placed"].map({"Yes": 1, "No": 0})
num_cols = [
    "CGPA", "Attendance", "Backlogs", "Internships", "Projects",
    "Aptitude_Score", "Technical_Score", "Communication_Score", "Placed_Num"
]

plt.figure(figsize=(10, 7))
sns.heatmap(df_corr[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Feature Correlation Heatmap (vs. Placement Status)", fontsize=13)
plt.tight_layout()
plt.show()

# --- Visualization 2: CGPA vs. Technical Score Scatter Plot by Placement ---
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="CGPA", y="Technical_Score", hue="Placed",
    palette={"Yes": "#2ecc71", "No": "#e74c3c"}, alpha=0.8
)
plt.title("CGPA vs. Technical Score by Placement Status", fontsize=13)
plt.xlabel("CGPA")
plt.ylabel("Technical Score")
plt.tight_layout()
plt.show()

# ==============================================================================
# STEP 3: PREPARE FEATURES & TARGET FOR MACHINE LEARNING
# ==============================================================================
# Convert text 'Department' column into binary (0/1) columns using One-Hot Encoding
X = pd.get_dummies(df.drop(columns=["Placed"]), columns=["Department"], drop_first=False)

# Convert target 'Placed' from 'Yes'/'No' into 1/0 for scikit-learn models
y = df["Placed"].map({"Yes": 1, "No": 0})

# Split data: 80% for training the models, 20% for testing accuracy on unseen students
# stratify=y keeps the exact same Yes/No placement ratio in both training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (mean=0, std=1) so large numbers (Attendance ~85) don't dominate small ones (CGPA ~7.5)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# STEP 4: TRAIN & COMPARE MULTIPLE ALGORITHMS
# ==============================================================================
# Logistic Regression & KNN run cleanly without Windows DLL blocks and scale monotonically with scores
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=7),
}

best_model_name = None
best_accuracy = 0.0
best_model = None

print("--- Model Accuracy Comparison ---")
for name, model in models.items():
    # Train the model on scaled training data
    model.fit(X_train_scaled, y_train)
    # Predict outcomes on unseen test data
    predictions = model.predict(X_test_scaled)
    # Calculate accuracy percentage
    acc = accuracy_score(y_test, predictions)
    print(f"{name}: {acc * 100:.2f}%")

    # Automatically select whichever model gets the highest accuracy
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_model = model

print(f"\nSelected Best Model: {best_model_name} ({best_accuracy * 100:.2f}% accuracy)\n")

# ==============================================================================
# STEP 5: COMPUTE PLACED STUDENT BENCHMARKS FOR RECOMMENDATIONS
# ==============================================================================
# Calculate the average metrics of PLACED students in our dataset as target thresholds
placed_students = df[df["Placed"] == "Yes"]
benchmarks = {
    "CGPA": round(placed_students["CGPA"].mean(), 2),
    "Attendance": round(placed_students["Attendance"].mean(), 1),
    "Internships": round(placed_students["Internships"].mean(), 1),
    "Projects": round(placed_students["Projects"].mean(), 1),
    "Aptitude_Score": round(placed_students["Aptitude_Score"].mean(), 1),
    "Technical_Score": round(placed_students["Technical_Score"].mean(), 1),
    "Communication_Score": round(placed_students["Communication_Score"].mean(), 1),
}

# ==============================================================================
# STEP 6: INTERACTIVE PREDICTION & PERSONALIZED RECOMMENDATIONS
# ==============================================================================
def predict_student_placement():
    print("=== Student Placement Predictor ===")
    name = input("Enter Student Name: ").strip()
    dept = input("Enter Department (CSE / IT / AIDS / ECE / EEE / MECH): ").strip().upper()
    cgpa = float(input("Enter CGPA (0 - 10): "))
    attendance = float(input("Enter Attendance Percentage (0 - 100): "))
    backlogs = int(input("Enter Number of Active Backlogs: "))
    internships = float(input("Enter Number of Internships Completed: "))
    projects = int(input("Enter Number of Projects Completed: "))
    aptitude = float(input("Enter Aptitude Score (0 - 100): "))
    technical = float(input("Enter Technical Score (0 - 100): "))
    communication = float(input("Enter Communication Score (0 - 100): "))

    # Build a single-row DataFrame matching the structure of our training features
    student_data = pd.DataFrame([{
        "CGPA": cgpa,
        "Attendance": attendance,
        "Backlogs": backlogs,
        "Internships": internships,
        "Projects": projects,
        "Aptitude_Score": aptitude,
        "Technical_Score": technical,
        "Communication_Score": communication,
        "Department": dept
    }])

    # One-hot encode Department and align columns with X_train (fills unknown departments with 0)
    student_encoded = pd.get_dummies(student_data, columns=["Department"])
    student_encoded = student_encoded.reindex(columns=X.columns, fill_value=0)

    # Scale the user's inputs using the exact same scaler fitted on training data
    student_input = scaler.transform(student_encoded)

    # Predict placement outcome (1 = Yes, 0 = No) and probability percentage
    prediction = best_model.predict(student_input)[0]
    probability = best_model.predict_proba(student_input)[0][1] * 100

    # Apply real-world recruiter eligibility rules (CGPA >= 6.0 and 0 active backlogs)
    if cgpa < 6.0 or backlogs > 0:
        prediction = 0
        probability = min(probability, 25.0)

    print("\n" + "=" * 45)
    print(f"Placement Report for {name} ({dept})")
    print(f"Estimated Placement Probability: {probability:.1f}%")
    if prediction == 1:
        print("Result: LIKELY TO BE PLACED")
    else:
        print("Result: UNLIKELY TO BE PLACED (Needs Improvement)")
    print("=" * 45)

    # Compare the student's metrics against the average benchmarks of placed students
    tips = []
    if backlogs > 0:
        tips.append(f"- Clear your {backlogs} active backlog(s) immediately; many recruiters filter out active backlogs.")
    if cgpa < benchmarks["CGPA"]:
        tips.append(f"- Raise your CGPA ({cgpa}) closer to the placed student average ({benchmarks['CGPA']}).")
    if technical < benchmarks["Technical_Score"]:
        tips.append(f"- Strengthen core technical/coding skills ({technical} vs. placed avg {benchmarks['Technical_Score']}).")
    if aptitude < benchmarks["Aptitude_Score"]:
        tips.append(f"- Practice quantitative & logical aptitude tests ({aptitude} vs. placed avg {benchmarks['Aptitude_Score']}).")
    if internships < benchmarks["Internships"]:
        tips.append(f"- Gain hands-on industry experience ({int(internships)} internships vs. placed avg {benchmarks['Internships']}).")
    if projects < benchmarks["Projects"]:
        tips.append(f"- Build more domain projects ({projects} projects vs. placed avg {benchmarks['Projects']}).")
    if communication < benchmarks["Communication_Score"]:
        tips.append(f"- Practice mock interviews & communication ({communication} vs. placed avg {benchmarks['Communication_Score']}).")
    if attendance < benchmarks["Attendance"]:
        tips.append(f"- Improve class/training attendance ({attendance}% vs. placed avg {benchmarks['Attendance']}%).")

    print("\nPersonalized Improvement Recommendations:")
    if tips:
        for tip in tips:
            print(tip)
    else:
        print("- Strong profile across all metrics! Focus on company-specific interview prep.")

# Run the program
if __name__ == "__main__":
    predict_student_placement()
