import pandas as pd


df = pd.read_csv("test/data/GOLevelAnalysis.csv")

# Group by 'Algorithm' and 'Level', then calculate mean and std of 'F1 Score'
result = (
    df.groupby(["Algorithm", "Level"])
    .agg({"F1 score": ["mean", "std"], "Elapsed time": "mean"})
    .reset_index()
)

# Rename columns for clarity
result.columns = [
    "Algorithm",
    "Level",
    "F1_score_mean",
    "F1_score_std",
    "Elapsed_time_mean",
]

# Add the a column: F1_Score_Mean divided by Elapsed_Time_Mean
result["Efficiency"] = 100 * (
    result["F1_score_mean"] / result["Elapsed_time_mean"]
)

# Display the result
print(result)

# Optionally, save the result to a new CSV file
result.to_csv("result_summary.csv", index=False)
