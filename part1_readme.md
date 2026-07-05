import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6), 'axes.labelsize': 12, 'axes.titlesize': 14})

load_dotenv()
api_key = os.getenv("API_KEY") # Example of handling required secrets safely

print("--- TASK 1: Loading Data ---")

df = pd.read_csv('raw_data.csv') 

print("\nFirst 5 Rows:")
print(df.head())
print("\nColumn Data Types:")
print(df.dtypes)
print(f"\nDataFrame Shape: {df.shape}")

print("\n--- TASK 2: Null Value Analysis ---")
null_counts = df.isnull().sum()
null_percentages = (null_counts / df.shape[0]) * 100

null_df = pd.DataFrame({'Null Count': null_counts, 'Null Percentage': null_percentages})
print(null_df)

high_null_cols = null_percentages[null_percentages > 20].index.tolist()
print(f"\nColumns exceeding 20% null rate (To be dropped/handled specially): {high_null_cols}")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cols_to_impute_median = [col for col in numeric_cols if null_percentages[col] <= 20 and null_percentages[col] > 0]

for col in cols_to_impute_median:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"Imputed missing values in numeric column '{col}' with median: {median_val}")

print("\n--- TASK 3: Duplicate Detection & Removal ---")
initial_shape = df.shape
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows identified: {duplicate_count}")

df = df.drop_duplicates()
final_shape = df.shape
rows_removed = initial_shape[0] - final_shape[0]
print(f"Rows removed: {rows_removed}. New shape: {final_shape}")

post_null_percentages = (df.isnull().sum() / df.shape[0]) * 100
print("\nVerifying if null percentages shifted post duplicate removal:")
print(pd.DataFrame({'New Null %': post_null_percentages}))


print("\n--- TASK 4: Data Type Correction ---")
mem_before = df.memory_usage(deep=True).sum()

categorical_candidates = df.select_dtypes(include=['object']).columns.tolist()
if categorical_candidates:
    cat_col = categorical_candidates[0]
    df[cat_col] = df[cat_col].astype('category')
    print(f"Converted repetitive string column '{cat_col}' to category type.")

mem_after = df.memory_usage(deep=True).sum()
print(f"Memory usage BEFORE conversion: {mem_before:,} bytes")
print(f"Memory usage AFTER conversion: {mem_after:,} bytes")
print(f"Absolute saved space: {mem_before - mem_after:,} bytes")

print("\n--- TASK 5: Descriptive Statistics & Skewness ---")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nDescriptive Statistics Table:")
print(df[numeric_cols].describe())

skewness_dict = {}
print("\nColumn Skewness Values:")
for col in numeric_cols:
    skew_val = df[col].skew()
    skewness_dict[col] = skew_val
    print(f"Column '{col}': Skewness = {skew_val:.4f}")

highest_skew_col = max(skewness_dict, key=lambda k: abs(skewness_dict[k]))
print(f"\nColumn with the highest absolute skewness: '{highest_skew_col}' (Skewness: {skewness_dict[highest_skew_col]:.4f})")

print("\n--- TASK 6: Outlier Detection with IQR ---")

outlier_cols = numeric_cols[:2]

for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    
    outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    print(f"Column '{col}': Found {outliers_count} rows out of bounds. Lower: {lower_bound:.2f}, Upper: {upper_bound:.2f}")

print("\n--- Generating Plots (Saving to Disk) ---")

plt.figure()
plt.plot(df.index, df[numeric_cols[0]], color='blue', alpha=0.7)
plt.title(f'Line Plot of {numeric_cols[0]} Over Index Sequence')
plt.xlabel('Row Index')
plt.ylabel(numeric_cols[0])
plt.tight_layout()
plt.savefig('line_plot.png')
plt.close()

categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
if categorical_cols and numeric_cols:
    plt.figure()
    df.groupby(categorical_cols[0])[numeric_cols[0]].mean().plot.bar(color='teal', edgecolor='black')
    plt.title(f'Mean of {numeric_cols[0]} across {categorical_cols[0]}')
    plt.xlabel(categorical_cols[0])
    plt.ylabel(f'Mean {numeric_cols[0]}')
    plt.tight_layout()
    plt.savefig('bar_chart.png')
    plt.close()

plt.figure()
sns.histplot(df[highest_skew_col], bins=20, kde=True, color='crimson')
plt.title(f'Distribution Histogram of Highly Skewed Column: {highest_skew_col}')
plt.xlabel(highest_skew_col)
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('histogram_skewed.png')
plt.close()

if len(numeric_cols) >= 2:
    plt.figure()
    sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1], alpha=0.6)
    plt.title(f'Scatter Matrix Relationship: {numeric_cols[0]} vs {numeric_cols[1]}')
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.tight_layout()
    plt.savefig('scatter_plot.png')
    plt.close()

if categorical_cols and numeric_cols:
    plt.figure()
    sns.boxplot(data=df, x=categorical_cols[0], y=numeric_cols[0])
    plt.title(f'Box Plot of {numeric_cols[0]} Stratified by {categorical_cols[0]}')
    plt.xlabel(categorical_cols[0])
    plt.ylabel(numeric_cols[0])
    plt.tight_layout()
    plt.savefig('box_plot.png')
    plt.close()

plt.figure(figsize=(8, 6))
pearson_corr = df[numeric_cols].corr(method='pearson')
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Pearson Linear Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

corr_matrix_abs = pearson_corr.abs()
np.fill_diagonal(corr_matrix_abs.values, 0)
max_corr_idx = corr_matrix_abs.unstack().idxmax()
print(f"Highest absolute Pearson correlation is between {max_corr_idx} with a value of {pearson_corr.loc[max_corr_idx[0], max_corr_idx[1]]:.4f}")

print("\n--- ADVANCED SECTION A: Imputation Strategy Comparison ---")

sorted_skew = sorted(skewness_dict.items(), key=lambda item: abs(item[1]), reverse=True)
top_skewed_cols = [sorted_skew[0][0], sorted_skew[1][0]]

for col in top_skewed_cols:
    col_mean = df[col].mean()
    col_median = df[col].median()
    print(f"Column '{col}' -> Pre-Imputation Mean: {col_mean:.4f} | Pre-Imputation Median: {col_median:.4f}")
    
    # Fill remaining nulls specifically in these two columns if any are left
    df[col] = df[col].fillna(col_median)

print("Verification of remaining nulls in top skewed columns:")
print(df[top_skewed_cols].isnull().sum())

print("\n--- ADVANCED SECTION B: Spearman Rank Correlation Comparison ---")
spearman_corr = df[numeric_cols].corr(method='spearman')
diff_matrix = (spearman_corr - pearson_corr).abs()

print("\nPearson Matrix Summary:\n", pearson_corr)
print("\nSpearman Rank Matrix Summary:\n", spearman_corr)

diff_unstacked = diff_matrix.unstack()

diff_unstacked = diff_unstacked[diff_unstacked.index.get_level_values(0) != diff_unstacked.index.get_level_values(1)]
top_3_diffs = diff_unstacked.sort_values(ascending=False).iloc[::2].head(3)

print("\nTop 3 Column Pairs with Largest Absolute |Spearman - Pearson| Differences:")
for pairs, val in top_3_diffs.items():
    p_val = pearson_corr.loc[pairs[0], pairs[1]]
    s_val = spearman_corr.loc[pairs[0], pairs[1]]
    print(f"Pair {pairs}: |Spearman - Pearson| = {val:.4f} (Pearson: {p_val:.2f}, Spearman: {s_val:.2f})")

print("\n--- ADVANCED SECTION C: Grouped Aggregation ---")
if categorical_cols and numeric_cols:
    group_res = df.groupby(categorical_cols[0])[numeric_cols[0]].agg(['mean', 'std', 'count'])
    print(group_res)
    
    highest_mean_group = group_res['mean'].idxmax()
    highest_std_group = group_res['std'].idxmax()
    
    min_mean = group_res['mean'].min()
    max_mean = group_res['mean'].max()
    mean_ratio = max_mean / min_mean if min_mean != 0 else np.nan
    print(f"\nHighest Mean Group: {highest_mean_group} | Highest Std Dev Group: {highest_std_group}")
    print(f"Ratio of Highest Group Mean to Lowest Group Mean: {mean_ratio:.2f}")

df.to_csv('cleaned_data.csv', index=False)
print("\nSuccessfully exported clean dataset to 'cleaned_data.csv'.")