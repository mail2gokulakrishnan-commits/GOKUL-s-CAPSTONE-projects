# Data Acquisition, Cleaning, and Exploratory Analysis
import os
print(os.listdir('.'))
## reading the file
import pandas as pd
df = pd.read_csv('raw_data.csv', encoding='latin1')
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
## Check duplicates
print("Duplicates:", df.duplicated().sum())
## Skewness
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("Skewness:\n", df[numeric_cols].skew())
## Correlation 
numeric_df = df[['Row ID', 'Postal Code', 'Sales', 'Quantity', 'Discount', 'Profit']]
pearson = numeric_df.corr(method='pearson')
spearman = numeric_df.corr(method='spearman')
diff = (spearman - pearson).abs()
print(diff.unstack().sort_values(ascending=False).drop_duplicates().head(10))
## Copying data
mem_before = df.memory_usage(deep=True).sum()
df_copy = df.copy()
df_copy['Order Date'] = pd.to_datetime(df_copy['Order Date'], format='mixed')
df_copy['Ship Date'] = pd.to_datetime(df_copy['Ship Date'], format='mixed')
df_copy['Ship Mode'] = df_copy['Ship Mode'].astype('category')
df_copy['Region'] = df_copy['Region'].astype('category')
mem_after = df_copy.memory_usage(deep=True).sum()
print(f"Before: {mem_before}, After: {mem_after}")
## Task 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Set plotting style for clean visualizations
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

## TASK 1: Data Acquisition & Initial Inspection

print("--- TASK 1: Loading Data ---")
### Replace 'Sample - Superstore.csv' with your local file path if different
file_path = "Sample - Superstore.csv"

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    # Creating dummy data replicating a Superstore structure for fallback execution
    print(f"File '{file_path}' not found. Generating mock Superstore data for demonstration...")
    np.random.seed(42)
    mock_size = 1000
    df = pd.DataFrame({
        'Row ID': np.arange(1, mock_size + 1),
        'Order ID': [f"CA-2026-{100000 + i}" for i in range(mock_size)],
        'Order Date': pd.date_range(start='2024-01-01', periods=mock_size, freq='D').strftime('%Y-%m-%d'),
        'Ship Mode': np.random.choice(['Standard Class', 'Second Class', 'First Class', 'Same Day'], mock_size),
        'Segment': np.random.choice(['Consumer', 'Corporate', 'Home Office'], mock_size),
        'Region': np.random.choice(['West', 'East', 'Central', 'South'], mock_size),
        'Category': np.random.choice(['Furniture', 'Office Supplies', 'Technology'], mock_size),
        'Sales': np.random.exponential(scale=250, size=mock_size) + 5,
        'Quantity': np.random.randint(1, 15, size=mock_size),
        'Discount': np.random.choice([0.0, 0.2, 0.4, 0.5, 0.7, 0.8], mock_size, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]),
        'Profit': np.random.normal(loc=20, scale=100, size=mock_size)
    })
    # Injecting some artificial issues to clean later
    df.loc[np.random.choice(df.index, 50, replace=False), 'Sales'] = np.nan
    df.loc[np.random.choice(df.index, 250, replace=False), 'Profit'] = np.nan # High null column
    df['Quantity'] = df['Quantity'].astype(str) # Intentional bad data type
    df = pd.concat([df, df.iloc[:15]], ignore_index=True) # Injecting duplicates

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Column Data Types ---")
print(df.dtypes)

print("\n--- DataFrame Shape ---")
print(df.shape)
## Task 2 Null value data analysis
print("\n--- TASK 2: Null Value Analysis ---")
null_counts = df.isnull().sum()
null_percentages = (null_counts / df.shape[0]) * 100

null_df = pd.DataFrame({
    'Null Count': null_counts,
    'Null Percentage (%)': null_percentages
})
print(null_df)

high_null_cols = null_percentages[null_percentages > 20].index.tolist()
print(f"\nColumns exceeding a 20% null rate: {high_null_cols}")

### Fill numeric columns below 20% nulls with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if null_percentages[col] <= 20 and null_counts[col] > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Imputed missing values in '{col}' with column median: {median_val}")
## Task3  Duplicate Detection & Removal
print("\n--- TASK 3: Duplicate Detection & Removal ---")
initial_rows = df.shape[0]
dup_count = df.duplicated().sum()
print(f"Number of duplicate rows detected: {dup_count}")

df_cleaned = df.drop_duplicates()
removed_rows = initial_rows - df_cleaned.shape[0]
print(f"Number of rows removed: {removed_rows}")

### Verify if removal changes null percentages
new_null_pct = (df_cleaned.isnull().sum() / df_cleaned.shape[0]) * 100
print("\nUpdated Null Percentages after duplicate removal:")
print(new_null_pct)
df = df_cleaned 
## Task4 Data type correction
print("\n--- TASK 4: Data Type Correction ---")
mem_before = df.memory_usage(deep=True).sum()

### Convert an incorrect inferred type (e.g., Quantity from object to numeric)
if 'Quantity' in df.columns and df['Quantity'].dtype == 'object':
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    print("Converted 'Quantity' from object to numeric.")

### Convert a repetitive string column to category
rep_cols = ['Ship Mode', 'Segment', 'Region', 'Category']
converted_cat = []
for col in rep_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')
        converted_cat.append(col)
print(f"Converted string columns {converted_cat} to category type.")

mem_after = df.memory_usage(deep=True).sum()
print(f"Memory Usage Before: {mem_before:,} bytes")
print(f"Memory Usage After: {mem_after:,} bytes")
print(f"Memory Saved: {((mem_before - mem_after) / mem_before) * 100:.2f}%")
## Task5 Descriptive Statistics and Skewness
print("\n--- TASK 5: Descriptive Statistics & Skewness ---")
numeric_df = df.select_dtypes(include=[np.number])
print(numeric_df.describe())

skewness_dict = {}
for col in numeric_df.columns:
    skew_val = df[col].skew()
    skewness_dict[col] = skew_val
    print(f"Skewness of '{col}': {skew_val:.4f}")

highest_skew_col = max(skewness_dict, key=lambda k: abs(skewness_dict[k]))
print(f"\nColumn with the highest absolute skewness: '{highest_skew_col}' (Skewness: {skewness_dict[highest_skew_col]:.4f})")
## Task 6 Outlier Detection with IQR
print("\n--- TASK 6: Outlier Detection with IQR ---")
### Pick two numeric columns for outlier analysis (e.g., Sales and Quantity)
outlier_cols = [c for c in ['Sales', 'Quantity'] if c in df.columns]

for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Column '{col}': Found {outliers.shape[0]} rows outside bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
## TASK 7: Visualizations
print("\n--- TASK 7: Generating Visualizations ---")

### Plot 1: Line Plot (using Sales index sequence)
plt.figure()
plt.plot(df.index[:100], df['Sales'].iloc[:100], label='Sales Sequence', color='tab:blue')
plt.title('Line Plot of Sales over First 100 Data Points')
plt.xlabel('Row Index Sequence')
plt.ylabel('Sales ($)')
plt.legend()
plt.tight_layout()
plt.savefig('plot1_line_plot.png')
plt.close()

### Plot 2: Bar Chart 
plt.figure()
if 'Category' in df.columns and 'Sales' in df.columns:
    df.groupby('Category', observed=False)['Sales'].mean().plot.bar(color='tab:orange')
    plt.title('Average Sales across Product Categories')
    plt.xlabel('Category')
    plt.ylabel('Mean Sales ($)')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plot2_bar_chart.png')
plt.close()

### Plot 3: Histogram of the most skewed variable
plt.figure()
sns.histplot(df[highest_skew_col].dropna(), bins=20, kde=True, color='tab:red')
plt.title(f'Histogram Distribution of Most Skewed Column: {highest_skew_col}')
plt.xlabel(highest_skew_col)
plt.ylabel('Count Frequency')
plt.tight_layout()
plt.savefig('plot3_histogram.png')
plt.close()

### Plot 4: Scatter Plot (Sales vs Quantity)
plt.figure()
if 'Sales' in df.columns and 'Quantity' in df.columns:
    sns.scatterplot(data=df, x='Quantity', y='Sales', alpha=0.6, color='tab:green')
    plt.title('Scatter Plot: Sales vs Quantity Purchased')
    plt.xlabel('Quantity Ordered')
    plt.ylabel('Sales Volume ($)')
plt.tight_layout()
plt.savefig('plot4_scatter_plot.png')
plt.close()

### Plot 5: Box Plot (Sales split by Market Segment)
plt.figure()
if 'Segment' in df.columns and 'Sales' in df.columns:
    sns.boxplot(data=df, x='Segment', y='Sales', hue='Segment', legend=False, palette='Set2')
    plt.title('Box Plot of Sales Distributed by Customer Segment')
    plt.xlabel('Customer Segment')
    plt.ylabel('Sales ($)')
plt.tight_layout()
plt.savefig('plot5_box_plot.png')
plt.close()

print("All five standard visualization figures have been exported successfully.")
## TASK 8: Correlation Heat Map
print("\n--- TASK 8: Correlation Heat Map ---")
numeric_df = df.select_dtypes(include=[np.number])
pearson_corr = numeric_df.corr(method='pearson')

plt.figure(figsize=(8, 6))
sns.heatmap(pearson_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Pearson Linear Correlation Coefficient Heatmap')
plt.tight_layout()
plt.savefig('plot6_correlation_heatmap.png')
plt.close()

### Find highest absolute correlation pair
corr_matrix_abs = pearson_corr.abs()
np.fill_diagonal(corr_matrix_abs.values, 0)
max_corr_idx = corr_matrix_abs.unstack().idxmax()
print(f"Pair with highest absolute Pearson correlation: {max_corr_idx} (r = {pearson_corr.loc[max_corr_idx]:.4f})")
## TASK 9a: Imputation Strategy Comparison
print("\n--- TASK 9b: Spearman Rank Correlation ---")
spearman_corr = numeric_df.corr(method='spearman')
diff_matrix = (spearman_corr - pearson_corr).abs()

print("\n--- Pearson Correlation Matrix ---")
print(pearson_corr)
print("\n--- Spearman Rank Correlation Matrix ---")
print(spearman_corr)

### Unstack matrices to evaluate the largest absolute discrepancies between indices
np.fill_diagonal(diff_matrix.values, 0)
diff_unstacked = diff_matrix.unstack().drop_duplicates().sort_values(ascending=False)

print("\nTop 3 Column Pairs with Largest Absolute Discrepancies |Spearman - Pearson|:")
count = 0
for idx, val in diff_unstacked.items():
    if count >= 3:
        break
    col1, col2 = idx
    print(f"Pair: {col1} & {col2} -> |Spearman - Pearson| = {val:.4f} (Spearman: {spearman_corr.loc[col1, col2]:.4f}, Pearson: {pearson_corr.loc[col1, col2]:.4f})")
    count += 1
## TASK 9c: Grouped Aggregation
print("\n--- TASK 9c: Grouped Aggregation ---")
# Use Segment (Categorical) and Sales (Numeric)
cat_feature = 'Segment' if 'Segment' in df.columns else df.select_dtypes(include=['category', 'object']).columns[0]
num_feature = 'Sales' if 'Sales' in df.columns else df.select_dtypes(include=[np.number]).columns[0]

grouped_stats = df.groupby(cat_feature, observed=False)[num_feature].agg(['mean', 'std', 'count'])
print(grouped_stats)

max_mean_idx = grouped_stats['mean'].idxmax()
min_mean_idx = grouped_stats['mean'].idxmin()
ratio_mean = grouped_stats.loc[max_mean_idx, 'mean'] / grouped_stats.loc[min_mean_idx, 'mean']
print(f"\nRatio of highest group mean to lowest group mean: {ratio_mean:.4f}")
## TASK 10: Saving Cleaned Data
output_file = 'cleaned_data.csv'
df.to_csv(output_file, index=False)
print(f"\nTask 10 Complete: Cleaned dataframe exported successfully to '{output_file}'.")