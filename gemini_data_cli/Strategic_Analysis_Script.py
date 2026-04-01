import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setup
OUTPUT_DIR = './eda_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style='whitegrid')

# Load Data
df = pd.read_csv('bestsellers with categories.csv')

# 1. Price Sweet Spot by Genre
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Genre', y='Price', palette=['#60a5fa', '#10b981'], showfliers=False)
plt.title('Price Distribution: The \"Sweet Spot\" by Genre', fontweight='bold')
plt.savefig(f'{OUTPUT_DIR}/price_sweet_spot_box.png')

# 2. Satisfaction Heatmap (Price vs Rating)
df['Price_Range'] = pd.cut(df['Price'], bins=[0, 10, 20, 30, 110], labels=['Budget (<$10)', 'Standard ($10-20)', 'Premium ($20-30)', 'Luxury (>$30)'])
satisfaction = df.groupby(['Price_Range', 'Genre'], observed=True)['User Rating'].mean().unstack()
plt.figure(figsize=(10, 6))
sns.heatmap(satisfaction, annot=True, cmap='RdYlGn', fmt='.3f')
plt.title('Customer Satisfaction vs Price Strategy')
plt.savefig(f'{OUTPUT_DIR}/satisfaction_price_heatmap.png')

# 3. Bestseller Survival (Rating Threshold)
book_stats = df.groupby('Name').agg(years_count=('Year', 'nunique'), avg_rating=('User Rating', 'mean')).reset_index()
book_stats['is_repeat'] = (book_stats['years_count'] > 1).astype(int)
rating_prob = book_stats.groupby('avg_rating')['is_repeat'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=rating_prob, x='avg_rating', y='is_repeat', marker='o', color='#ef4444')
plt.axhline(0.5, color='black', linestyle='--')
plt.title('Bestseller Survival: Probability of Repeating by Rating')
plt.savefig(f'{OUTPUT_DIR}/bestseller_survival_curve.png')

print(f"Strategic Analysis Complete. Charts saved to {OUTPUT_DIR}")
