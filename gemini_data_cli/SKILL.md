---
name: gemini-eda-agent-v2
description: >
  Use this skill whenever the user wants to perform exploratory data analysis,
  data cleaning, insight generation, or problem framing on a dataset.
  Triggers: 'analyse this CSV', 'clean my data', 'EDA', 'explore this dataset',
  'what's wrong with my data', 'get insights', 'help me understand this data',
  'prepare data for modelling'. This skill covers the FULL analyst workflow —
  from problem definition through to prioritised recommendations ready for action.
version: 2.0.0
phases: [context, data_quality, data_understanding, insight_generation, output]
---

# Gemini CLI — Full Analyst EDA & Insight Pipeline

## Philosophy

A data analyst's job is not to clean data. It is to answer a business question
using data. Cleaning is just one step in service of that goal.

This skill enforces four phases in strict order:
  PHASE 0 — Understand the problem and the data before touching anything
  PHASE 1 — Assess and fix data quality
  PHASE 2 — Understand what the data is actually saying
  PHASE 3 — Translate findings into insights with business impact
  PHASE 4 — Deliver prioritised, actionable recommendations

Never skip phases. Never jump to cleaning before understanding context.
Never produce observations without connecting them to a so-what.

---

## Critical Setup — Run at Start of Every Session

```python
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # REQUIRED — no display in CLI
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

OUTPUT_DIR = './eda_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SAFETY RULE — always snapshot original immediately after load
# df_original = df.copy()   ← set this as soon as df is loaded
# df_clean    = df.copy()   ← all cleaning happens here only
```

---

# PHASE 0 — CONTEXT & PROBLEM DEFINITION

> Run before any code. This phase requires human input. Do not skip.

## Step 0.0 — Problem Definition

Before loading any data, ask and document:

```
QUESTIONS TO ANSWER WITH THE DATA OWNER / STAKEHOLDER:
1. What business question are we trying to answer?
   (e.g. reduce readmissions / predict churn / optimise scheduling)
2. Who will consume this analysis — technical or non-technical?
3. What decision will the output inform?
4. What does "success" look like for this analysis?
5. Is there a target variable / outcome we are predicting?
6. What time period does this data cover and is it complete?
7. Has any filtering or sampling already been applied to the data?
```

Document answers in a `problem_definition.md` file before proceeding.

## Step 0.1 — Data Dictionary Capture

```python
# Before profiling, document what each column means in business terms
# Fill this in with the data owner — do not assume

data_dictionary = {
    'column_name': {
        'business_definition': 'What does this column represent?',
        'valid_values':        'Known valid range or accepted categories',
        'business_rules':      'e.g. visit_date must always be after date_of_birth',
        'notes':               'Any known quirks, system limitations, or caveats'
    }
    # ... one entry per column
}
```

## Step 0.2 — Data Sufficiency Check

```python
df = pd.read_csv(filepath)
df_original = df.copy()
df_clean    = df.copy()

date_cols = [c for c in df.columns if 'date' in c.lower()]
for dc in date_cols:
    df_clean[dc] = pd.to_datetime(df_clean[dc], format='mixed', errors='coerce')

date_col = date_cols[0] if date_cols else None

print("DATA SUFFICIENCY ASSESSMENT")
print(f"  Rows            : {len(df):,}")
print(f"  Columns         : {len(df.columns)}")
print(f"  Date range      : {df_clean[date_col].min().date()} → {df_clean[date_col].max().date()}" if date_col else "  Date range: N/A")

# Flag common sufficiency problems
if len(df) < 1000:
    print("  ⚠️  < 1,000 rows — small dataset, statistical significance limited")
if date_col:
    days = (df_clean[date_col].max() - df_clean[date_col].min()).days
    if days < 365:
        print("  ⚠️  < 1 year of data — may not capture seasonal patterns")

# Structural insight — are rows independent or longitudinal?
id_cols = [c for c in df.columns if 'id' in c.lower()]
for id_col in id_cols:
    n_unique = df[id_col].nunique()
    if n_unique < len(df):
        avg_rows = len(df) / n_unique
        print(f"\n  ⚠️  LONGITUDINAL DATA DETECTED")
        print(f"     {id_col}: {n_unique} unique values across {len(df)} rows")
        print(f"     Average {avg_rows:.1f} rows per {id_col}")
        print(f"     → Unit of analysis must be decided: row-level OR {id_col}-level")
        print(f"     → Row-level analysis risks data leakage and inflated sample size")
```

---

# PHASE 1 — DATA QUALITY

## Step 1.0 — Load & Validate

```python
try:
    df = pd.read_csv(filepath, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(filepath, encoding='latin-1')
    print("⚠️  Encoding fallback: latin-1")

df_original = df.copy()
df_clean    = df.copy()

print(f"✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Memory: {df.memory_usage(deep=True).sum()/1e6:.2f} MB")
print(f"\nDtypes:\n{df.dtypes.to_string()}")
print(f"\nFirst 5 rows:\n{df.head(5).to_string()}")

# Structural validation
issues = []
empty_cols    = [c for c in df.columns if df[c].isna().all()]
constant_cols = [c for c in df.select_dtypes(include='number').columns if df[c].nunique() == 1]
id_like_cols  = [c for c in df.columns if df[c].nunique() == len(df) and df[c].dtype == object]
for c in df.columns:
    if df[c].dropna().apply(type).nunique() > 1:
        issues.append(f"🟡 Mixed types in column: {c}")
if empty_cols:    issues.append(f"🔴 Completely empty: {empty_cols}")
if constant_cols: issues.append(f"🟡 Zero variance: {constant_cols}")
if id_like_cols:  issues.append(f"🟡 Possible ID columns (all unique): {id_like_cols}")
for i in issues: print(i)
```

## Step 1.1 — Missing Value Analysis

```python
missing     = df_clean.isnull().sum()
missing_pct = (missing / len(df_clean) * 100).round(2)
missing_df  = pd.DataFrame({
    'Missing Count': missing, 'Missing %': missing_pct, 'Dtype': df_clean.dtypes,
    'Severity': missing_pct.apply(
        lambda x: '🔴 Critical (>50%)' if x > 50
        else '🟡 Warning (10-50%)' if x > 10
        else '🟢 Minor (<10%)' if x > 0 else '✓ None'
    ),
    'Recommendation': missing_pct.apply(
        lambda x: 'Drop column' if x > 50
        else 'Impute — check dtype' if x > 10
        else 'Impute or drop rows' if x > 0 else '—'
    )
}).sort_values('Missing %', ascending=False)

print(missing_df.to_string())

# Visualise
cols_miss = missing_df[missing_df['Missing Count'] > 0]
if len(cols_miss) > 0:
    fig, ax = plt.subplots(figsize=(10, max(4, len(cols_miss)*0.5)))
    colors = ['#ef4444' if p>50 else '#f59e0b' if p>10 else '#10b981'
              for p in cols_miss['Missing %']]
    ax.barh(cols_miss.index, cols_miss['Missing %'], color=colors)
    ax.axvline(50, color='red', linestyle='--', alpha=0.5, label='50%')
    ax.axvline(10, color='orange', linestyle='--', alpha=0.5, label='10%')
    ax.set_title('Missing Values by Column'); ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/missing_values.png', dpi=150); plt.close()
```

## Step 1.2 — Duplicate Detection

```python
n_exact = df_clean.duplicated().sum()
print(f"Exact duplicate rows: {n_exact}")

# Check key ID columns for partial duplicates
key_cols = [c for c in df_clean.columns
            if any(k in c.lower() for k in ['id','key','code','ref'])]
for c in key_cols:
    n = df_clean.duplicated(subset=[c]).sum()
    if n > 0:
        print(f"⚠️  Duplicate {c}: {n} rows ({n/len(df_clean)*100:.1f}%)")
        print(f"   → In healthcare/CRM: could be repeat encounters (expected)")
        print(f"   → Or could be data entry errors (needs review)")
```

## Step 1.3 — Dtype & Format Checks

```python
# Auto-detect and convert date columns stored as strings
for c in df_clean.columns:
    if df_clean[c].dtype == object:
        sample = df_clean[c].dropna().head(20)
        try:
            pd.to_datetime(sample, format='mixed')
            df_clean[c] = pd.to_datetime(df_clean[c], format='mixed', errors='coerce')
            print(f"✓ Converted {c} → datetime")
        except: pass

# Detect numeric columns stored as strings (e.g. "$3,500" or "3.5%")
for c in df_clean.select_dtypes(include='object').columns:
    cleaned = df_clean[c].str.replace(r'[\$,%,]','',regex=True)
    try:
        pd.to_numeric(cleaned.dropna().head(20))
        print(f"⚠️  {c} may be numeric stored as string — consider fix_dtype")
    except: pass
```

## Step 1.4 — Domain & Business Rule Validation

```python
# Validate relationships between columns using known business rules
# Customise these rules per dataset — examples below:

validation_results = []

# Rule: date_a must be before date_b
# e.g. visit_date must be after date_of_birth
if 'visit_date' in df_clean.columns and 'date_of_birth' in df_clean.columns:
    invalid = (df_clean['visit_date'] < df_clean['date_of_birth']).sum()
    validation_results.append({
        'rule': 'visit_date > date_of_birth',
        'violations': invalid,
        'severity': '🔴 Critical' if invalid > 0 else '✓ OK'
    })

# Rule: no future dates
for dc in df_clean.select_dtypes(include='datetime').columns:
    future = (df_clean[dc] > pd.Timestamp.now()).sum()
    validation_results.append({
        'rule': f'{dc} not in future',
        'violations': future,
        'severity': '🔴 Critical' if future > 0 else '✓ OK'
    })

# Rule: numeric ranges
# e.g. age must be between 0 and 120
if 'patient_age' in df_clean.columns:
    invalid_age = ((df_clean['patient_age'] < 0) | (df_clean['patient_age'] > 120)).sum()
    validation_results.append({
        'rule': '0 ≤ patient_age ≤ 120',
        'violations': invalid_age,
        'severity': '🔴 Critical' if invalid_age > 0 else '✓ OK'
    })

# Rule: cross-column consistency
# e.g. if age and DOB both exist, they should agree
if 'patient_age' in df_clean.columns and 'date_of_birth' in df_clean.columns and 'visit_date' in df_clean.columns:
    calc_age = ((df_clean['visit_date'] - df_clean['date_of_birth']).dt.days / 365.25).round(0)
    mismatch = ((calc_age - df_clean['patient_age']).abs() > 2).sum()
    df_clean['patient_age_calculated'] = calc_age
    validation_results.append({
        'rule': 'patient_age consistent with date_of_birth',
        'violations': mismatch,
        'severity': '🔴 Critical' if mismatch > 0 else '✓ OK'
    })

vr_df = pd.DataFrame(validation_results)
print("Business Rule Validation:")
print(vr_df.to_string(index=False))
```

## Step 1.5 — Anomaly Flags with Severity

```python
flags = []

for c in df_clean.columns:
    miss_pct = df_clean[c].isna().mean() * 100
    if miss_pct > 50:
        flags.append({'Column':c,'Issue':f'Missing {miss_pct:.1f}%','Severity':'🔴 Critical','Business Impact':'Column unusable — will introduce massive bias','Recommendation':'Drop column'})
    elif miss_pct > 10:
        flags.append({'Column':c,'Issue':f'Missing {miss_pct:.1f}%','Severity':'🟡 Warning','Business Impact':'Missing data will bias distributions and model training','Recommendation':'Impute — choose strategy based on dtype and distribution'})

for c in df_clean.select_dtypes(include='number').columns:
    q1,q3 = df_clean[c].quantile(0.25), df_clean[c].quantile(0.75)
    iqr   = q3 - q1
    oc    = int(((df_clean[c] < q1-1.5*iqr) | (df_clean[c] > q3+1.5*iqr)).sum())
    skew  = df_clean[c].skew()
    if oc/len(df_clean) > 0.05:
        flags.append({'Column':c,'Issue':f'{oc} outliers ({oc/len(df_clean)*100:.1f}%)','Severity':'🟡 Warning','Business Impact':'Outliers will distort means, correlations, and linear models','Recommendation':'Investigate — cap with IQR fence or log-transform'})
    if abs(skew) > 2:
        flags.append({'Column':c,'Issue':f'High skew ({skew:.2f})','Severity':'🟡 Warning','Business Impact':'Skewed features reduce linear model performance','Recommendation':'Log or sqrt transform before modelling'})

flags_df = pd.DataFrame(flags) if flags else pd.DataFrame()
if len(flags_df):
    print(flags_df.to_string(index=False))
    flags_df.to_csv(f'{OUTPUT_DIR}/anomaly_flags.csv', index=False)
```

---

# PHASE 2 — DATA UNDERSTANDING

## Step 2.0 — Numeric Profiling

```python
num_cols = df_clean.select_dtypes(include='number').columns.tolist()
if num_cols:
    profile = df_clean[num_cols].describe(percentiles=[.05,.25,.5,.75,.95]).T
    profile['skewness']     = df_clean[num_cols].skew().round(3)
    profile['kurtosis']     = df_clean[num_cols].kurt().round(3)
    profile['outliers_IQR'] = [int(((df_clean[c] < df_clean[c].quantile(.25)-1.5*(df_clean[c].quantile(.75)-df_clean[c].quantile(.25))) | (df_clean[c] > df_clean[c].quantile(.75)+1.5*(df_clean[c].quantile(.75)-df_clean[c].quantile(.25)))).sum()) for c in num_cols]
    print(profile.to_string())
```

## Step 2.1 — Categorical Profiling

```python
cat_cols = df_clean.select_dtypes(include='object').columns.tolist()
for c in cat_cols:
    vc   = df_clean[c].value_counts()
    rare = (vc / len(df_clean) < 0.01).sum()
    print(f"\n{c}: {df_clean[c].nunique()} unique | {df_clean[c].isna().sum()} missing | {rare} rare labels (<1%)")
    print(vc.head(5).to_string())
```

## Step 2.2 — Temporal Analysis

```python
date_cols = df_clean.select_dtypes(include='datetime').columns.tolist()
if date_cols:
    primary_date = date_cols[0]
    df_clean['_year']  = df_clean[primary_date].dt.year
    df_clean['_month'] = df_clean[primary_date].dt.month
    df_clean['_ym']    = df_clean[primary_date].dt.to_period('M')

    print("Volume by year:")
    print(df_clean.groupby('_year').size().to_string())

    # Gap detection — months with zero records
    all_months  = pd.period_range(df_clean[primary_date].min(), df_clean[primary_date].max(), freq='M')
    actual      = df_clean['_ym'].unique()
    gaps        = [str(m) for m in all_months if m not in actual]
    if gaps: print(f"⚠️  Months with zero records (possible data gaps): {gaps}")

    # For longitudinal data — time between records per entity
    id_cols = [c for c in df_clean.columns if 'id' in c.lower() and '_' not in c[0]]
    if id_cols:
        df_sorted = df_clean.sort_values([id_cols[0], primary_date])
        df_sorted['_days_gap'] = df_sorted.groupby(id_cols[0])[primary_date].diff().dt.days
        gap = df_sorted['_days_gap'].dropna()
        print(f"\nDays between records per {id_cols[0]}: mean={gap.mean():.0f}d, median={gap.median():.0f}d, min={gap.min():.0f}d, max={gap.max():.0f}d")
        same_day = (gap == 0).sum()
        if same_day > 0: print(f"⚠️  {same_day} same-day records — investigate duplicates vs multi-encounter")
```

## Step 2.3 — Cohort & Segmentation Analysis

```python
# Slice data by key categorical dimensions
# Customise segmentation columns based on the dataset
segment_cols = [c for c in df_clean.columns
                if df_clean[c].dtype == object and 2 <= df_clean[c].nunique() <= 20]

for sc in segment_cols[:3]:   # limit to top 3 segments
    print(f"\nSegmentation by {sc}:")
    seg = df_clean.groupby(sc).agg(
        count=('visit_id' if 'visit_id' in df_clean.columns else df_clean.columns[0], 'count'),
    )
    print(seg.to_string())

# If age column exists, create age bands
age_col = next((c for c in df_clean.columns if 'age' in c.lower()), None)
if age_col and df_clean[age_col].dtype in ['float64','int64']:
    df_clean['_age_group'] = pd.cut(df_clean[age_col],
        bins=[0,18,30,45,60,75,120],
        labels=['<18','18-30','31-45','46-60','61-75','75+'])
    print(f"\nDistribution by age group:\n{df_clean['_age_group'].value_counts().sort_index().to_string()}")
```

## Step 2.4 — Distribution & Correlation Plots

```python
# Numeric histograms
if num_cols:
    n = len(num_cols); cols_r = 3; rows = (n+cols_r-1)//cols_r
    fig, axes = plt.subplots(rows, cols_r, figsize=(15, rows*3.5))
    axes = axes.flatten() if rows > 1 else [axes]*n
    for i, c in enumerate(num_cols):
        axes[i].hist(df_clean[c].dropna(), bins=30, color='#10b981', alpha=0.75)
        axes[i].axvline(df_clean[c].mean(), color='red', lw=1.5, linestyle='--', label='mean')
        axes[i].axvline(df_clean[c].median(), color='orange', lw=1.5, linestyle=':', label='median')
        axes[i].set_title(c, fontsize=9); axes[i].legend(fontsize=7)
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/distributions_numeric.png', dpi=150); plt.close()
    # GEMINI VISION: pass PNG — ask "are any distributions heavily skewed or bimodal?"

# Correlation matrix
if len(num_cols) >= 2:
    corr = df_clean[num_cols].corr()
    fig, ax = plt.subplots(figsize=(max(7,len(num_cols)), max(5,len(num_cols)-1)))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', vmin=-1, vmax=1,
                center=0, ax=ax, linewidths=0.5)
    ax.set_title('Correlation Matrix')
    plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/correlation_matrix.png', dpi=150); plt.close()
    high = [(corr.columns[i], corr.columns[j], corr.iloc[i,j])
            for i in range(len(corr.columns)) for j in range(i)
            if abs(corr.iloc[i,j]) > 0.85]
    if high:
        print(f"⚠️  Highly correlated pairs (|r|>0.85): {high}")
    # GEMINI VISION: pass PNG — ask "which feature clusters are most correlated?"
```

## Step 2.5 — Patient / Entity Frequency Segmentation

```python
# For longitudinal datasets — segment entities by how often they appear
# Replace 'patient_id' with your entity ID column
id_col = next((c for c in df_clean.columns if 'id' in c.lower() and df_clean[c].nunique() < len(df_clean)), None)

if id_col:
    freq = df_clean.groupby(id_col).size().reset_index(name='record_count')
    freq['frequency_segment'] = pd.cut(
        freq['record_count'],
        bins=[0, 2, 5, 999],
        labels=['Low (1-2)', 'Medium (3-5)', 'High (6+)']
    )
    print(f"\nEntity Frequency Segmentation ({id_col}):")
    print(freq.groupby('frequency_segment', observed=True).agg(
        entities=('record_count', 'count'),
        avg_records=('record_count', 'mean'),
        max_records=('record_count', 'max')
    ).round(1).to_string())

    # Flag: a small % of entities driving most records = high utilisation signal
    top20_pct = freq.nlargest(max(1, int(len(freq)*0.2)), 'record_count')['record_count'].sum()
    pct_driven = top20_pct / len(df_clean) * 100
    print(f"\n⚑ Top 20% of {id_col}s drive {pct_driven:.1f}% of all records")
    if pct_driven > 60:
        print(f"  → High concentration — investigate this group specifically")

    # Visualise
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(freq['record_count'], bins=20, color='#10b981', alpha=0.8)
    axes[0].axvline(freq['record_count'].mean(), color='red', lw=2, linestyle='--',
                    label=f"Mean={freq['record_count'].mean():.1f}")
    axes[0].set_title(f'Records per {id_col}', fontweight='bold')
    axes[0].set_xlabel('Record Count'); axes[0].legend()

    seg_counts = freq['frequency_segment'].value_counts()
    axes[1].pie(seg_counts.values, labels=seg_counts.index,
                colors=['#10b981','#60a5fa','#ef4444'],
                autopct='%1.0f%%', startangle=90)
    axes[1].set_title(f'{id_col} Frequency Segments', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/entity_frequency_segmentation.png', dpi=150)
    plt.close()
    print(f"✓ entity_frequency_segmentation.png saved")
```

## Step 2.6 — Feature Relationship Profiling (beyond correlation)

```python
# Categorical vs target (if target variable is defined)
# Replace 'target_col' with your actual target variable name
target_col = None   # e.g. 'churned', 'readmitted', 'diagnosis_group'

if target_col and target_col in df_clean.columns:
    print(f"\nTarget variable: {target_col}")
    print(f"Distribution:\n{df_clean[target_col].value_counts(normalize=True).round(3).to_string()}")
    baseline = df_clean[target_col].value_counts(normalize=True).max()
    print(f"Baseline accuracy (always predict majority): {baseline:.1%}")

    # Numeric feature vs target
    for c in num_cols:
        groups = df_clean.groupby(target_col)[c].mean()
        print(f"\n{c} mean by {target_col}:\n{groups.to_string()}")

    # Categorical feature vs target (frequency cross-tab)
    for c in cat_cols[:5]:
        ct = pd.crosstab(df_clean[c], df_clean[target_col], normalize='index').round(3)
        print(f"\n{c} vs {target_col}:\n{ct.to_string()}")
```

---

# PHASE 3 — INSIGHT GENERATION

## Step 3.0 — Observation → Insight → Impact Framework

For every significant finding, follow this exact structure.
An observation without an insight is just noise.
An insight without an impact is just trivia.

```python
insights = []

def add_insight(category, observation, insight, business_impact, action_required, decision_owner, impact_level):
    """
    impact_level: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'
    """
    insights.append({
        'category':        category,
        'observation':     observation,
        'insight':         insight,
        'business_impact': business_impact,
        'action_required': action_required,
        'decision_owner':  decision_owner,
        'impact_level':    impact_level
    })

# EXAMPLES — replace with findings from your dataset:

# add_insight(
#     category       = 'Data Structure',
#     observation    = '42 unique patients, 200 visits — avg 4.8 visits/patient',
#     insight        = 'This is longitudinal data, not cross-sectional. Each patient has a multi-year history.',
#     business_impact= 'HIGH — Treating rows as independent inflates sample size and causes data leakage.',
#     action_required= 'Decide unit of analysis: visit-level or patient-level before any modelling.',
#     decision_owner = 'Data Owner + Clinical Lead',
#     impact_level   = 'HIGH'
# )

# Print all insights
for ins in insights:
    print(f"\n[{ins['impact_level']}] {ins['category']}")
    print(f"  OBSERVATION : {ins['observation']}")
    print(f"  INSIGHT     : {ins['insight']}")
    print(f"  IMPACT      : {ins['business_impact']}")
    print(f"  ACTION      : {ins['action_required']}")
    print(f"  OWNER       : {ins['decision_owner']}")

pd.DataFrame(insights).to_csv(f'{OUTPUT_DIR}/insights_and_impact.csv', index=False)
```

## Step 3.1 — Hypothesis Generation

```python
# Based on insights, generate testable hypotheses for next steps
hypotheses = []

def add_hypothesis(hypothesis, test_method, data_needed, priority):
    hypotheses.append({
        'hypothesis':    hypothesis,
        'test_method':   test_method,
        'data_needed':   data_needed,
        'priority':      priority
    })

# EXAMPLES:
# add_hypothesis(
#     hypothesis  = 'High-frequency patients (6+ visits) are predominantly chronic condition patients',
#     test_method = 'Cross-tab visit frequency segment vs ICD code category (chronic vs acute)',
#     data_needed = 'Current dataset is sufficient',
#     priority    = 'HIGH'
# )
# add_hypothesis(
#     hypothesis  = 'Early-onset T2D patients (31-45) have higher comorbidity rates than older T2D patients',
#     test_method = 'Compare secondary ICD codes across age groups',
#     data_needed = 'Secondary diagnosis codes needed — not in current extract',
#     priority    = 'HIGH'
# )

pd.DataFrame(hypotheses).to_csv(f'{OUTPUT_DIR}/hypotheses.csv', index=False)
```

## Step 3.2 — Analytical Assumptions Log

```python
# Document every assumption being made — if any is wrong, the analysis breaks
assumptions = []

def add_assumption(assumption, risk_if_wrong, how_to_validate, status):
    """
    status: '✓ Validated' | '⚠️ Unvalidated' | '🔴 Known Issue'
    """
    assumptions.append({
        'assumption':       assumption,
        'risk_if_wrong':    risk_if_wrong,
        'how_to_validate':  how_to_validate,
        'status':           status
    })

# ALWAYS include these baseline assumptions:
add_assumption(
    assumption      = 'Dataset is complete — no rows were filtered before this extract',
    risk_if_wrong   = 'Sample is not representative — models will not generalise',
    how_to_validate = 'Ask data owner: is this all records or a sample/filter?',
    status          = '⚠️ Unvalidated'
)
add_assumption(
    assumption      = 'All IDs are stable — same entity always gets the same ID across time',
    risk_if_wrong   = 'Entity history aggregation will mix records from different entities',
    how_to_validate = 'Check if demographic fields (DOB, sex) are consistent per ID across all rows',
    status          = '⚠️ Unvalidated'
)

## Step 3.4 — Strategic Business Simulations (Profit vs. Satisfaction)

```python
# ALWAYS run these simulations if business improvement is the goal:

# 1. The Value-for-Money (VFM) Matrix
# Identify which price-tiers yield the highest satisfaction (Rating)
df_clean['price_tier'] = pd.qcut(df_clean['Price'].rank(method='first'), q=4, labels=['Economy', 'Standard', 'Premium', 'Luxury'])
vfm = df_clean.groupby(['price_tier', 'Genre'], observed=True)['User Rating'].mean().unstack()
print("STRATEGIC INSIGHT: Value-for-Money Heatmap Data")
print(vfm.to_string())

# 2. Portfolio Optimization (60/40 Rule Check)
# Calculate the volume vs. engagement (Reviews) trade-off
portfolio = df_clean.groupby('Genre').agg(
    volume_share=('Name', 'count'),
    engagement_share=('Reviews', 'sum')
)
portfolio['volume_pct'] = portfolio['volume_share'] / portfolio['volume_share'].sum()
portfolio['engagement_pct'] = portfolio['engagement_share'] / portfolio['engagement_share'].sum()
print("\nSTRATEGIC INSIGHT: Portfolio Engagement vs Volume")
print(portfolio.to_string())

# 3. New Entrant (Innovation) vs. Veteran Stability
# Compare satisfaction and engagement for new vs. repeat entities
# (Requires logic from the specific dataset, e.g., Author first appearance)
```

## Step 3.5 — The 'Profit-Satisfaction' Matrix

Final recommendations must balance the 'So-What' (Insight) with the 'How-Much' (Profit):
- **Golden Goose:** High Satisfaction + High Margin (Protect quality)
- **Engine:** High Satisfaction + Low Margin (Maximize volume/cross-sell)
- **Risk Zone:** Low Satisfaction + High Margin (Address the 'Value Gap' or lower price)
## Step 3.6 — Sustainability & Risk Audit

Every strategic recommendation must pass the 'Long-Term Win' test:
1. **Identify the Tactical Win:** (e.g., $0.99 for volume)
2. **Identify the Strategic Risk:** (e.g., Brand devaluation, margin erosion)
3. **Define the Pivot Point:** (e.g., Exit Phase 1 at 500 reviews or Day 60)
4. **Pros vs. Cons Table:** Must be generated for any pricing or portfolio shift.

```python
# PROS / CONS Generator for Strategy:
# 1. Short-term impact on Sales Velocity
# 2. Long-term impact on Customer Life-Time Value (LTV)
# 3. Impact on Brand Perception and 'Satisfied' Customer Base
```


```python
print("REPRESENTATIVENESS ASSESSMENT")
print("─" * 50)

# Sample size adequacy
n = len(df_clean)
if n < 100:   print(f"🔴 {n} rows — too small for reliable statistical inference")
elif n < 1000: print(f"🟡 {n} rows — limited; results may not be statistically robust")
else:          print(f"✓ {n} rows — adequate for most analyses")

# Class balance (if target exists)
if target_col and target_col in df_clean.columns:
    balance = df_clean[target_col].value_counts(normalize=True)
    min_class = balance.min()
    if min_class < 0.1:
        print(f"🔴 Severe class imbalance — minority class is {min_class:.1%} of data")
        print(f"   → Use class_weight='balanced' or SMOTE before modelling")
    elif min_class < 0.2:
        print(f"🟡 Moderate class imbalance — minority class is {min_class:.1%}")

# Temporal coverage
if date_cols:
    days = (df_clean[date_cols[0]].max() - df_clean[date_cols[0]].min()).days
    print(f"Date coverage: {days} days ({days/365:.1f} years)")
    if days < 365: print("⚠️  Less than 1 year — seasonal patterns may not be captured")
```

---

# PHASE 4 — CLEANING, OUTPUT & RECOMMENDATIONS

## Step 4.0 — Cleaning Operations Catalogue

All cleaning on `df_clean` only. Log every step. Never modify `df_original`.

```python
cleaning_log = []

def log_step(operation, column, detail, rows_before, rows_after):
    cleaning_log.append({
        'operation': operation, 'column': column, 'detail': detail,
        'rows_before': rows_before, 'rows_after': rows_after,
        'rows_removed': rows_before - rows_after,
        'timestamp': pd.Timestamp.now().isoformat()
    })

def safe_clean(df, col, fn, name):
    if col not in df.columns:
        print(f"⚠️  '{col}' not found. Available: {list(df.columns)}"); return df
    try:
        result = fn(df, col); print(f"✓ {name} on '{col}'"); return result
    except Exception as e:
        print(f"⚠️  {name} failed on '{col}': {e}"); return df

# ── APPROVED OPERATIONS ─────────────────────────────────────────

# drop_duplicates
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
log_step('drop_duplicates','all',f'Removed {before-len(df_clean)} exact duplicate rows',before,len(df_clean))

# impute_median — numeric, skewed, missing < 30%
# col = 'YOUR_COLUMN'
# med = df_clean[col].median()
# df_clean[col] = df_clean[col].fillna(med)
# log_step('impute_median', col, f'Filled NaN with median {med:.4f}', len(df_clean), len(df_clean))

# impute_mean — numeric, normal distribution, missing < 20%
# col = 'YOUR_COLUMN'
# mn = df_clean[col].mean()
# df_clean[col] = df_clean[col].fillna(mn)
# log_step('impute_mean', col, f'Filled NaN with mean {mn:.4f}', len(df_clean), len(df_clean))

# impute_mode — categorical, missing < 30%
# col = 'YOUR_COLUMN'
# mode = df_clean[col].mode()[0]
# df_clean[col] = df_clean[col].fillna(mode)
# log_step('impute_mode', col, f'Filled NaN with mode "{mode}"', len(df_clean), len(df_clean))

# impute_constant — categorical, 'Unknown' is valid business category
# col = 'YOUR_COLUMN'
# df_clean[col] = df_clean[col].fillna('Unknown')
# log_step('impute_constant', col, 'Filled NaN with "Unknown"', len(df_clean), len(df_clean))

# drop_column — missing > 60%, zero variance, ID-like not needed
# col = 'YOUR_COLUMN'
# df_clean = df_clean.drop(columns=[col])
# log_step('drop_column', col, 'Column removed', len(df_clean), len(df_clean))

# cap_outliers_iqr — outlier % < 10%, likely data entry errors
# col = 'YOUR_COLUMN'
# q1,q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
# iqr = q3-q1; lo,hi = q1-1.5*iqr, q3+1.5*iqr
# n = ((df_clean[col]<lo)|(df_clean[col]>hi)).sum()
# df_clean[col] = df_clean[col].clip(lo,hi)
# log_step('cap_outliers_iqr', col, f'Capped {n} values to [{lo:.4f},{hi:.4f}]', len(df_clean), len(df_clean))

# fix_dtype — date or numeric stored as string
# col = 'YOUR_COLUMN'
# df_clean[col] = pd.to_datetime(df_clean[col], format='mixed', errors='coerce')
# log_step('fix_dtype', col, 'Converted string → datetime', len(df_clean), len(df_clean))

# standardise_labels — inconsistent casing
# col = 'YOUR_COLUMN'
# df_clean[col] = df_clean[col].str.strip().str.title()
# log_step('standardise_labels', col, 'Normalised to Title Case', len(df_clean), len(df_clean))

# flag_duplicates — DO NOT DROP in regulated industries (healthcare, finance, legal)
# Use flagging instead of dropping — let domain experts decide
# col = 'YOUR_ID_COLUMN'
# df_clean['is_duplicate'] = df_clean.duplicated(subset=[col], keep='first')
# log_step('flag_duplicates', col, f'Flagged {df_clean["is_duplicate"].sum()} rows — NOT dropped', len(df_clean), len(df_clean))
```

## Step 4.1 — Before/After Comparison & Export

```python
print("BEFORE vs AFTER CLEANING")
print(f"{'Metric':<35} {'Before':>10} {'After':>10}")
print("-"*57)
print(f"{'Rows':<35} {len(df_original):>10,} {len(df_clean):>10,}")
print(f"{'Columns':<35} {len(df_original.columns):>10} {len(df_clean.columns):>10}")
print(f"{'Missing values':<35} {df_original.isna().sum().sum():>10,} {df_clean.isna().sum().sum():>10,}")
print(f"{'Duplicate rows':<35} {df_original.duplicated().sum():>10,} {df_clean.duplicated().sum():>10,}")

# Save — never overwrite source
output_path = f'{OUTPUT_DIR}/cleaned_{os.path.basename(filepath)}'
df_clean.to_csv(output_path, index=False)
pd.DataFrame(cleaning_log).to_csv(f'{OUTPUT_DIR}/cleaning_log.csv', index=False)
print(f"✓ Cleaned file saved: {output_path}")
print(f"✓ Original file untouched")
```

## Step 4.2 — Prioritised Recommendations

```python
# Final output — ranked by business impact and effort
recommendations = []

def add_rec(priority, severity, issue, fix, blocks, effort):
    """
    severity: '🔴 CRITICAL' | '🟠 HIGH' | '🟡 MEDIUM' | '🟢 LOW'
    effort:   'Low' | 'Medium' | 'High'
    """
    recommendations.append({
        'priority': priority, 'severity': severity,
        'issue': issue, 'fix': fix, 'blocks': blocks, 'effort': effort
    })

# Populate from insights and anomaly flags found earlier
# EXAMPLE:
# add_rec(1,'🔴 CRITICAL','5 age/DOB mismatches','Send records to clinical team',
#         'ALL age-based analysis','Low — already identified')
# add_rec(2,'🔴 CRITICAL','Unit of analysis undefined','Stakeholder alignment meeting',
#         'Model design, feature engineering, metrics','Low — stakeholder conversation')

recs_df = pd.DataFrame(recommendations)
if len(recs_df):
    recs_df.to_csv(f'{OUTPUT_DIR}/prioritised_recommendations.csv', index=False)

    # Visualise
    fig, ax = plt.subplots(figsize=(13, max(4, len(recs_df)*0.7)))
    colors_map = {'🔴 CRITICAL':'#ef4444','🟠 HIGH':'#f97316','🟡 MEDIUM':'#f59e0b','🟢 LOW':'#10b981'}
    bar_colors  = [colors_map.get(r,'#94a3b8') for r in recs_df['severity']]
    labels = [f"#{r['priority']} {r['issue'][:45]}..." if len(r['issue'])>45 else f"#{r['priority']} {r['issue']}" for _, r in recs_df.iterrows()]
    ax.barh(labels[::-1], recs_df['priority'].max()+1-recs_df['priority'].iloc[::-1].values,
            color=bar_colors[::-1], alpha=0.85, height=0.6)
    ax.set_title('Prioritised Recommendations by Business Impact', fontweight='bold')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=v,label=k) for k,v in colors_map.items()], fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/prioritised_recommendations.png', dpi=150); plt.close()
```

## Step 4.3 — Full EDA Report

```python
report = f"""# EDA Report — {os.path.basename(filepath)}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Problem Context
[Fill in from Step 0.0 — stakeholder conversation]

## Dataset Summary
- Rows: {len(df_original):,} → {len(df_clean):,} after cleaning
- Columns: {len(df_original.columns)} → {len(df_clean.columns)} after cleaning
- Date range: [fill from temporal analysis]
- Sample representativeness: [fill from Step 3.3]

## Key Insights
[Paste top 5 insights from insights_and_impact.csv]

## Critical Issues Requiring Human Decision
[Paste priority 1-2 items from prioritised_recommendations.csv]

## Analytical Assumptions
[Paste unvalidated assumptions — these need stakeholder sign-off]

## Cleaning Operations Applied
[Auto-generated from cleaning_log.csv]

## Recommended Next Steps
1. [From recommendations]
2. [Stakeholder questions to resolve]
3. [Additional data needed]

## Output Files
- cleaned_[filename].csv — cleaned dataset
- cleaning_log.csv — full audit trail
- anomaly_flags.csv — data quality issues
- insights_and_impact.csv — insights with business impact
- assumptions_log.csv — assumptions requiring validation
- prioritised_recommendations.csv — ranked action items
- hypotheses.csv — testable hypotheses for next steps
- [charts] — distributions, correlations, temporal, segmentation
"""
with open(f'{OUTPUT_DIR}/eda_report.md','w') as f:
    f.write(report)
print("✓ eda_report.md saved")
```

---

# Output Files Checklist

Every pipeline run must produce ALL of these:

```
eda_outputs/
├── eda_report.md                     ← human-readable summary for stakeholders
├── problem_definition.md             ← Phase 0 context document
├── cleaning_log.csv                  ← every operation with timestamp
├── anomaly_flags.csv                 ← data quality issues with severity
├── insights_and_impact.csv           ← insights with business impact
├── assumptions_log.csv               ← assumptions needing validation
├── hypotheses.csv                    ← testable hypotheses for next steps
├── prioritised_recommendations.csv   ← ranked action items
├── cleaned_<original_filename>.csv   ← cleaned data (NEVER overwrites source)
├── missing_values.png
├── distributions_numeric.png
├── distributions_categorical.png
├── correlation_matrix.png
├── temporal_analysis.png             ← if date columns exist
├── cohort_segmentation.png           ← if categorical segments exist
├── entity_frequency_segmentation.png ← if longitudinal/repeat-ID data exists
└── prioritised_recommendations.png   ← visual priority chart
```

---

# Quick Reference — Analyst Decision Guide

```
CLEANING DECISIONS:
  Missing > 60%          → drop_column
  Missing 10-60% (num)   → impute_median (skewed) or impute_mean (normal)
  Missing 10-60% (cat)   → impute_mode or impute_constant('Unknown')
  Missing < 10%          → impute or drop_rows — your call
  Outliers > 10%         → INVESTIGATE before capping — may be real signal
  Outliers < 10%         → cap_outliers_iqr
  High skew (|s|>2)      → log_transform for modelling only
  Duplicates (regulated) → flag_duplicates NEVER silent drop
  Duplicates (non-reg)   → drop_duplicates after confirming safe

INSIGHT QUALITY CHECK — before writing any insight, ask:
  1. Is this obvious? (if yes — not an insight, it's a description)
  2. Would a stakeholder change a decision based on this? (if no — cut it)
  3. Can it be actioned? (if no — add what data is needed to make it actionable)
  4. Who owns the action? (if unclear — it won't get done)

DATA SAFETY RULES (non-negotiable):
  1. df_original = df.copy() immediately — never modify it
  2. All cleaning on df_clean only
  3. Output file must have different name/path from input file
  4. Log every operation — no silent changes
  5. In regulated industries: flag don't drop, document don't assume
  6. Cloud/DB: read-only credentials on source, write to staging schema only
```

---

# Error Handling Patterns

The agent must handle these failure modes gracefully — never crash silently:

| Error | Cause | Fix |
|---|---|---|
| `UnicodeDecodeError` | Non-UTF-8 file | Retry with `encoding='latin-1'` |
| `EmptyDataError` | File is empty | Abort with clear message — do not continue |
| `DataFrame empty after cleaning` | Over-aggressive row drops | Abort, show which step caused it, suggest loosening criteria |
| `Column not found` | Typo in column name | Print available columns and ask user to confirm |
| `All values NaN after imputation` | Column was entirely missing | Flag and skip imputation, recommend drop instead |
| Mixed dtypes in numeric col | Currency/comma formatting | Apply `fix_dtype` before profiling |
| `pd.to_datetime` fails silently | Inconsistent date formats | Use `format='mixed'` and check coerced NaN count after |

```python
# Safe wrapper — use for ALL cleaning steps
def safe_clean(df, col, fn, name):
    if col not in df.columns:
        print(f"⚠️  '{col}' not found. Available: {list(df.columns)}")
        return df
    try:
        result = fn(df, col)
        print(f"✓ {name} applied to '{col}'")
        return result
    except Exception as e:
        print(f"⚠️  {name} failed on '{col}': {e}")
        return df   # always return original df on failure — never leave None

# Guard against empty dataframe after cleaning
if len(df_clean) == 0:
    raise ValueError("⛔ df_clean is empty after cleaning steps. "
                     "Review drop operations — likely too aggressive. "
                     "Restore from df_original and retry.")

# Guard against date conversion producing all NaT
for dc in df_clean.select_dtypes(include='datetime').columns:
    nat_count = df_clean[dc].isna().sum()
    if nat_count > len(df_clean) * 0.1:
        print(f"⚠️  {dc}: {nat_count} NaT values after datetime conversion — check date format")
```

---

# Gemini Vision Integration Points

At these steps, save the plot and pass the PNG back to Gemini for visual interpretation.
This is unique to Gemini CLI — use it at every chart step.

| Step | Plot saved | Question to ask Gemini |
|---|---|---|
| Step 1.1 | `missing_values.png` | "Which columns have critical missing data patterns?" |
| Step 2.0 | `distributions_numeric.png` | "Are any distributions heavily skewed or bimodal?" |
| Step 2.1 | `distributions_categorical.png` | "Are there dominant categories or class imbalances?" |
| Step 2.2 | `temporal_analysis.png` | "Are there unusual spikes, drops, or gaps in the timeline?" |
| Step 2.3 | `cohort_segmentation.png` | "Which segment stands out as most important to investigate?" |
| Step 2.4 | `correlation_matrix.png` | "Which feature clusters show multicollinearity concerns?" |
| Step 2.5 | `entity_frequency_segmentation.png` | "Is the frequency distribution healthy or heavily skewed?" |
| Step 4.2 | `prioritised_recommendations.png` | "Do these priorities look right given what you know about the domain?" |

```python
# Pattern for Gemini vision calls — after saving any chart:
# gemini.send_message([
#     {"type": "image", "path": f"{OUTPUT_DIR}/chart_name.png"},
#     {"type": "text",  "text": "Describe the key patterns and anomalies in this chart."}
# ])
# Gemini CLI supports multimodal in the same session — leverage this at every plot step.
```

---

# Phase 2 Roadmap — Database & Cloud Connections

See original SKILL.md v1.0 for BigQuery, Snowflake, and Postgres connection
patterns, read-only credential setup, raw→staging→prod schema pattern,
Snowflake Time Travel, and dbt model structure.

Key safety rules remain unchanged:
  - Never write to raw/source schema
  - Use read-only service accounts for profiling
  - All cleaning outputs go to staging schema
  - Every transformation is a versioned SQL or Python file in Git
