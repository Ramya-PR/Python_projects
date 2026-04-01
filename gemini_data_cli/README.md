# Amazon Bestsellers Analysis (2009-2019): The Profit-Satisfaction Matrix

## Project Overview
This project provides a comprehensive EDA and strategic analysis of the **Amazon Bestsellers** dataset, exploring the relationships between genre, pricing, and customer satisfaction to drive business growth.

### Key Insights & Findings
- **The "Bestseller Paradox":** High ratings (4.9) don't guarantee long-term survival; most multi-year bestsellers sit in the 4.0-4.5 "Utility" range.
- **Engagement Gap:** Fiction books drive **73% more reviews** on average than Non-Fiction, despite being 37% less expensive.
- **The Value Gap:** Fiction titles priced between $10 and $20 see a significant drop in customer satisfaction (4.51 rating).
- **Author Dominance:** Just **10% of bestseller entries** are dominated by a small group of repeat high-performers like Jeff Kinney and Gary Chapman.

### Strategic Recommendations
1. **Tiered Pricing Architecture:** Use $9.99 for Fiction (to drive volume) and $19.99 for Non-Fiction (to maximize authority/margin).
2. **Review Harvesting:** Launch new authors at a "Discovery Price" ($4.99) to build the social proof needed for long-term survival.
3. **The 60/40 Portfolio:** Maintain a mix of 60% Non-Fiction (Profit) and 40% Fiction (Engagement).

### Data Visualizations
The project includes deep-insight visualizations:
- `price_sweet_spot_box.png`: Identifying the pricing sweet spot by genre.
- `satisfaction_price_heatmap.png`: Highlighting the relationship between pricing and user ratings.
- `bestseller_survival_curve.png`: The rating threshold needed for multi-year success.

### Technical Approach
- **Python / Pandas:** For advanced data manipulation and longitudinal analysis.
- **Seaborn / Matplotlib:** For creating business-critical visualizations.
- **Strategic Simulation:** Analysis of "Value-for-Money" and "New Entrant vs. Veteran" performance.

This dataset is imported from Kaggle: https://www.kaggle.com/datasets/obaidhere/amazon-bestsellers-price-vs-user-rating-analysis/data

---
*Created by Ramya Ponnuvel Rajathy as part of the Gemini EDA Strategic Analysis Pipeline.*
