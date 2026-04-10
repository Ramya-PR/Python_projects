# A/B Testing History & Validation Log
## Project: Consumer Shopping Trends 2026

This log tracks all hypotheses suggested and tested to optimize shopping preferences and spend across segments.

---

### TEST 1: Premium Hybrid Growth (BOPIS)
- **Hypothesis:** Adding a "Reserve In-Store" (BOPIS) option for high-value items will increase total conversion for the Premium segment.
- **Segment:** Premium Spenders (High Tech-Savvy + High Need for Touch/Feel).
- **Metric:** Conversion Rate (Completed vs. Abandoned).
- **Design:** 50/50 split (Control: Store/Online only | Variant: Added Reserve-In-Store).
- **Status:** **WIN (Statistically Significant)**
- **Results:** 
  - Control Conversion: 85.9%
  - Variant Conversion: 91.5% (Uplift: +5.6%)
  - P-Value: < 0.0001
- **Business Action:** Roll out "Reserve Online, Experience In-Store" feature for the Premium segment.

---

### TEST 2: Value Online Trust (Speed & Trust)
- **Hypothesis:** Reducing delivery messaging from 4 to 2 days and adding trust seals will increase average monthly spend for Value shoppers.
- **Segment:** Value Spenders (Price sensitive, high online interest).
- **Metric:** Average Monthly Spend ($).
- **Design:** 50/50 split (Control: Standard UI | Variant: Speed & Trust messaging).
- **Status:** **LEARNING (Not Statistically Significant)**
- **Results:**
  - Control Avg Spend: $81,751.18
  - Variant Avg Spend: $81,942.26 (Uplift: negligible)
  - P-Value: 0.8331
- **Business Action:** Do not roll out. Iterate on a new hypothesis focused on direct price incentives for this segment.

---

### NEXT TEST (Proposed): Value Price Sensitivity
- **Hypothesis:** Targeted "Bundle & Save" offers will be more effective than "Speed & Trust" for the Value segment.
- **Status:** Pending refinement based on Price Sensitivity Analysis.

--- 
### TEST 3: Targeted Basket-Value Rewards
- **Hypothesis:** Targeted rewards (e.g., Free Shipping on $500+) for 'High-Potential' Value shoppers will increase total monthly spend.
- **Segment:** High-Potential Value Spenders (High orders + high discount sensitivity).
- **Metric:** Average Monthly Spend ($).
- **Status:** **LEARNING (Not Significant)**
- **Results:**
  - Control Avg Spend: $81969.05
  - Variant Avg Spend: $80495.67
  - P-Value: 0.2557
- **Business Action:** Roll out targeted basket-value rewards for the High-Potential Value segment.
