# NotebookLM Research Plan: Retirement Simulation Project

This document outlines the strategy for using NotebookLM to deepen the understanding and development of the comprehensive financial simulation app.

## 1. Key Research Questions

These questions are designed to guide the "chat" with your documents in NotebookLM.

### **Financial Methodology & Validity**
*   "How does the 'Volatility Drag' correction implemented in the simulation compare to standard academic models for long-term geometric compounding?"
*   "What are the limitations of using a 'Random Walk' model (independent monthly returns) versus models that account for market memory or mean reversion?"
*   "How do the Student's t and Laplace distributions specifically differ from the Normal distribution in their impact on 'tail risk' (extreme failure scenarios)?"

### **Strategy & Optimization**
*   "Based on the margin simulation results, what is the correlation between the 'Margin Borrow Limit' and the rate of portfolio depletion (bankruptcy)?"
*   "In what specific market conditions (simulated by the varying distributions) does the Tax-Gain Harvesting strategy provide the most measurable benefit?"
*   "How does the 'Margin Investing' strategy's risk profile change when interest rates rise (simulated by higher mean margin rates)?"

### **User Education & Behavioral Finance**
*   "What are the most effective analogies or explanations for 'Sequence of Returns Risk' suitable for a younger audience?"
*   "What are the psychological barriers to 'Debt Avalanche' vs. 'Debt Snowball', and how can a simulation visualize the trade-off?"

## 2. Core Simulation Components for Analysis

These are the specific logic blocks you should upload or focus on within NotebookLM to get the best technical feedback.

*   **The Volatility Correction Logic:**
    *   *Focus:* The specific formula `monthly_return = geometric_monthly_return + 0.5 * (monthly_std_dev**2)`.
    *   *Goal:* Verify if there are edge cases (e.g., extremely high volatility) where this approximation breaks down.
*   **The Tax-Gain Harvesting Algorithm:**
    *   *Focus:* The step-up basis logic and the interaction with the "wash sale" rule (even if simplified).
    *   *Goal:* Ensure the logic correctly models the *deferral* of tax rather than just elimination.
*   **The Deleveraging (Forced Sale) Logic:**
    *   *Focus:* The mathematical trigger `margin_loan > margin_limit`.
    *   *Goal:* Analyze if the "instant" sale model is too optimistic compared to real-world "margin calls" which might happen at unfavorable intra-month prices.

## 3. Data Extraction Requirements (Input for NotebookLM)

To get the best results, you need to feed NotebookLM high-quality, structured text.

### **A. Code Documentation**
*   **Action:** Extract all docstrings and comments from `simulation.py` into a single text file.
*   **Why:** NotebookLM is great at explaining code intent if the comments are provided clearly.

### **B. Simulation Results Logs**
*   **Action:** Run a small set of "extreme" simulations (e.g., high volatility, high spend) and save the tabular output to a PDF or text file.
*   **Why:** You can ask NotebookLM to "Analyze the failure patterns in this simulation log."

### **C. Comparative Literature (External)**
*   **Action:** Find 2-3 PDF papers or articles on:
    *   "Safe Withdrawal Rates" (Trinity Study).
    *   "Lifecycle Investing" (Ayres & Nalebuff) - regarding young people using leverage.
    *   "Volatility Drag" (Geometric vs Arithmetic mean).
*   **Why:** Uploading these alongside your code allows NotebookLM to "Compare my project's methodology with the findings in the Trinity Study."

## 4. Suggested NotebookLM Workflow

1.  **Create a New Notebook:** Title it "Retirement Sim Engine".
2.  **Upload Sources:**
    *   `simulation.py` (The actual code).
    *   `README.md` (The project overview).
    *   `future_plan.md` (The roadmap).
    *   (Optional) External papers on Lifecycle Investing.
3.  **Start Asking:** Use the "Key Research Questions" above as your starting prompts.
