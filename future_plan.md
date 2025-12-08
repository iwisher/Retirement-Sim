# Mobile App Future Roadmap: Comprehensive Retirement & Wealth Building

This document outlines the feature roadmap for transforming the current niche simulator into a comprehensive "Super App" for financial planning, catering to both wealth accumulation (younger users) and decumulation (retirees).

## 1. Core Engines (Simulation Modes)
Expand beyond the "Margin Strategy" to cover standard financial paths.

*   **The "Standard" Bucket Strategy**
    *   Simulate classic "Cash / Bonds / Stocks" allocations.
    *   **Sequence of Returns Risk:** Visualize impacts of market crashes immediately after retirement.
    *   **Safe Withdrawal Rate (SWR) Tester:** Test the "4% Rule" against dynamic spending rules (e.g., spending ceilings/floors).
*   **The FIRE Engine (Financial Independence, Retire Early)**
    *   **"Years to Freedom" Calculator:** Metric based on savings rate vs. expenses, not age.
    *   **Coast FIRE Calculator:** Show milestones where active saving can stop (e.g., "Save $100k by 30, then just cover costs").

## 2. The "Wealth Builder" (For Young People)
Focus on education and the "build strong" mindset.

*   **"Time Machine" Visualizer**
    *   Visual graph showing the Cost of Waiting.
    *   *Example:* "Invest $1 at 22 = $88 at 65. Invest $1 at 32 = $40 at 65."
*   **Debt Destroyer**
    *   **Avalanche vs. Snowball Simulator:** Compare interest savings (math) vs. psychological wins (behavior) for debt payoff.
*   **"Lifestyle Creep" Warning**
    *   Simulate the long-term impact of raising spending with every raise versus investing the difference.

## 3. Stress Testing & "Black Swans"
Test plans against specific, non-random events.

*   **Historical Replay**
    *   Run the user's plan through actual historical timelines (1929 Crash, 1970s Stagflation, 2000 Dot Com, 2008 Crisis).
*   **Life Event Shocks**
    *   Inject specific timeline events: Child birth, College tuition, Home purchase, Medical emergency.
*   **Inflation Resilience**
    *   Toggle between Low (2%), Medium (4%), and Hyper (8%+) inflation scenarios to test purchasing power.

## 4. Education & AI Integration
Leverage the Gemini integration for personalized guidance.

*   **The "Why" Button**
    *   Context-aware explanations of complex terms (e.g., Sharpe Ratio, Tax Drag) in simple language.
*   **Personalized "Next Best Action"**
    *   AI suggestions based on simulation failure points.
    *   *Example:* "Working 1 extra year reduces failure rate from 20% to 2%."
*   **Scenario Comparison**
    *   A/B test decisions: "Buy a luxury car now" vs. "Invest the money" -> Show the difference in final Net Worth.

## 5. Tax & Estate Optimization
*   **Roth vs. Traditional Calculator**
    *   Simulate tax implications based on current vs. estimated future tax brackets.
*   **RMD Forecaster**
    *   Visualize the "Tax Bomb" from Required Minimum Distributions (RMDs) at age 73+.

## Technical Architecture & Next Steps

### Recommended Tech Stack
*   **Frontend:** React Native (for a polished, native feel on iOS/Android) or Flutter.
*   **Backend:** Python (FastAPI/Flask) to reuse the existing heavy-math logic, or port logic to TypeScript for offline-first capability.

### Code Refactoring Needed
To support these features, `simulation.py` needs to be modularized:
1.  **MarketEngine:** Handle different return models and historical data.
2.  **TaxEngine:** Separate tax logic (Roth, Traditional, Capital Gains) from the core loop.
3.  **EventEngine:** Handle life events and dynamic spending rules.
