# Product Requirements Document (PRD): Retirement Simulator "Game Mode"

**Version:** 1.0
**Date:** 2025-12-14
**Status:** Draft / Design Input

## 1. Product Overview
The **Retirement Simulator** is a dual-platform (iOS & Web) application designed to help users model and understand complex retirement strategies, specifically focusing on the "Buy, Borrow, Die" strategy (living off margin loans secured by a portfolio).

The core innovation is the **Interactive Game Mode**, which transforms a static Monte Carlo simulation into a year-by-year decision-making experience. Users "play" through their retirement, reacting to market shocks and adjusting their spending dynamically, rather than just setting static parameters at the start.

## 2. Target Audience
*   **Retirees & Pre-retirees**: Individuals evaluating the risks of margin-based flexible spending strategies.
*   **Financial Enthusiasts**: Users who want to "stress test" their intuition against market volatility.

## 3. Core Feature Sets

### A. Standard Simulation (Batch Mode)
*   **Input**: Static parameters (Portfolio Value, Annual Spending, Margin Limit, Volatility, etc.).
*   **Execution**: Runs 1,000+ Monte Carlo simulations instantly.
*   **Output**:
    *   Success/Failure probability.
    *   Net Worth Trajectory Chart (Min, Max, Avg, Percentiles).
    *   **AI Analysis**: A summary report powered by Google Gemini, assessing risk and providing personalized recommendations.
    *   **Settings**: Users can input their own Gemini API Key for privacy.

### B. Interactive "Game Mode" (The New Core Experience)
This is the primary focus for the new UI design.

#### **User Flow**
1.  **Setup**: User configures initial portfolio and difficulty settings (Market Volatility, Interest Rates).
2.  **Game Loop (Year 1 to Year X)**:
    *   **Status Display**: Shows current Year, Net Worth, "Survival Probability" (based on remaining simulations), and current Market Conditions.
    *   **Decision Phase**: User sets parameters for the *upcoming* year:
        *   *Annual Spending*: Slider/Input (e.g., $80k - $200k).
        *   *Enable Margin Investing*: Toggle (Risk-on behavior).
        *   *Margin Buffer*: Slider (Safety margin).
    *   **Action**: User clicks "Simulate Next Year".
    *   **Result**: The engine simulates 12 months.
    *   **Event Feedback**: App reports significant events (e.g., "Market Crash! Portfolio down 20%", "Inflation Spike! Expenses up 5%").
3.  **Corrective Action (The "Time Machine")**:
    *   **Undo/Retry**: If a user creates a bad outcome (e.g., bankruptcy), they can "Undo" the last year.
    *   **Comparison View**: When retrying, the UI allows comparing the *Previous Decision* vs. *Current Decision* to see how changes affected the outcome.
    *   **AI "What If" Analysis**: Gemini analyzes the difference between the two attempts (e.g., "By reducing spending by $10k, you avoided a forced liquidation event").
4.  **End Game**:
    *   **Success**: Reached target years with Net Worth > 0.
    *   **Failure**: Net Worth hits $0 (Bankruptcy).

## 4. Design Requirements & UI Elements

### 4.1. Navigation
*   **Tab-Based Navigation**:
    1.  **Standard Mode**: For quick data modeling.
    2.  **Game Mode**: For the interactive experience.
    3.  **Settings**: For API Key management.

### 4.2. Game Mode UI Components
*   **Dashboard Header**:
    *   **Year Counter**: Large, clear display (e.g., "Year 5 of 30").
    *   **Health Meter**: Visual indicator of portfolio health (Green/Yellow/Red) or "Probability of Survival".
    *   **Net Worth**: Big number currency display.
*   **Chart Area**:
    *   **Dynamic Line Chart**: Plots Net Worth history. Updates incrementally as the user advances years.
*   **Control Panel (Action Area)**:
    *   **Spending Slider**: Interactive slider with immediate visual feedback on the projected burn rate.
    *   **Margin Controls**: Toggles and sliders for advanced users.
    *   **"Simulate Year" Button**: Prominent primary action button.
*   **Feedback/Event Stream**:
    *   **Toast/Cards**: Popups for "Market Shock" or "Margin Call Warning".
    *   **Analysis Box**: A collapsable area showing Gemini's commentary on the year's performance.

### 4.3. Comparison UI (Specific Requirement)
*   **Split View or Overlay**: When retrying a year, show "Attempt 1" (Failed) vs "Attempt 2" (Current) metrics side-by-side.
    *   *Metrics*: Spending, Margin Rate, Realized Loss.
*   **AI Insight**: A dedicated text block for Gemini to explain *why* the new strategy worked better.

### 4.4. Onboarding & API Key
*   **API Key Input**: A secure field in settings or the standard simulation view.
*   **Instructional Overlay**: A helper popup explaining how to get a Google Gemini API Key from AI Studio.

## 5. Technical Specifications (Built Today)

### 5.1. iOS Client (`RetirementSimulatorApp`)
*   **Framework**: SwiftUI (iOS 16.0+).
*   **Architecture**: MVVM.
*   **Network Layer**: `NetworkService.swift` using `URLSession` to talk to Python backend.
*   **State Management**: `ObservableObject` ViewModels tracking the game state.

### 5.2. Python Backend (`Retirement-Sim`)
*   **Framework**: FastAPI.
*   **Core Logic**: `MonteCarloEngine` class (supports stateful stepping).
*   **API Endpoints**:
    *   `POST /game/start`: Init session.
    *   `POST /game/step`: Run one year.
    *   `POST /game/undo`: Rewind state.
    *   `POST /simulate`: Batch run.
    *   `POST /ai-analysis`: Gemini integration.
*   **AI Provider**: `google-generativeai` (Gemini Flash model).

## 6. Future Considerations for Design
*   **Gamification**: Badges for "Surviving 1929 Crash" or "Zero Margin Call Run".
*   **Accessibility**: High contrast modes for graph readability.
*   **Localization**: The AI Advisor already supports Multi-language (En/Zh/Ja); UI should reflect this.
