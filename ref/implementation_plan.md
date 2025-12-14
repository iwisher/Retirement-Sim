# Implementation Plan - Gamified Retirement Simulation

The goal is to transform the existing retirement simulation into an interactive "game" where the user can adjust spending and margin strategies annually based on the performance of the previous year.

## User Review Required

> [!IMPORTANT]
> **Performance Consideration**: Storing the state for 8000+ simulations in Python objects (lists/dicts) might have a memory/performance cost, but given the complexity of the "short-term buckets" logic (FIFO/LIFO), a full numpy vectorization is too risky/complex for this iteration. We will stick to the Python loop approach but chunk it by year.

> [!NOTE]
> The "Game Mode" will run alongside the existing standard simulation mode in `simulation.py`, but `app.py` will primarily feature this new interactive mode or offer it as a distinct option. For now, we will integrate it as a new "Interactive Simulation" tab or replace the main flow? The user request implies replacing or enhancing the main experience. We will maintain the existing "Run Full Simulation" capability but perhaps put the Game Mode front and center or as a "Challenge" mode. *Decision: We will add a toggle or separate tab for "Interactive Mode" to keep the original functionality intact for comparison.*

## Proposed Changes

### 1. Refactor `simulation.py`

#### [MODIFY] [simulation.py](file:///d:/DevSpace/Retirement-Sim/simulation.py)

- **Create `MonteCarloEngine` Class**:
    - **`__init__(self, inputs)`**: Initializes the simulation state for `num_simulations`.
        - State for each sim: `long_term_value`, `long_term_basis`, `short_term_monthly_buckets`, `margin_loan`, `loss_carryover`, etc.
    - **`step_year(self, annual_inputs, year_index, events=None)`**:
        - Runs the simulation for 12 months for *all* simulations.
        - `annual_inputs` allows overriding `annual_spending`, `enable_margin_investing`, `margin_investing_buffer` for this specific year.
        - `events`: Optional list of events to apply this year (e.g., `{'type': 'market_shock', 'magnitude': -0.20}` or `{'type': 'expense_shock', 'amount': 50000}`).
        - Returns aggregate stats for the year (min/max/avg net worth, % simulations survived, % margin calls triggered).
    - **Backwards Compatibility**: Keep `run_simulation` but make it just instantiate `MonteCarloEngine` and call `step_year` 10 times.

- **Implement `EventEngine` (Simple implementation within `simulation.py` or new file)**:
    - Logic to apply "Shocks" during the `step_year` loop.
    - Example: If `market_shock` is active, force the random return for a specific month (or all months) to be negative.
    - Example: If `expense_shock` is active, add one-time cost to `monthly_spending` for month 1.

### 2. Update `app.py`

#### [MODIFY] [app.py](file:///d:/DevSpace/Retirement-Sim/app.py)

- **Add State Management**: Use `gr.State` to hold the `MonteCarloEngine` instance and current year index.
- **UI Redesign**:
    - Add a "Game Mode" or "Interactive Simulation" section.
    - **Initial Setup**: Similar to current inputs.
    - **Game Loop Interface**:
        - Display: "Year X Complete".
        - **[NEW] Events Log**: Show if any random events occurred (e.g., "Medical Emergency: $20k added to expenses").
        - Stats: Current Avg Net Worth, Probability of Failure (Zero Net Worth), Margin Call Risk (probability).
        - **[NEW] "The Why Button"**: Small help icons (?) next to stats explaining them (e.g., "Why is my risk high?").
        - **Guidance / Next Best Action**:
            - "Your margin loan is high (40%). Consider reducing spending."
            - "Gemini Analysis" (optional deep dive).
        - Controls: "Adjust Spending for Next Year", "Enable/Disable Margin Investing".
        - Action: "Simulate Next Year".
    - **Final Results**: Reuse the existing plot/summary logic but for the final concatenated history.

## Verification Plan

### Automated Tests
- **`verify_interactive_logic.py`**:
    - Run the simulator in "All at once" mode via `run_simulation`.
    - Run the simulator in "Step by step" mode (10 step calls) with same initial seeds/inputs.
    - Verify that the final results (Avg Net Worth history) are Identical.
    - **Verify Event Injection**:
        - Run with a specific seed and NO event.
        - Run with same seed AND an `expense_shock`.
        - Verify net worth is lower by exactly the shock amount (plus interest effects).

### Manual Verification
- **Web UI Check**:
    - Launch `app.py`.
    - Start "Interactive Mode".
    - Run Year 1. Check stats.
    - Change Spending (e.g., increase massive amount).
    - Run Year 2. Verify Net Worth drops significantly (more than standard).
    - Check "Guidance" text updates based on risk.
    - Complete 10 years and check Final Plot.
