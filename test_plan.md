# Test Plan for `simulation.py`

This plan outlines the unit tests to ensure the correctness of the `MonteCarloEngine` and the overall simulation logic.

## 1. MonteCarloEngine Initialization
- **Test**: `test_initialization`
- **Purpose**: Verify that the engine initializes with the correct arrays and state based on input dictionary.
- **Checks**:
    - `long_term_value` shape and initial values.
    - `short_term_values` shape (N, 12).
    - `decision_history` is empty.
    - Derived constants (`geometric_monthly_return`, `monthly_margin_rate`) are correct.

## 2. Step Year Logic
- **Test**: `test_step_year_mechanics`
- **Purpose**: Verify the fundamental mechanics of a single year step using deterministic seeds/mocks (if possible) or tolerance checks.
- **Checks**:
    - Net worth changes after 1 year.
    - `decision_history` grows by 1 and contains correct inputs.
    - Return value dictionary contains all expected keys (`avg_net_worth`, `pct_survived`, `avg_market_return`, `avg_margin_rate`).

- **Test**: `test_margin_logic`
- **Purpose**: Verify margin loan behavior.
- **Checks**:
    - If `enable_margin_investing` is True, `margin_loan` should increase (borrowing to invest) and `short_term_values` should reflect new assets.
    - If False, `margin_loan` should only increase due to spending shortfall (if any).

- **Test**: `test_events`
- **Purpose**: Verify that simulated events impact the calculation.
- **Checks**:
    - `market_shock`: Net worth should be significantly lower than a control run with same seed.
    - `expense_shock`: Net worth should be lower (higher spending/loan).

## 3. Run Simulation Wrapper
- **Test**: `test_run_simulation_wrapper`
- **Purpose**: Verify the backward-compatible `run_simulation` function.
- **Checks**:
    - Returns a tuple `(results, all_sims_history)`.
    - `results` dict contains `avg_net_worth`, `p99`, etc., with correct array lengths (120 months for 10 years).
    - `all_sims_history` shape is correct.

## 4. Edge Cases
- **Test**: `test_bankruptcy`
- **Purpose**: Verify behavior when net worth hits zero or negative.
- **Checks**:
    - `pct_survived` calculation accuracy.
