import pytest
import numpy as np
from simulation import MonteCarloEngine, run_simulation

@pytest.fixture
def base_inputs():
    return {
        'initial_portfolio_value': 1_000_000,
        'initial_cost_basis': 800_000,
        'annual_spending': 40_000,
        'monthly_passive_income': 0,
        'portfolio_annual_return': 0.10,
        'portfolio_annual_std_dev': 0.15,
        'annual_dividend_yield': 0.02,
        'margin_loan_annual_avg_interest_rate': 0.05,
        'margin_loan_annual_interest_rate_std_dev': 0.01,
        'brokerage_margin_limit': 0.50,
        'federal_tax_free_gain_limit': 80_000,
        'tax_harvesting_profit_threshold': 0.10,
        'num_simulations': 100,
        'return_distribution_model': 'Normal',
        'interest_rate_distribution_model': 'Normal',
        'enable_margin_investing': False,
        'margin_investing_buffer': 0.10
    }

def test_initialization(base_inputs):
    """Verify engine initializes with correct shapes and values."""
    engine = MonteCarloEngine(base_inputs)
    
    assert engine.num_simulations == 100
    assert engine.long_term_value.shape == (100,)
    assert np.all(engine.long_term_value == 1_000_000)
    assert engine.short_term_values.shape == (100, 12)
    assert len(engine.decision_history) == 0

def test_step_year_mechanics(base_inputs):
    """Verify basic stepping functionality."""
    engine = MonteCarloEngine(base_inputs)
    
    annual_inputs = {
        'annual_spending': base_inputs['annual_spending'],
        'enable_margin_investing': False,
        'margin_investing_buffer': 0.10
    }
    
    # Step Year 0
    stats = engine.step_year(annual_inputs, 0)
    
    # Check decision history
    assert len(engine.decision_history) == 1
    assert engine.decision_history[0]['year'] == 0
    assert engine.decision_history[0]['annual_spending'] == 40_000
    
    # Check stats structure
    expected_keys = [
        'avg_net_worth', 'min_net_worth', 'max_net_worth', 
        'pct_survived', 'avg_market_return', 'avg_margin_rate'
    ]
    for k in expected_keys:
        assert k in stats
        
    # Check net worth history update
    # Should have 1 entry (array of N)
    assert len(engine.full_net_worth_history) == 12 # 12 months simulated
    # Or does it append monthly? 
    # Logic: "for m in range(12): ... self.full_net_worth_history.append(nw.copy())"
    # So yes, 12 entries per year.

def test_margin_logic(base_inputs):
    """Verify margin investing increases loan."""
    # Control: No margin
    np.random.seed(42)
    engine_control = MonteCarloEngine(base_inputs)
    annual_inputs_control = {'annual_spending': 40000, 'enable_margin_investing': False}
    engine_control.step_year(annual_inputs_control, 0)
    final_loan_control = engine_control.margin_loan.mean()
    
    # Test: Margin enabled
    np.random.seed(42) # Same seed for market returns
    engine_margin = MonteCarloEngine(base_inputs)
    # Give a buffer that allows borrowing (limit 50% - buffer 10% = 40% target. Current loan 0. Should borrow.)
    annual_inputs_margin = {
        'annual_spending': 40000, 
        'enable_margin_investing': True,
        'margin_investing_buffer': 0.10
    }
    engine_margin.step_year(annual_inputs_margin, 0)
    final_loan_margin = engine_margin.margin_loan.mean()
    
    assert final_loan_margin > final_loan_control, "Margin investing should increase loan balance"

def test_events_apply_shock(base_inputs):
    """Verify events impact the result."""
    # Control run
    np.random.seed(42)
    engine_control = MonteCarloEngine(base_inputs)
    annual_inputs = {'annual_spending': 40000}
    stats_control = engine_control.step_year(annual_inputs, 0)
    
    # Shock run
    np.random.seed(42)
    engine_shock = MonteCarloEngine(base_inputs)
    # Market shock -50%
    events = [{'type': 'market_shock', 'magnitude': -0.50}]
    stats_shock = engine_shock.step_year(annual_inputs, 0, events=events)
    
    assert stats_shock['avg_net_worth'] < stats_control['avg_net_worth']
    # Check simple math roughly: (1 + r - 0.5) vs (1 + r)
    # The shock is added to the log return or raw return? 
    # The code was: random_returns += market_shock. 
    # random_returns are roughly 10% annual / 12 ~ 0.8% monthly. 
    # -50% shock makes month 1 return -49%. Significant drop.
    
    # Also test expense shock
    np.random.seed(42)
    engine_expense = MonteCarloEngine(base_inputs)
    events_exp = [{'type': 'expense_shock', 'amount': 200_000}] # Huge expense
    stats_exp = engine_expense.step_year(annual_inputs, 0, events=events_exp)
    
    assert stats_exp['avg_net_worth'] < stats_control['avg_net_worth']

def test_run_simulation_wrapper(base_inputs):
    """Verify the backward-compatible wrapper."""
    base_inputs['num_simulations'] = 50 # speed up
    results, all_sims = run_simulation(base_inputs)
    
    # Check results dict
    assert 'avg_net_worth' in results
    assert len(results['avg_net_worth']) == 120 # 10 years * 12 months
    
    # Check history array
    # Expected (N, 120) list of lists
    assert len(all_sims) == 50
    assert len(all_sims[0]) == 120
