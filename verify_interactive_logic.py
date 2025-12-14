
import numpy as np
from simulation import MonteCarloEngine, run_simulation

def verify():
    # Setup inputs
    inputs = {
        'initial_portfolio_value': 2000000,
        'initial_cost_basis': 1400000,
        'annual_spending': 130000,
        'monthly_passive_income': 1000,
        'portfolio_annual_return': 0.08,
        'portfolio_annual_std_dev': 0.19,
        'annual_dividend_yield': 0.02,
        'margin_loan_annual_avg_interest_rate': 0.06,
        'margin_loan_annual_interest_rate_std_dev': 0.05,
        'brokerage_margin_limit': 0.70,
        'federal_tax_free_gain_limit': 123250,
        'tax_harvesting_profit_threshold': 0.30,
        'num_simulations': 1000, # Smaller batch for speed
        'return_distribution_model': 'Normal',
        'return_distribution_df': 5,
        'interest_rate_distribution_model': 'Normal',
        'interest_rate_distribution_df': 5,
        'enable_margin_investing': False,
        'margin_investing_buffer': 0.10
    }
    
    # 1. Run via batch wrapper
    np.random.seed(42) # Seed global
    results_batch, _ = run_simulation(inputs)
    avg_nw_batch = results_batch['avg_net_worth'][-1]
    
    # 2. Run via interactive stepping
    np.random.seed(42) # Reset seed
    engine = MonteCarloEngine(inputs)
    for y in range(10):
        annual_inputs = {
            'annual_spending': inputs['annual_spending'],
            'enable_margin_investing': inputs.get('enable_margin_investing', False),
            'margin_investing_buffer': inputs.get('margin_investing_buffer', 0.10)
        }
        engine.step_year(annual_inputs, y)
        
    avg_nw_interactive = engine.full_net_worth_history[-1].mean()
    
    print(f"Batch Avg NW:       ${avg_nw_batch:,.2f}")
    print(f"Interactive Avg NW: ${avg_nw_interactive:,.2f}")
    
    # Validation
    diff = abs(avg_nw_batch - avg_nw_interactive)
    if diff < 1.0:
        print("✅ SUCCESS: Results match exactly.")
    else:
        print(f"❌ FAILURE: Results diverge by ${diff:,.2f}")
        # Note: Vectorized operations might have slight float drift vs looped, but with reset seed 
        # using the exact same logic sequence, they should be identical.
        # My new simulation logic IS the logic for both, so they must match if seed works.

if __name__ == "__main__":
    verify()
