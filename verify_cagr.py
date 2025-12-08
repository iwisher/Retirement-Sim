import numpy as np

def test_cagr():
    # Parameters
    annual_return = 0.08
    annual_std_dev = 0.20
    num_years = 10
    num_months = num_years * 12
    num_simulations = 10000

    # Derived monthly parameters (as in simulation.py)
    monthly_return = (1 + annual_return)**(1/12) - 1
    monthly_std_dev = annual_std_dev / np.sqrt(12)

    print(f"Target Annual Return (CAGR): {annual_return:.4f}")
    print(f"Monthly Return (Geometric): {monthly_return:.6f}")
    print(f"Monthly Std Dev: {monthly_std_dev:.6f}")

    # Simulation
    final_values = []
    for _ in range(num_simulations):
        value = 1.0
        for _ in range(num_months):
            # Using Normal distribution as default
            r = np.random.normal(monthly_return, monthly_std_dev)
            value *= (1 + r)
        final_values.append(value)

    # Calculate Realized CAGR
    avg_final_value = np.mean(final_values)
    median_final_value = np.median(final_values)
    
    realized_cagr_mean = (avg_final_value)**(1/num_years) - 1
    realized_cagr_median = (median_final_value)**(1/num_years) - 1

    print(f"Average Final Value: {avg_final_value:.4f}")
    print(f"Median Final Value: {median_final_value:.4f}")
    print(f"Realized CAGR (from Mean Final Value): {realized_cagr_mean:.4f}")
    print(f"Realized CAGR (from Median Final Value): {realized_cagr_median:.4f}")
    
    # Expected Drag
    # Arithmetic Mean of Log Returns approx = mu - sigma^2/2
    # Here we are doing: value *= (1+r). 
    # E[1+r] = 1 + monthly_return.
    # So Expected Value should track (1+monthly_return)^N.
    # So the Mean Final Value should be correct (close to target).
    # BUT the Median Final Value (which represents the 'typical' outcome) will be lower due to volatility drag.
    # Geometric Mean of the distribution ~ Arithmetic Mean - Variance/2
    
    expected_drag = (annual_std_dev**2) / 2
    print(f"Expected Volatility Drag (approx sigma^2/2): {expected_drag:.4f}")
    print(f"Target - Drag: {annual_return - expected_drag:.4f}")

if __name__ == "__main__":
    test_cagr()
