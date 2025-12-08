import numpy as np

def test_fix():
    # Parameters
    annual_return = 0.08
    annual_std_dev = 0.20
    num_years = 10
    num_months = num_years * 12
    num_simulations = 10000

    # Derived monthly parameters
    monthly_return = (1 + annual_return)**(1/12) - 1
    monthly_std_dev = annual_std_dev / np.sqrt(12)

    # PROPOSED FIX: Adjust the mean for volatility drag
    # We want the Geometric Mean to be 'monthly_return'.
    # Arithmetic Mean (mu) should be approx Geometric Mean + Variance/2
    adjusted_monthly_return = monthly_return + 0.5 * (monthly_std_dev**2)

    print(f"Target Annual Return (CAGR): {annual_return:.4f}")
    print(f"Monthly Return (Geometric): {monthly_return:.6f}")
    print(f"Adjusted Monthly Mean (Arithmetic): {adjusted_monthly_return:.6f}")

    # Simulation with adjusted mean
    final_values = []
    for _ in range(num_simulations):
        value = 1.0
        for _ in range(num_months):
            # Using Normal distribution with ADJUSTED mean
            r = np.random.normal(adjusted_monthly_return, monthly_std_dev)
            value *= (1 + r)
        final_values.append(value)

    # Calculate Realized CAGR
    avg_final_value = np.mean(final_values)
    median_final_value = np.median(final_values)
    
    realized_cagr_mean = (avg_final_value)**(1/num_years) - 1
    realized_cagr_median = (median_final_value)**(1/num_years) - 1

    print(f"Realized CAGR (from Median Final Value) with FIX: {realized_cagr_median:.4f}")
    
    error = abs(realized_cagr_median - annual_return)
    print(f"Error: {error:.4f}")
    
    if error < 0.005: # Allow small margin of error due to randomness
        print("SUCCESS: Fix aligns Median CAGR with Target.")
    else:
        print("FAILURE: Fix does not align Median CAGR with Target.")

if __name__ == "__main__":
    test_fix()
