
import numpy as np
import matplotlib.pyplot as plt

def run_simulation(inputs):
    """
    Runs the Monte Carlo retirement simulation.

    Args:
        inputs (dict): A dictionary containing all the user-defined simulation parameters.

    Returns:
        tuple: A tuple containing:
            - results (dict): A dictionary containing the aggregated simulation results.
            - all_simulations_net_worth (list): A list of lists, where each inner list
              contains the net worth for each month of a single simulation.
    """

    # --- Helper function for random number generation --- 
    def _get_random_number(model, loc, scale, df=None):
        if model == "Student's t":
            # The standard_t distribution has a variance of df/(df-2) for df > 2
            # We need to scale it to have the desired standard deviation (scale)
            if df is None or df <= 2:
                df = 5 # Fallback to a reasonable default
            scaled_std = scale / np.sqrt(df / (df - 2))
            return loc + np.random.standard_t(df) * scaled_std
        elif model == 'Laplace':
            # The Laplace distribution scale parameter 'b' is std/sqrt(2)
            return np.random.laplace(loc, scale / np.sqrt(2))
        else: # Default to Normal
            return np.random.normal(loc, scale)


    # Extract inputs from the dictionary
    initial_portfolio_value = inputs['initial_portfolio_value']
    initial_cost_basis = inputs['initial_cost_basis']
    annual_spending = inputs['annual_spending']
    monthly_passive_income = inputs['monthly_passive_income']
    portfolio_annual_return = inputs['portfolio_annual_return']
    portfolio_annual_std_dev = inputs['portfolio_annual_std_dev']
    quarterly_dividend_yield = inputs['quarterly_dividend_yield']
    margin_loan_annual_avg_interest_rate = inputs['margin_loan_annual_avg_interest_rate']
    margin_loan_annual_interest_rate_std_dev = inputs['margin_loan_annual_interest_rate_std_dev']
    brokerage_margin_limit = inputs['brokerage_margin_limit']
    federal_tax_free_gain_limit = inputs['federal_tax_free_gain_limit']
    tax_harvesting_profit_threshold = inputs['tax_harvesting_profit_threshold']
    num_simulations = inputs['num_simulations']
    # New distribution inputs
    return_distribution_model = inputs.get('return_distribution_model', 'Normal')
    return_distribution_df = inputs.get('return_distribution_df', 5)
    interest_rate_distribution_model = inputs.get('interest_rate_distribution_model', 'Normal')
    interest_rate_distribution_df = inputs.get('interest_rate_distribution_df', 5)

    # --- Simulation setup ---
    num_months = 120
    monthly_spending = annual_spending / 12
    monthly_return = (1 + portfolio_annual_return)**(1/12) - 1
    monthly_std_dev = portfolio_annual_std_dev / np.sqrt(12)

    all_simulations_net_worth = []

    for _ in range(num_simulations):
        # --- Initialize scenario variables ---
        long_term_value = initial_portfolio_value
        long_term_basis = initial_cost_basis
        # A list of 12 tuples, each representing a monthly bucket of (value, basis)
        short_term_monthly_buckets = [(0, 0)] * 12
        margin_loan = 0

        total_margin_interest_paid_this_year = 0
        gains_realized_this_year = 0
        total_dividend_income_this_year = 0

        current_annual_margin_rate = _get_random_number(
            interest_rate_distribution_model,
            margin_loan_annual_avg_interest_rate,
            margin_loan_annual_interest_rate_std_dev,
            interest_rate_distribution_df
        )

        monthly_net_worth = []

        for month in range(1, num_months + 1):
            # --- Monthly simulation loop ---

            # Step 1: Asset Aging (FIFO)
            # Age out the oldest short-term bucket into the long-term portfolio
            aged_value, aged_basis = short_term_monthly_buckets.pop(0)
            long_term_value += aged_value
            long_term_basis += aged_basis
            # Add a new empty bucket for the current month's acquisitions
            short_term_monthly_buckets.append((0, 0))

            # Step 2: Calculate Market Returns & Update Portfolio
            random_monthly_return = _get_random_number(
                return_distribution_model,
                monthly_return,
                monthly_std_dev,
                return_distribution_df
            )
            long_term_value *= (1 + random_monthly_return)
            short_term_monthly_buckets = [
                (value * (1 + random_monthly_return), basis)
                for value, basis in short_term_monthly_buckets
            ]

            # Step 3: Handle Quarterly Dividends
            total_portfolio_value = long_term_value + sum(b[0] for b in short_term_monthly_buckets)
            if month % 3 == 0:
                dividend_payment = total_portfolio_value * quarterly_dividend_yield
                margin_loan -= dividend_payment
                total_dividend_income_this_year += dividend_payment

            # Step 4: Cover Expenses & Update Margin Loan
            cash_shortfall = monthly_spending - monthly_passive_income
            margin_loan += cash_shortfall
            monthly_margin_interest = margin_loan * (current_annual_margin_rate / 12)
            margin_loan += monthly_margin_interest
            total_margin_interest_paid_this_year += monthly_margin_interest

            # Step 5: Check for Forced Selling (Deleveraging)
            short_term_value = sum(b[0] for b in short_term_monthly_buckets)
            total_portfolio_value = long_term_value + short_term_value
            margin_limit = max(0, total_portfolio_value * brokerage_margin_limit)
            if margin_loan > margin_limit:
                amount_to_sell = (margin_loan - margin_limit) / (1 - brokerage_margin_limit)
                sell_from_long_term = 0
                if long_term_value > 0:
                    sell_from_long_term = min(amount_to_sell, long_term_value)
                    gain_from_lt_sale = (sell_from_long_term / long_term_value) * (long_term_value - long_term_basis)
                    long_term_basis -= (sell_from_long_term / long_term_value) * long_term_basis
                    long_term_value -= sell_from_long_term
                    gains_realized_this_year += gain_from_lt_sale
                    margin_loan -= sell_from_long_term

                amount_to_sell_from_st = amount_to_sell - sell_from_long_term
                if amount_to_sell_from_st > 0 and short_term_value > 0:
                    # Sell from short-term buckets, LIFO (newest first)
                    for i in range(len(short_term_monthly_buckets) - 1, -1, -1):
                        if amount_to_sell_from_st <= 0:
                            break
                        bucket_value, bucket_basis = short_term_monthly_buckets[i]
                        if bucket_value > 0:
                            sell_from_this_bucket = min(amount_to_sell_from_st, bucket_value)

                            gain_from_bucket = (sell_from_this_bucket / bucket_value) * (bucket_value - bucket_basis)
                            new_bucket_basis = bucket_basis - (sell_from_this_bucket / bucket_value) * bucket_basis
                            new_bucket_value = bucket_value - sell_from_this_bucket
                            short_term_monthly_buckets[i] = (new_bucket_value, new_bucket_basis)

                            gains_realized_this_year += gain_from_bucket
                            margin_loan -= sell_from_this_bucket
                            amount_to_sell_from_st -= sell_from_this_bucket


            # Step 6: Execute End-of-Year Tax Strategy
            if month % 12 == 0:
                unrealized_long_term_gain = long_term_value - long_term_basis
                if long_term_value > 0:
                    unrealized_long_term_gain_percentage = unrealized_long_term_gain / long_term_value
                else:
                    unrealized_long_term_gain_percentage = 0


                if unrealized_long_term_gain_percentage > tax_harvesting_profit_threshold:
                    total_investment_income_so_far = gains_realized_this_year + total_dividend_income_this_year
                    gains_to_harvest = federal_tax_free_gain_limit - total_investment_income_so_far

                    if gains_to_harvest > 0 and unrealized_long_term_gain > 0:
                        value_to_harvest = gains_to_harvest / unrealized_long_term_gain_percentage
                        if value_to_harvest > long_term_value:
                            value_to_harvest = long_term_value

                        harvested_basis = (value_to_harvest / long_term_value) * long_term_basis
                        actual_gain_harvested = value_to_harvest - harvested_basis
                        long_term_value -= value_to_harvest
                        long_term_basis -= harvested_basis
                        # Add the harvested amount to the newest short-term bucket
                        newest_bucket_value, newest_bucket_basis = short_term_monthly_buckets[-1]
                        short_term_monthly_buckets[-1] = (
                            newest_bucket_value + value_to_harvest,
                            newest_bucket_basis + value_to_harvest # Stepped-up basis
                        )
                        gains_realized_this_year += actual_gain_harvested #gains_to_harvest

                # Calculate and "Pay" California Tax
                total_investment_income = gains_realized_this_year + total_dividend_income_this_year
                net_investment_income = total_investment_income - total_margin_interest_paid_this_year
                # Simplified CA tax calculation
                ca_tax_due = max(0, net_investment_income * 0.093) #net_investment_income * 0.093
                margin_loan += ca_tax_due

                # Reset annual counters and set new margin rate
                total_margin_interest_paid_this_year = 0
                gains_realized_this_year = 0
                total_dividend_income_this_year = 0
                current_annual_margin_rate = _get_random_number(
                    interest_rate_distribution_model,
                    margin_loan_annual_avg_interest_rate,
                    margin_loan_annual_interest_rate_std_dev,
                    interest_rate_distribution_df
                )

            # Step 7: Record Net Worth
            net_worth = (long_term_value + sum(b[0] for b in short_term_monthly_buckets)) - margin_loan
            monthly_net_worth.append(net_worth)

            # Stop simulation if average net worth is below zero
            # if all_simulations_net_worth and np.mean([sim[-1] for sim in all_simulations_net_worth if sim]) < 0:
            #     break

        all_simulations_net_worth.append(monthly_net_worth)

    # --- Final Aggregation ---
    max_len = max(len(sim) for sim in all_simulations_net_worth if sim)
    padded_simulations = [sim + [sim[-1]] * (max_len - len(sim)) if sim else [0] * max_len for sim in all_simulations_net_worth]

    results = {
        'max_net_worth': np.max(padded_simulations, axis=0),
        'avg_net_worth': np.mean(padded_simulations, axis=0),
        'min_net_worth': np.min(padded_simulations, axis=0)
    }

    return results, all_simulations_net_worth

def plot_results(results):
    """
    Plots the simulation results.

    Args:
        results (dict): A dictionary containing the aggregated simulation results.
    """
    months = range(1, len(results['avg_net_worth']) + 1)
    plt.figure(figsize=(12, 8))
    plt.plot(months, results['max_net_worth'], label='Max Net Worth', color='green')
    plt.plot(months, results['avg_net_worth'], label='Average Net Worth', color='blue')
    plt.plot(months, results['min_net_worth'], label='Min Net Worth', color='red')
    plt.fill_between(months, results['min_net_worth'], results['max_net_worth'], color='gray', alpha=0.2)
    plt.title('Monte Carlo Retirement Simulation')
    plt.xlabel('Months')
    plt.ylabel('Net Worth ($)')
    plt.legend()
    plt.grid(True)
    # Format y-axis to display currency
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.show()


def main():
    """
    Main function to run the simulation with default inputs and plot the results.
    """
    # --- User-Defined Inputs ---
    inputs = {
        'initial_portfolio_value': 1000000,
        'initial_cost_basis': 700000,
        'annual_spending': 120000,
        'monthly_passive_income': 1000,
        'portfolio_annual_return': 0.10,
        'portfolio_annual_std_dev': 0.19,
        'quarterly_dividend_yield': 0.01,
        'margin_loan_annual_avg_interest_rate': 0.06,
        'margin_loan_annual_interest_rate_std_dev': 0.015,
        'brokerage_margin_limit': 0.50,
        'federal_tax_free_gain_limit': 123250,
        'tax_harvesting_profit_threshold': 0.30,
        'num_simulations': 1000,
        'return_distribution_model': 'Normal',
        'return_distribution_df': 5,
        'interest_rate_distribution_model': 'Normal',
        'interest_rate_distribution_df': 5
    }

    results, _ = run_simulation(inputs)

    # --- Print and Plot Results ---
    print("--- Simulation Results ---")
    print(f"{ 'Month':<10}{'Min Net Worth':<20}{'Avg Net Worth':<20}{'Max Net Worth':<20}")
    print("-" * 70)

    stop_month = -1
    for i, avg_net_worth in enumerate(results['avg_net_worth']):
        if avg_net_worth < 0:
            stop_month = i + 1
            break

    for i in range(len(results['avg_net_worth'])):
        print(
            f"{i+1:<10}"
            f"${results['min_net_worth'][i]:<19,.2f}"
            f"${results['avg_net_worth'][i]:<19,.2f}"
            f"${results['max_net_worth'][i]:<19,.2f}"
        )

    if stop_month != -1:
        print(f"\n--- Simulation Failed ---")
        print(f"The average net worth dropped below zero in month {stop_month}.")
    else:
        final_avg_net_worth = results['avg_net_worth'][-1]
        print(f"\n--- Strategy Survived 10 Years ---")
        print(f"The average net worth after 10 years is ${final_avg_net_worth:,.2f}.")

    plot_results(results)


if __name__ == '__main__':
    main()
