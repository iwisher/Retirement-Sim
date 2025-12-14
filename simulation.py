
import numpy as np
import matplotlib.pyplot as plt

class MonteCarloEngine:
    def __init__(self, inputs):
        self.inputs = inputs
        self.num_simulations = inputs['num_simulations']
        
        # --- State Initialization (Vectorized for N simulations) ---
        # Long-term portfolio
        self.long_term_value = np.full(self.num_simulations, inputs['initial_portfolio_value'], dtype=np.float64)
        self.long_term_basis = np.full(self.num_simulations, inputs['initial_cost_basis'], dtype=np.float64)
        
        # Short-term buckets: Shape (N, 12). Column 0 is oldest (Asset Aging), Column 11 is newest.
        self.short_term_values = np.zeros((self.num_simulations, 12), dtype=np.float64)
        self.short_term_basis = np.zeros((self.num_simulations, 12), dtype=np.float64)
        
        # Margin Loan & Carryover
        self.margin_loan = np.zeros(self.num_simulations, dtype=np.float64)
        self.net_investment_loss_carryover = np.zeros(self.num_simulations, dtype=np.float64)
        
        # History
        self.full_net_worth_history = [] # Will store (N,) array for each month
        
        # Pre-calculate constant params
        self.portfolio_annual_return = inputs['portfolio_annual_return']
        self.portfolio_annual_std_dev = inputs['portfolio_annual_std_dev']
        
        # Adjust monthly return for volatility drag
        self.geometric_monthly_return = (1 + self.portfolio_annual_return)**(1/12) - 1
        self.monthly_std_dev = self.portfolio_annual_std_dev / np.sqrt(12)
        self.monthly_return_loc = self.geometric_monthly_return + 0.5 * (self.monthly_std_dev**2)
        
        self.margin_loan_annual_avg_interest_rate = inputs['margin_loan_annual_avg_interest_rate']
        self.margin_loan_annual_interest_rate_std_dev = inputs['margin_loan_annual_interest_rate_std_dev']
        
        self.monthly_margin_rate = (1 + self.margin_loan_annual_avg_interest_rate)**(1/12) - 1
        self.monthly_margin_rate_std_dev = self.margin_loan_annual_interest_rate_std_dev / np.sqrt(12)
        
        self.return_distribution_model = inputs.get('return_distribution_model', 'Normal')
        self.return_distribution_df = inputs.get('return_distribution_df', 5)
        self.interest_rate_distribution_model = inputs.get('interest_rate_distribution_model', 'Normal')
        self.interest_rate_distribution_df = inputs.get('interest_rate_distribution_df', 5)

    def _get_random_values(self, model, loc, scale, df, size):
        if model == "Student's t":
            if df is None or df <= 2: df = 5
            scaled_std = scale / np.sqrt(df / (df - 2))
            return loc + np.random.standard_t(df, size=size) * scaled_std
        elif model == 'Laplace':
             return np.random.laplace(loc, scale / np.sqrt(2), size=size)
        else: # Normal
            return np.random.normal(loc, scale, size=size)

    def step_year(self, annual_inputs, year_index, events=None):
        """
        Runs the simulation for 12 months (one year).
        annual_inputs: dict with keys 'annual_spending', 'enable_margin_investing', 'margin_investing_buffer'
        events: list of dicts, e.g., {'type': 'market_shock', 'magnitude': -0.2}
        """
        monthly_spending = annual_inputs['annual_spending'] / 12
        enable_margin_investing = annual_inputs.get('enable_margin_investing', False)
        margin_investing_buffer = annual_inputs.get('margin_investing_buffer', 0.10)
        
        monthly_passive_income = self.inputs['monthly_passive_income']
        annual_dividend_yield = self.inputs['annual_dividend_yield']
        brokerage_margin_limit = self.inputs['brokerage_margin_limit']
        
        # Event processing
        market_shock = 0
        expense_shock = 0
        # Event processing
        market_shock = 0
        expense_shock = 0
        if events:
            for e in events:
                if e['type'] == 'market_shock':
                    market_shock = e.get('magnitude', 0)
                elif e['type'] == 'expense_shock':
                    expense_shock = e.get('amount', 0)
                    
        # Stats accumulation
        self.year_market_returns = np.zeros(self.num_simulations)
        self.year_margin_rates = np.zeros(self.num_simulations)


        # Loop 12 months
        for m in range(12):
            month_abs = year_index * 12 + m + 1
            
            # --- Step 1: Asset Aging (FIFO) ---
            # Oldest bucket (col 0) moves to long term
            aged_val = self.short_term_values[:, 0]
            aged_basis = self.short_term_basis[:, 0]
            self.long_term_value += aged_val
            self.long_term_basis += aged_basis
            
            # Shift everything left
            self.short_term_values[:, :-1] = self.short_term_values[:, 1:]
            self.short_term_basis[:, :-1] = self.short_term_basis[:, 1:]
            # Clear newest bucket
            self.short_term_values[:, -1] = 0
            self.short_term_basis[:, -1] = 0
            
            # --- Step 2: Market Returns ---
            random_returns = self._get_random_values(
                self.return_distribution_model, 
                self.monthly_return_loc, 
                self.monthly_std_dev, 
                self.return_distribution_df, 
                self.num_simulations
            )
            
            # Apply shock if first month (or spread it? assuming 1-time shock in month 1 for simplicity)
            if m == 0 and market_shock != 0:
                # Override random return with shock? Or add to it? 
                # Usually a shock replaces the random move or is a large drift.
                # Let's add it to the random return to keep volatility valid.
                random_returns += market_shock

            self.long_term_value *= (1 + random_returns)
            self.short_term_values *= (1 + random_returns[:, np.newaxis])
            
            # Accumulate market return (compounding)
            # We want annual return: (1+r1)*(1+r2)... - 1
            if m == 0:
                self.year_market_returns = (1 + random_returns)
            else:
                self.year_market_returns *= (1 + random_returns)

            
            # --- Step 2.5: Margin Investing ---
            if enable_margin_investing:
                total_portfolio = self.long_term_value + np.sum(self.short_term_values, axis=1)
                investing_limit = np.maximum(0, total_portfolio * (brokerage_margin_limit - margin_investing_buffer))
                borrowable = np.maximum(0, investing_limit - self.margin_loan)
                
                # Invest borrowing into newest bucket
                self.margin_loan += borrowable
                self.short_term_values[:, -1] += borrowable
                self.short_term_basis[:, -1] += borrowable

            # --- Step 3: Dividends (Quarterly) ---
            if (month_abs) % 3 == 0:
                total_portfolio = self.long_term_value + np.sum(self.short_term_values, axis=1)
                divs = total_portfolio * (annual_dividend_yield / 4)
                self.margin_loan -= divs
                # Track divs for tax (simplified: accumulated annual var not needed for state logic yet? 
                # verify_interactive_logic might need exact match. Original accumulates `total_dividend_income_this_year`)
                # We'll need to track this if we implement the tax calc at year end.
                # Let's add an instance var for current year accumulation if not already.
                if not hasattr(self, 'current_year_divs'): self.current_year_divs = np.zeros(self.num_simulations)
                self.current_year_divs += divs
            
            # --- Step 4: Expenses & Margin Interest ---
            current_spending = monthly_spending
            if m == 0 and expense_shock != 0:
                current_spending += expense_shock
                
            shortfall = current_spending - monthly_passive_income
            self.margin_loan += shortfall
            
            margin_rates = self._get_random_values(
                self.interest_rate_distribution_model,
                self.monthly_margin_rate,
                self.monthly_margin_rate_std_dev,
                self.interest_rate_distribution_df,
                self.num_simulations
            )
            interest = self.margin_loan * margin_rates
            self.margin_loan += interest
            
            # Accumulate margin rate (average)
            # Simple average of monthly rates * 12 is approx annual APR
            self.year_margin_rates += margin_rates * 12

            
            if not hasattr(self, 'current_year_interest'): self.current_year_interest = np.zeros(self.num_simulations)
            self.current_year_interest += interest

            # --- Step 5: Forced Selling (Deleveraging) ---
            total_short = np.sum(self.short_term_values, axis=1)
            total_value = self.long_term_value + total_short
            
            # Safety check to avoid div by zero
            # If total_value is <= 0, we are essentially bust, loan is effectively infinite ratio.
            # But let's follow logic:
            limit = np.maximum(0, total_value * brokerage_margin_limit)
            excess = self.margin_loan - limit
            mask_sell = excess > 0 # Boolean array of sims that need to sell
            
            if np.any(mask_sell):
                # Vectorized selling is tricky with logic branches.
                # However, since mask_sell is subset, we can operate on subset or use `np.where`.
                # Given complexity of "LT then ST LIFO", maybe iterate just the masked indices?
                # Or use vectorized math with masks.
                
                # Amount to sell
                amt = np.zeros(self.num_simulations)
                amt[mask_sell] = excess[mask_sell] / (1 - brokerage_margin_limit)
                
                # Sell LT
                # available LT
                lt_avail = self.long_term_value
                sell_lt = np.minimum(amt, lt_avail)
                sell_lt = np.maximum(0, sell_lt) # clamp
                
                # Calc gains from LT sale
                ratio_lt = np.zeros(self.num_simulations)
                # avoid div/0
                mask_lt_pos = lt_avail > 0
                ratio_lt[mask_lt_pos] = sell_lt[mask_lt_pos] / lt_avail[mask_lt_pos]
                
                gain_lt = ratio_lt * (self.long_term_value - self.long_term_basis)
                
                self.long_term_value -= sell_lt
                self.long_term_basis -= (ratio_lt * self.long_term_basis)
                self.margin_loan -= sell_lt
                
                if not hasattr(self, 'current_year_gains'): self.current_year_gains = np.zeros(self.num_simulations)
                self.current_year_gains += gain_lt
                
                # Sell ST (LIFO)
                amt_remaining = amt - sell_lt
                # Iterate buckets backwards 11 -> 0
                for i in range(11, -1, -1):
                    # Mask for those who still need to sell
                    mask_st = amt_remaining > 1e-9 # float tolerance
                    if not np.any(mask_st):
                        break
                        
                    b_val = self.short_term_values[:, i]
                    b_basis = self.short_term_basis[:, i]
                    
                    sell_st = np.minimum(amt_remaining, b_val)
                    sell_st[~mask_st] = 0 # only sell if needed
                    
                    ratio_st = np.zeros(self.num_simulations)
                    mask_b_pos = b_val > 0
                    ratio_st[mask_b_pos] = sell_st[mask_b_pos] / b_val[mask_b_pos]
                    
                    gain_st = ratio_st * (b_val - b_basis)
                    
                    self.short_term_values[:, i] -= sell_st
                    # basis update
                    self.short_term_basis[:, i] -= (ratio_st * b_basis)
                    
                    self.margin_loan -= sell_st
                    self.current_year_gains += gain_st
                    amt_remaining -= sell_st

            # --- Record Net Worth ---
            nw = (self.long_term_value + np.sum(self.short_term_values, axis=1)) - self.margin_loan
            # Copy to history
            self.full_net_worth_history.append(nw.copy())

        # --- End of Year: Tax Strategy ---
        self._run_tax_strategy()
        
        # --- Return Stats for the Year ---
        # Current NW (last month)
        current_nw = self.full_net_worth_history[-1]
        
        # Calculate realized stats for the year
        # Market Return: Convert cumulative multiplier to percentage change
        avg_market_return = np.mean(self.year_market_returns - 1) * 100
        
        # Margin Rate: Average of annualized monthly rates divided by 12 months (we summed them)
        avg_margin_rate = np.mean(self.year_margin_rates / 12) * 100
        
        return {
            'avg_net_worth': np.mean(current_nw),
            'min_net_worth': np.min(current_nw),
            'max_net_worth': np.max(current_nw),
            'median_net_worth': np.median(current_nw),
            'p25': np.percentile(current_nw, 25),
            'p75': np.percentile(current_nw, 75),
            'pct_survived': np.mean(current_nw > 0),
            'avg_market_return': avg_market_return,
            'avg_margin_rate': avg_margin_rate
        }


    def _run_tax_strategy(self):
        # Unpack params
        threshold = self.inputs['tax_harvesting_profit_threshold']
        limit = self.inputs['federal_tax_free_gain_limit']
        
        # Ensure accumulator arrays exist
        if not hasattr(self, 'current_year_gains'): self.current_year_gains = np.zeros(self.num_simulations)
        if not hasattr(self, 'current_year_divs'): self.current_year_divs = np.zeros(self.num_simulations)
        if not hasattr(self, 'current_year_interest'): self.current_year_interest = np.zeros(self.num_simulations)
        
        # 1. Gain Harvesting
        unrealized = self.long_term_value - self.long_term_basis
        ratio = np.zeros(self.num_simulations)
        mask_pos = self.long_term_value > 0
        ratio[mask_pos] = unrealized[mask_pos] / self.long_term_value[mask_pos]
        
        # Identify who harvests
        mask_harvest = (ratio > threshold) & (unrealized > 0)
        
        income_so_far = self.current_year_gains + self.current_year_divs
        room = limit - income_so_far
        
        # Only if room > 0
        mask_harvest &= (room > 0)
        
        if np.any(mask_harvest):
            # value to harvest = room / ratio
            # Must not exceed LT value
            val_to_harvest = np.zeros(self.num_simulations)
            val_to_harvest[mask_harvest] = room[mask_harvest] / ratio[mask_harvest]
            
            # clamp to actual LT value
            val_to_harvest = np.minimum(val_to_harvest, self.long_term_value)
            
            # Basis of harvested portion
            # Basis of harvested portion
            basis_harvested = np.zeros(self.num_simulations)
            # Only calculate where LT value is positive to avoid div/0
            mask_calc = mask_harvest & (self.long_term_value > 1e-9)
            
            basis_harvested[mask_calc] = (val_to_harvest[mask_calc] / self.long_term_value[mask_calc]) * self.long_term_basis[mask_calc]
            # fix potential nan if lt_value is 0 (though mask_pos handles it?)
            basis_harvested[np.isnan(basis_harvested)] = 0
            
            real_gain = val_to_harvest - basis_harvested
            
            # Execute: Remove from LT, Add to ST (newest)
            self.long_term_value[mask_harvest] -= val_to_harvest[mask_harvest]
            self.long_term_basis[mask_harvest] -= basis_harvested[mask_harvest]
            
            self.short_term_values[:, -1][mask_harvest] += val_to_harvest[mask_harvest]
            # Stepped up basis!
            self.short_term_basis[:, -1][mask_harvest] += val_to_harvest[mask_harvest] 
            
            self.current_year_gains[mask_harvest] += real_gain[mask_harvest]

        # 2. Variable State Tax (CA)
        total_income = self.current_year_gains + self.current_year_divs
        net_inc = total_income - self.current_year_interest
        
        taxable = net_inc + self.net_investment_loss_carryover
        tax = np.maximum(0, taxable * 0.093)
        self.margin_loan += tax
        
        # Carryover update
        self.net_investment_loss_carryover = np.minimum(0, taxable)
        # If positive, it resets to 0 (implied by minimum(0, ...)? No, if taxable > 0, carryover is 0)
        # Logic: if taxable < 0, carryover = taxable. Else 0.
        self.net_investment_loss_carryover = np.where(taxable < 0, taxable, 0)
        
        # Reset accumulations
        self.current_year_gains[:] = 0
        self.current_year_divs[:] = 0
        self.current_year_interest[:] = 0


# Wrapper for backwards compatibility
def run_simulation(inputs):
    engine = MonteCarloEngine(inputs)
    
    # 10 years
    for y in range(10):
        # Standard inputs per year
        yr_inputs = {
            'annual_spending': inputs['annual_spending'],
            'enable_margin_investing': inputs.get('enable_margin_investing', False),
            'margin_investing_buffer': inputs.get('margin_investing_buffer', 0.10)
        }
        engine.step_year(yr_inputs, y)
        
    # Aggregate results
    # engine.full_net_worth_history is List of (N,) arrays. len = 120.
    # We want [ [m1, m2...], ... ] for each sim?
    # No, existing returns:
    # results (dict), all_simulations_net_worth (list of lists)
    
    # Convert history: List of (N,) -> (120, N) -> (N, 120)
    history_arr = np.array(engine.full_net_worth_history) # shape (120, N)
    all_sims_nw = history_arr.T.tolist() # shape (N, 120) -> list of lists
    
    # Compute Aggregates
    # Pad if failed? logic in original code handles "early failure" by breaking loop for that sim?
    # No, original code: if avg < 0 breaks.
    # Note: Vectorized approach runs all 120 months regardless of negative NW.
    # "padded_simulations" in original code implies some might be shorter?
    # Original: `if all_simulations_net_worth and np.mean(...) < 0: break`. That breaks the *whole* simulation loop (all sims).
    # My engine runs full 120 months.
    
    # Check max len:
    padded = np.array(all_sims_nw) # (N, 120) unless jagged? It's not jagged here.
    
    results = {
        'max_net_worth': np.max(padded, axis=0),
        'p99_net_worth': np.percentile(padded, 99, axis=0),
        'p75_net_worth': np.percentile(padded, 75, axis=0),
        'median_net_worth': np.median(padded, axis=0),
        'avg_net_worth': np.mean(padded, axis=0),
        'p25_net_worth': np.percentile(padded, 25, axis=0),
        'p01_net_worth': np.percentile(padded, 1, axis=0),
        'min_net_worth': np.min(padded, axis=0)
    }
    
    return results, all_sims_nw

if __name__ == '__main__':
    # Default execution
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
        'num_simulations': 10000,
        'return_distribution_model': 'Laplace',
        'return_distribution_df': 5,
        'interest_rate_distribution_model': 'Laplace',
        'interest_rate_distribution_df': 5,
        'enable_margin_investing': False,
        'margin_investing_buffer': 0.10
    }
    
    results, _ = run_simulation(inputs)
    # Simple print as before...
    # (Copied from original main)
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
