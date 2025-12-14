import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulation import run_simulation, MonteCarloEngine
import google.generativeai as genai

# --- Standard Simulation Function (Existing) ---
def run_and_display_simulation(
    # Core Inputs
    initial_portfolio_value, initial_cost_basis, annual_spending,
    monthly_passive_income, annual_dividend_yield, federal_tax_free_gain_limit,
    annual_return, annual_std_dev, margin_rate, margin_rate_std_dev,
    margin_limit, simulation_count, tax_harvesting_profit_threshold,
    # New Distribution Inputs
    return_dist_model, return_dist_df,
    interest_rate_dist_model, interest_rate_dist_df,
    # New Margin Investing Inputs
    enable_margin_investing, margin_investing_buffer
):
    """
    Runs the simulation and formats the output for the Gradio interface.
    """
    inputs = {
        'initial_portfolio_value': initial_portfolio_value,
        'initial_cost_basis': initial_cost_basis,
        'annual_spending': annual_spending,
        'monthly_passive_income': monthly_passive_income,
        'portfolio_annual_return': annual_return / 100,
        'portfolio_annual_std_dev': annual_std_dev / 100,
        'annual_dividend_yield': annual_dividend_yield / 100,
        'margin_loan_annual_avg_interest_rate': margin_rate / 100,
        'margin_loan_annual_interest_rate_std_dev': margin_rate_std_dev / 100,
        'brokerage_margin_limit': margin_limit / 100,
        'federal_tax_free_gain_limit': federal_tax_free_gain_limit,
        'tax_harvesting_profit_threshold': tax_harvesting_profit_threshold / 100,
        'num_simulations': int(simulation_count),
        # New distribution params
        'return_distribution_model': return_dist_model,
        'return_distribution_df': return_dist_df,
        'interest_rate_distribution_model': interest_rate_dist_model,
        'interest_rate_distribution_df': interest_rate_dist_df,
        'enable_margin_investing': enable_margin_investing,
        'margin_investing_buffer': margin_investing_buffer / 100
    }

    results, _ = run_simulation(inputs)
    
    # Plotting and summary generation logic reused from original
    # (Simplified for brevity as it's identical logic)
    fig = create_plot(results)
    summary_title, summary_text = create_summary(results)
    df = create_dataframe(results)
    
    display_df = df.copy()
    for col in display_df.columns:
        if col != 'Month':
            display_df[col] = display_df[col].map('${:,.2f}'.format)

    return (
        gr.update(visible=True), # results_box
        summary_title,
        summary_text,
        fig,
        display_df,
        df,
        gr.update(open=True) # monthly_data_accordion
    )

def get_gemini_analysis(api_key, summary_text, df):
    """Analyzes the simulation results using the Gemini Pro API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash') # Using faster model

        system_prompt = '''You are a helpful financial analyst assistant.'''
        user_query = f"""
        Here are the results of a 10-year retirement simulation. Please provide a brief analysis.
        
        **Simulation Results Summary:**
        {summary_text}
        
        **Detailed Monthly Data:**
        {df.to_string()}
        """
        
        prompt = f"{system_prompt}\n\n{user_query}"
        response = model.generate_content(prompt, stream=True)
        output = ""
        for chunk in response:
            output += chunk.text
            yield output

    except Exception as e:
        yield f"An error occurred: {e}"

# --- Helper Functions for UI ---
def create_summary(results):
    final_avg = results['avg_net_worth'][-1]
    if final_avg < 0:
        return "## 🔴 Strategy Failed", "Average net worth dropped below zero."
    return "## ✅ Strategy Survived", f"Average net worth after 10 years: **${final_avg:,.0f}**"

def create_plot(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    months = range(1, len(results['avg_net_worth']) + 1)
    
    # Plot lines
    ax.plot(months, results['max_net_worth'], label='Max', color='#ffa600', linestyle='--')
    ax.plot(months, results['p99_net_worth'], label='99th', color='#ffa600', linestyle=':')
    ax.plot(months, results['p75_net_worth'], label='75th', color='#ffa600')
    ax.plot(months, results['median_net_worth'], label='Median', color='#bc5090', linewidth=2)
    ax.plot(months, results['avg_net_worth'], label='Average', color='#003f5c', linewidth=2)
    ax.plot(months, results['p25_net_worth'], label='25th', color='#ff6361')
    ax.plot(months, results['p01_net_worth'], label='1st', color='#ff6361', linestyle=':')
    ax.plot(months, results['min_net_worth'], label='Min', color='#ff6361', linestyle='--')
    
    ax.fill_between(months, results['p25_net_worth'], results['p75_net_worth'], color='gray', alpha=0.4)
    ax.set_title('Net Worth Projection')
    ax.set_xlabel('Month')
    ax.set_ylabel('Net Worth ($)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:,.0f}k'))
    plt.tight_layout()
    return fig

def create_dataframe(results):
    months = range(1, len(results['avg_net_worth']) + 1)
    return pd.DataFrame({
        'Month': months,
        'Min': results['min_net_worth'],
        '1st': results['p01_net_worth'],
        '25th': results['p25_net_worth'],
        'Median': results['median_net_worth'],
        'Avg': results['avg_net_worth'],
        '75th': results['p75_net_worth'],
        '99th': results['p99_net_worth'],
        'Max': results['max_net_worth']
    })

# --- Interactive Mode Logic ---

def init_interactive_mode(
    initial_portfolio_value, initial_cost_basis, annual_spending,
    monthly_passive_income, annual_dividend_yield, federal_tax_free_gain_limit,
    annual_return, annual_std_dev, margin_rate, margin_rate_std_dev,
    margin_limit, simulation_count, tax_harvesting_profit_threshold,
    return_dist_model, return_dist_df,
    interest_rate_dist_model, interest_rate_dist_df
):
    inputs = {
        'initial_portfolio_value': initial_portfolio_value,
        'initial_cost_basis': initial_cost_basis,
        'annual_spending': annual_spending,
        'monthly_passive_income': monthly_passive_income,
        'portfolio_annual_return': annual_return / 100,
        'portfolio_annual_std_dev': annual_std_dev / 100,
        'annual_dividend_yield': annual_dividend_yield / 100,
        'margin_loan_annual_avg_interest_rate': margin_rate / 100,
        'margin_loan_annual_interest_rate_std_dev': margin_rate_std_dev / 100,
        'brokerage_margin_limit': margin_limit / 100,
        'federal_tax_free_gain_limit': federal_tax_free_gain_limit,
        'tax_harvesting_profit_threshold': tax_harvesting_profit_threshold / 100,
        'num_simulations': int(simulation_count),
        'return_distribution_model': return_dist_model,
        'return_distribution_df': return_dist_df,
        'interest_rate_distribution_model': interest_rate_dist_model,
        'interest_rate_distribution_df': interest_rate_dist_df
    }
    
    engine = MonteCarloEngine(inputs)
    
    return (
        engine, 
        0, # Year
        gr.update(visible=True), # Game Container
        "## 🏁 Simulation Started\nYear 0 Complete. Ready for Year 1.",
        "No events yet.",
        create_empty_plot(),
        gr.update(visible=False) # Hide setup, show game
    )

def step_interactive_year(engine, year, pending_spending, pending_margin_enable, pending_margin_buffer):
    if year >= 10:
        return engine, year, "## 🏁 Simulation Complete", "You have finished the 10-year simulation.", create_plot_from_engine(engine)

    # 1. Generate Random Events
    events = []
    event_msg = "No massive events this year."
    if np.random.random() < 0.10:
        shock = np.random.uniform(-0.25, -0.10)
        events.append({'type': 'market_shock', 'magnitude': shock})
        event_msg = f"⚠️ **MARKET SHOCK!** A sudden crash caused an additional {shock:.1%} drop in the market."
    
    # 2. Step Engine
    annual_inputs = {
        'annual_spending': pending_spending,
        'enable_margin_investing': pending_margin_enable,
        'margin_investing_buffer': pending_margin_buffer / 100 
    }
    
    stats = engine.step_year(annual_inputs, year, events)
    year += 1
    
    # 3. Create Status String
    avg_nw = stats['avg_net_worth']
    survived = stats['pct_survived'] * 100
    
    status_md = f"## Year {year} Complete\n"
    status_md += f"**Avg Net Worth:** ${avg_nw:,.0f}  |  **Survival Chance:** {survived:.1f}%"
    
    # 4. Plot
    fig = create_plot_from_engine(engine)
    
    return engine, year, status_md, event_msg, fig

def create_empty_plot():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "Simulation initialized...", ha='center')
    return fig

def create_plot_from_engine(engine):
    history_arr = np.array(engine.full_net_worth_history) # (TotalMonths, N)
    if history_arr.size == 0:
        return create_empty_plot()
        
    months = range(1, history_arr.shape[0] + 1)
    
    # Calculate stats
    avg = np.mean(history_arr, axis=1)
    median = np.median(history_arr, axis=1)
    p25 = np.percentile(history_arr, 25, axis=1)
    p75 = np.percentile(history_arr, 75, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(months, avg, label='Average', color='#003f5c', linewidth=2)
    ax.plot(months, median, label='Median', color='#bc5090')
    ax.fill_between(months, p25, p75, color='gray', alpha=0.3, label='IQR')
    
    ax.set_title("Live Net Worth Projection")
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:,.0f}k'))
    return fig


# --- GRADIO UI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft(), css=".input-card {border-top: 3px solid #6c5ce7}") as demo:
    gr.Markdown("# 🚀 Retirement Simulator 2.0")
    
    with gr.Tabs():
        # TAB 1: STANDARD (Existing)
        with gr.Tab("Standard Simulation"):
            with gr.Row():
                initial_portfolio_value = gr.Number(value=2000000, label="Portfolio Value")
                initial_cost_basis = gr.Number(value=1400000, label="Cost Basis")
                annual_spending = gr.Number(value=130000, label="Annual Spending")
            with gr.Row():
                monthly_passive_income = gr.Number(value=1000, label="Monthly Passive Income")
                annual_dividend_yield = gr.Number(value=2.0, label="Dividend Yield (%)")
                federal_tax_free_gain_limit = gr.Number(value=123250, label="Tax Free Gain Limit")
            
            with gr.Accordion("Advanced Market Settings", open=False):
                with gr.Row():
                    annual_return = gr.Number(value=8, label="Return (%)")
                    annual_std_dev = gr.Number(value=19, label="Std Dev (%)")
                    margin_rate = gr.Number(value=6, label="Margin Rate (%)")
                    margin_rate_std_dev = gr.Number(value=5, label="Margin Rate Std Dev (%)")
                    margin_limit = gr.Number(value=70, label="Margin Limit (%)")
                with gr.Row():
                    simulation_count = gr.Number(value=1000, label="Simulations")
                    tax_harvesting_profit_threshold = gr.Number(value=30, label="Harvest Threshold (%)")
            
            with gr.Accordion("Distribution Models", open=False):
                return_dist_model = gr.Radio(['Normal', "Student's t", 'Laplace'], value='Laplace', label="Return Dist")
                return_dist_df = gr.Number(value=5, label="Return DF")
                interest_rate_dist_model = gr.Radio(['Normal', "Student's t", 'Laplace'], value='Laplace', label="Interest Dist")
                interest_rate_dist_df = gr.Number(value=5, label="Interest DF")
                
            with gr.Accordion("Margin Investing (Static)", open=False):
                enable_margin_investing = gr.Checkbox(label="Enable Margin Investing")
                margin_investing_buffer = gr.Slider(0, 50, value=10, label="Buffer (%)")
                
            btn_run_std = gr.Button("Run Standard Simulation", variant="primary")
            
            with gr.Group(visible=False) as std_results_box:
                std_summary_title = gr.Markdown()
                std_summary_text = gr.Markdown()
                std_plot = gr.Plot()
                std_accordion = gr.Accordion("Data", open=False)
                with std_accordion:
                    std_df = gr.Dataframe()
                    
                with gr.Accordion("Gemini Analysis", open=False):
                    gemini_key = gr.Textbox(label="API Key", type="password")
                    btn_analyze = gr.Button("Analyze Results")
                    gemini_out = gr.Markdown()

            btn_run_std.click(
                run_and_display_simulation,
                inputs=[initial_portfolio_value, initial_cost_basis, annual_spending, monthly_passive_income,
                        annual_dividend_yield, federal_tax_free_gain_limit, annual_return, annual_std_dev,
                        margin_rate, margin_rate_std_dev, margin_limit, simulation_count, tax_harvesting_profit_threshold,
                        return_dist_model, return_dist_df, interest_rate_dist_model, interest_rate_dist_df,
                        enable_margin_investing, margin_investing_buffer],
                outputs=[std_results_box, std_summary_title, std_summary_text, std_plot, std_df, gr.State(), std_accordion]
            )
            
            btn_analyze.click(
                get_gemini_analysis,
                inputs=[gemini_key, std_summary_text, std_df],
                outputs=[gemini_out]
            )

        # TAB 2: INTERACTIVE (New)
        with gr.Tab("Game Mode 🎮"):
            gr.Markdown("Step year-by-year and adjust your strategy based on market conditions.")
            
            # Interactive State
            engine_state = gr.State()
            year_state = gr.State(0)
            
            with gr.Group(visible=True) as game_setup_box:
                gr.Markdown("### Initial Configuration")
                gr.Markdown("*Uses configuration from 'Standard Simulation' tab.*")
                btn_start_game = gr.Button("Start Simulation", variant="primary")

            with gr.Group(visible=False) as game_container:
                
                with gr.Row():
                    with gr.Column(scale=1):
                        game_status_md = gr.Markdown("## Ready")
                        game_event_log = gr.Markdown("No events.")
                        
                        gr.Markdown("### Decisions for Next Year")
                        cur_spending = gr.Number(value=130000, label="Annual Spending ($)")
                        cur_margin_enable = gr.Checkbox(label="Enable Margin Investing")
                        cur_margin_buffer = gr.Slider(0, 50, value=10, label="Buffer (%)")
                        
                        btn_next_year = gr.Button("Simulate Next Year ➡️")
                        
                    with gr.Column(scale=2):
                        live_plot = gr.Plot()
            
            btn_start_game.click(
                init_interactive_mode,
                inputs=[
                    initial_portfolio_value, initial_cost_basis, annual_spending, monthly_passive_income,
                    annual_dividend_yield, federal_tax_free_gain_limit, annual_return, annual_std_dev,
                    margin_rate, margin_rate_std_dev, margin_limit, simulation_count, tax_harvesting_profit_threshold,
                    return_dist_model, return_dist_df, interest_rate_dist_model, interest_rate_dist_df
                ],
                outputs=[engine_state, year_state, game_container, game_status_md, game_event_log, live_plot, game_setup_box]
            )
            
            btn_next_year.click(
                step_interactive_year,
                inputs=[engine_state, year_state, cur_spending, cur_margin_enable, cur_margin_buffer],
                outputs=[engine_state, year_state, game_status_md, game_event_log, live_plot]
            )

if __name__ == "__main__":
    demo.launch()
