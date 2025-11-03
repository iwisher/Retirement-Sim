import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulation import run_simulation
import google.generativeai as genai


def run_and_display_simulation(
    # Core Inputs
    initial_portfolio_value, initial_cost_basis, annual_spending,
    annual_return, annual_std_dev, margin_rate, margin_rate_std_dev,
    margin_limit, simulation_count, tax_harvesting_profit_threshold,
    # New Distribution Inputs
    return_dist_model, return_dist_df,
    interest_rate_dist_model, interest_rate_dist_df
):
    """
    Runs the simulation and formats the output for the Gradio interface.
    """
    inputs = {
        'initial_portfolio_value': initial_portfolio_value,
        'initial_cost_basis': initial_cost_basis,
        'annual_spending': annual_spending,
        'monthly_passive_income': 1000,  # Hardcoded as per original UI
        'portfolio_annual_return': annual_return / 100,
        'portfolio_annual_std_dev': annual_std_dev / 100,
        'quarterly_dividend_yield': 0.01,  # Hardcoded as per original UI
        'margin_loan_annual_avg_interest_rate': margin_rate / 100,
        'margin_loan_annual_interest_rate_std_dev': margin_rate_std_dev / 100,
        'brokerage_margin_limit': margin_limit / 100,
        'federal_tax_free_gain_limit': 123250,  # Hardcoded
        'tax_harvesting_profit_threshold': tax_harvesting_profit_threshold / 100,
        'num_simulations': int(simulation_count),
        # New distribution params
        'return_distribution_model': return_dist_model,
        'return_distribution_df': return_dist_df,
        'interest_rate_distribution_model': interest_rate_dist_model,
        'interest_rate_distribution_df': interest_rate_dist_df
    }

    results, _ = run_simulation(inputs)

    # --- Create Summary ---
    stop_month = -1
    for i, avg_net_worth in enumerate(results['avg_net_worth']):
        if avg_net_worth < 0:
            stop_month = i + 1
            break

    if stop_month != -1:
        summary_title = "## 🔴 Strategy Failed"
        summary_text = f"The average net worth dropped below zero in month **{stop_month}**."
    else:
        final_avg_net_worth = results['avg_net_worth'][-1]
        summary_title = "## ✅ Strategy Survived 10 Years"
        summary_text = f"The average net worth after 10 years is **${final_avg_net_worth:,.0f}**."

    # --- Create Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    months = range(1, len(results['avg_net_worth']) + 1)
    ax.plot(months, results['max_net_worth'],
            label='Max Net Worth', color='#ffa600', linestyle='--')
    ax.plot(months, results['p75_net_worth'],
            label='75th Percentile', color='#ffa600')
    ax.plot(months, results['avg_net_worth'],
            label='Average Net Worth', color='#003f5c', linewidth=2)
    ax.plot(months, results['p25_net_worth'],
            label='25th Percentile', color='#ff6361')
    ax.plot(months, results['min_net_worth'],
            label='Min Net Worth', color='#ff6361', linestyle='--')
    ax.fill_between(months, results['p25_net_worth'],
                    results['p75_net_worth'], color='gray', alpha=0.2, label='Interquartile Range (25-75%)')
    ax.set_title('Simulated Net Worth Over 10 Years', fontsize=16)
    ax.set_xlabel('Month')
    ax.set_ylabel('Net Worth ($)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'${x/1000:,.0f}k'))
    plt.tight_layout()

    # --- Create DataFrame ---
    df = pd.DataFrame({
        'Month': months,
        'Min Net Worth': results['min_net_worth'],
        '25th Percentile': results['p25_net_worth'],
        'Avg Net Worth': results['avg_net_worth'],
        '75th Percentile': results['p75_net_worth'],
        'Max Net Worth': results['max_net_worth']
    })

    for col in ['Min Net Worth', '25th Percentile', 'Avg Net Worth', '75th Percentile', 'Max Net Worth']:
        df[col] = df[col].map('${:,.2f}'.format)

    return (
        gr.update(visible=True),
        summary_title,
        summary_text,
        fig,
        df,
        gr.update(open=True),
        gr.update(open=True)
    )


def get_gemini_analysis(
    # API Key and results
    api_key, summary_text, df,
    # Core Inputs
    initial_portfolio_value, initial_cost_basis, annual_spending,
    annual_return, annual_std_dev, margin_rate, margin_rate_std_dev,
    margin_limit, simulation_count, tax_harvesting_profit_threshold,
    # Distribution Inputs
    return_dist_model, return_dist_df,
    interest_rate_dist_model, interest_rate_dist_df
):
    """
    Analyzes the simulation results using the Gemini Pro API.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')

        system_prompt = '''
        You are a helpful financial analyst assistant. Your role is to provide a clear, concise, and neutral interpretation of Monte Carlo simulation results for a retirement plan.

        Your primary duty is data interpretation. You must not give prescriptive financial advice or use forceful commands (e.g., "you must," "you should"). Your language must remain objective.

        Structure your response as follows:

        One-Sentence Summary: Begin with a single summary sentence describing the primary outcome of the simulation (e.g., the statistical probability of the plan succeeding).

        Key Factors: Explain the key variables within the simulation (such as market volatility or withdrawal rate) that most significantly influence this outcome.

        Risk Analysis & Mitigation Strategies: If the simulation indicates a high risk of failure, first identify the primary statistical drivers causing it. After identifying the cause, neutrally present common strategies and variable adjustments used to mitigate such risks. For each strategy (e.g., "reducing withdrawals," "securing a cash buffer," or "adjusting asset allocation"), briefly explain the potential tradeoffs or impact it would have on the plan according to financial principles.

        Length: Keep the entire response to no longer than six paragraphs.
        '''

        input_summary = f"""
        **Initial Financial Situation:**
        - Initial Portfolio Value: ${initial_portfolio_value:,.2f}
        - Initial Cost Basis: ${initial_cost_basis:,.2f}
        - Annual Spending: ${annual_spending:,.2f}

        **Market and Loan Assumptions:**
        - Portfolio Annual Average Return: {annual_return}% (Distribution: {return_dist_model}, DF: {return_dist_df if return_dist_model == "Student's t" else 'N/A'})
        - Portfolio Annual Standard Deviation: {annual_std_dev}%
        - Margin Loan Annual Average Interest Rate: {margin_rate}% (Distribution: {interest_rate_dist_model}, DF: {interest_rate_dist_df if interest_rate_dist_model == "Student's t" else 'N/A'})
        - Margin Loan Annual Interest Rate Std. Dev.: {margin_rate_std_dev}%

        **Strategy Parameters:**
        - Brokerage Margin Limit: {margin_limit}% of portfolio value
        - Tax Harvesting Profit Threshold: {tax_harvesting_profit_threshold}%
        - Number of Simulations: {simulation_count}
        """

        simulation_logic = """
        **Simulation Logic Overview:**
        The simulation models a retirement strategy over 10 years, month by month.
        - **Monthly Cycle:** The portfolio grows or shrinks based on random market returns. Dividends are paid quarterly and reduce the margin loan. Monthly spending is covered by borrowing on margin. Interest accrues on the loan. If the loan exceeds the margin limit, assets are sold to deleverage.
        - **Annual Cycle:** At year-end, a tax-gain harvesting strategy is executed. It sells assets to realize gains up to the federal tax-free limit, then immediately repurchases them to step-up the cost basis. California state tax is calculated on the net investment income (dividends + gains - margin interest) and the tax bill is added to the margin loan balance for the new year.
        """

        user_query = f"""
        Here are the results of a 10-year retirement simulation. Please provide a brief analysis.

        {input_summary}

        {simulation_logic}

        **Simulation Results Summary:**
        {summary_text}

        **Detailed Monthly Data:**
        {df.to_string()}
        """

        prompt = f"{system_prompt}\n\n{user_query}"

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"


# --- Gradio UI ---
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
        .input-card {padding: 1rem; border-radius: 0.75rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1); text-align: center; border-top: 4px solid var(--color-accent);}
        .step-card {padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05); text-align: center; border-top: 4px solid; margin-bottom: 1rem;}
        .step-card-1 {border-color: #4B8BBE;}
        .step-card-2 {border-color: #F0A07B;}
        .step-card-3 {border-color: #E57333;}
        .step-card-title {font-size: 1.1rem; font-weight: 600; color: #333; margin-bottom: 0.5rem; background-color: rgba(0,0,0,0.05); padding: 0.25rem 0.5rem; border-radius: 0.25rem; display: inline-block;}
        .step-card-desc {font-size: 1rem; color: #555;}
        .arrow-down {text-align: center; font-size: 2rem; color: #ccc; margin: -0.5rem 0;}
        .summary-card {padding: 2rem; border-radius: 0.75rem; text-align: center;}
        .summary-survived {background-color: #E8F5E9; border-left: 8px solid #4CAF50;}
        .summary-failed {background-color: #FFEBEE; border-left: 8px solid #F44336;}
        .summary-title {font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem;}
        .summary-text {font-size: 1.25rem;}
    """,
) as demo:
    gr.Markdown(
        """
        <div style=\"text-align: center; font-family: 'Inter', sans-serif;">
            <h1 style=\"font-size: 3rem; font-weight: 800; color: #003f5c;">The Simulation Engine</h1>
            <p style=\"font-size: 1.25rem; color: #555; margin-top: 0.5rem;">A Visual Guide to How Your Retirement Future is Calculated</p>
        </div>
        """
    )

    with gr.Accordion("Step 1: The Starting Point - Your Financial DNA", open=True):
        gr.Markdown("<p style='text-align: center; font-size: 1.1rem; font-family: Inter, sans-serif;'>The simulation begins with your unique parameters. Change any value below and run the simulation to see how it impacts your 10-year outlook.</p>")
        with gr.Row():
            initial_portfolio_value = gr.Number(value=1000000, label="PORTFOLIO VALUE ($)",
                                                elem_classes="input-card", info="The starting value of your investment portfolio.")
            initial_cost_basis = gr.Number(value=700000, label="COST BASIS ($)", elem_classes="input-card",
                                           info="The original value of your assets for tax purposes.")
            annual_spending = gr.Number(value=120000, label="ANNUAL SPENDING ($)", elem_classes="input-card",
                                        info="The total amount of money you plan to spend annually.")
        with gr.Row():
            annual_return = gr.Number(value=10, label="AVG. ANNUAL RETURN (%)", elem_classes="input-card",
                                      info="The expected average annual return of your portfolio.")
            annual_std_dev = gr.Number(value=19, label="ANNUAL STD. DEV. (%)", elem_classes="input-card",
                                       info="The annualized standard deviation of your portfolio's returns (volatility).")
            margin_rate = gr.Number(value=6, label="AVG. MARGIN RATE (%)", elem_classes="input-card",
                                    info="The average annual interest rate on your margin loan.")
        with gr.Row():
            margin_rate_std_dev = gr.Number(value=1.5, label="MARGIN STD. DEV. (%)", elem_classes="input-card",
                                            info="The standard deviation of the margin loan interest rate.")
            margin_limit = gr.Number(value=50, label="MARGIN BORROW LIMIT (%)", elem_classes="input-card",
                                     info="The maximum percentage of your portfolio you are willing to borrow on margin.")
            simulation_count = gr.Number(value=1000, label="# OF SIMULATIONS", elem_classes="input-card",
                                         info="The number of different market scenarios to simulate.")
        with gr.Row():
            tax_harvesting_profit_threshold = gr.Number(value=30, label="TAX HARVEST PROFIT THRESHOLD (%)",
                                                        elem_classes="input-card", info="The unrealized profit percentage that triggers tax-gain harvesting.")

        with gr.Accordion("Advanced Settings: Distribution Models", open=False):
            gr.Markdown("<p style='text-align: center; font-size: 1rem; font-family: Inter, sans-serif;'>Use these settings to model for 'fat-tail' risk, where extreme market events are more likely than a Normal distribution predicts.</p>")
            with gr.Row():
                with gr.Column():
                    return_dist_model = gr.Radio(['Normal', "Student's t", 'Laplace'], label="Portfolio Return Distribution", value='Normal',
                                                 info="Choose the probability distribution for generating random market returns.")
                    return_dist_df = gr.Slider(minimum=2.1, maximum=30, value=5, step=0.1, label="Degrees of Freedom (for Student's t)",
                                               info="Lower values create 'fatter tails' (more extreme events). Only used if Student's t is selected.", visible=False)
                with gr.Column():
                    interest_rate_dist_model = gr.Radio(['Normal', "Student's t", 'Laplace'], label="Margin Rate Distribution", value='Normal',
                                                        info="Choose the probability distribution for generating random margin interest rates.")
                    interest_rate_dist_df = gr.Slider(minimum=2.1, maximum=30, value=5, step=0.1, label="Degrees of Freedom (for Student's t)",
                                                      info="Lower values create 'fatter tails'. Only used if Student's t is selected.", visible=False)

        run_button = gr.Button(
            "Run Simulation", variant="primary", scale=1, elem_id="run_button")

    with gr.Accordion("Step 2 & 3: The Monthly Cycle & Annual Reset", open=False):
        gr.Markdown(
            """
            <div style=\"text-align: center; font-family: 'Inter', sans-serif;">
                <h2 style=\"font-size: 2rem; font-weight: 700; color: #58508d;">2. The Monthly Cycle: The Engine's Core</h2>
                <p style=\"font-size: 1.1rem; color: #555;">For 120 months, each simulation runs through a precise sequence of events. This is the heart of the model, where market chances meet your financial plan month after month.</p>
            </div>
            """
        )
        with gr.Blocks():
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    <div class="step-card step-card-1">
                        <div class="step-card-title">STEP 1</div>
                        <div class="step-card-desc">
                            <strong>Market Moves & Dividends</strong><br>
                            A random market return is generated and applied to your portfolio. Every 3 months, dividends are paid out and used to reduce your margin loan.
                        </div>
                    </div>
                    """,
                )
                gr.Markdown('<div class="arrow-down">▼</div>')
                gr.Markdown(
                    """
                    <div class="step-card step-card-2">
                        <div class="step-card-title">STEP 2</div>
                        <div class="step-card-desc">
                            <strong>Covering Expenses</strong><br>
                            Your monthly spending is funded first by passive income. Any shortfall is covered by borrowing more on margin, increasing your loan balance.
                        </div>
                    </div>
                    """,
                )
                gr.Markdown('<div class="arrow-down">▼</div>')
                gr.Markdown(
                    """
                    <div class="step-card step-card-3">
                        <div class="step-card-title">STEP 3</div>
                        <div class="step-card-desc">
                            <strong>Risk Check: Margin Call</strong><br>
                            We check if your loan has exceeded your specified limit (e.g., 50%) of your portfolio's value. If so, a forced sale occurs, selling the exact amount of stock needed to bring the loan back to the limit.
                        </div>
                    </div>
                    """,
                )
        gr.Markdown(
            """
            <div style=\"text-align: center; font-family: 'Inter', sans-serif; margin-top: 2rem;">
                <h2 style=\"font-size: 2rem; font-weight: 700; color: #58508d;">3. The Annual Reset: Tax & Strategy</h2>
                <p style=\"font-size: 1.1rem; color: #555;">At the end of each simulated year, a critical series of financial maneuvers takes place. This is where your tax strategy is executed to optimize your portfolio for the year ahead.</p>
            </div>
            """
        )
        with gr.Row():
            gr.Markdown(
                """
                <div class="step-card" style="border-color: #6A5ACD;">
                    <div class="step-card-title" style="background-color: #E6E6FA;">YEAR-END 1</div>
                    <div class="step-card-desc">
                        <strong>Tax-Gain Harvesting</strong><br>
                        If your long-term unrealized gains are above 30%, we sell just enough stock to realize gains up to your $123,250 federal tax-free limit. The cash is used to buy back the stock immediately, raising your cost basis ("step-up") and reducing future taxes.
                    </div>
                </div>
                """,
            )
            gr.Markdown(
                """
                <div class="step-card" style="border-color: #6A5ACD;">
                    <div class="step-card-title" style="background-color: #E6E6FA;">YEAR-END 2</div>
                    <div class="step-card-desc">
                        <strong>State Tax Calculation</strong><br>
                        We calculate your California state tax. Your investment income (dividends + harvested gains) is offset by the margin interest you paid. The final tax bill is added to your margin loan balance for the new year.
                    </div>
                </div>
                """,
            )

    with gr.Group(visible=False) as results_box:
        gr.Markdown(
            """
        <div style=\"text-align: center; font-family: 'Inter', sans-serif;">
            <h2 style=\"font-size: 2rem; font-weight: 700; color: #58508d;">4. The Final Verdict: Your Simulated Future</h2>
            <p style=\"font-size: 1.1rem; color: #555;">After running the simulations, the results below show the range of possibilities for your net worth.</p>
        </div>
        """
        )
        with gr.Group() as summary_card:
            summary_title_output = gr.Markdown()
            summary_text_output = gr.Markdown()
        plot_output = gr.Plot()
        with gr.Accordion("View Monthly Data", open=False) as monthly_data_accordion:
            dataframe_output = gr.Dataframe(headers=["Month", "Min Net Worth", "25th Percentile", "Avg Net Worth", "75th Percentile", "Max Net Worth"], datatype=[
                                            "number", "str", "str", "str", "str", "str"])

        with gr.Accordion("Get Gemini Analysis", open=False) as gemini_analysis_accordion:
            gemini_key = gr.Textbox(
                label="Enter your Gemini API Key", type="password")
            analyze_button = gr.Button("Analyze Results")
            gemini_analysis_output = gr.Markdown()

    # --- Event Handlers ---
    def update_summary_style(summary_title):
        if "Survived" in summary_title:
            return gr.update(elem_classes="summary-card summary-survived")
        elif "Failed" in summary_title:
            return gr.update(elem_classes="summary-card summary-failed")
        return gr.update()

    def toggle_df_slider(dist_model):
        return gr.update(visible=dist_model == "Student's t")

    return_dist_model.change(toggle_df_slider, inputs=return_dist_model, outputs=return_dist_df)
    interest_rate_dist_model.change(toggle_df_slider, inputs=interest_rate_dist_model, outputs=interest_rate_dist_df)

    # Main simulation button
    run_button.click(
        fn=run_and_display_simulation,
        inputs=[
            initial_portfolio_value, initial_cost_basis, annual_spending,
            annual_return, annual_std_dev, margin_rate, margin_rate_std_dev,
            margin_limit, simulation_count, tax_harvesting_profit_threshold,
            return_dist_model, return_dist_df,
            interest_rate_dist_model, interest_rate_dist_df
        ],
        outputs=[
            results_box,
            summary_title_output,
            summary_text_output,
            plot_output,
            dataframe_output,
            monthly_data_accordion,
            gemini_analysis_accordion
        ],
        show_progress='full'
    ).then(
        fn=update_summary_style,
        inputs=summary_title_output,
        outputs=summary_card
    )

    # Gemini analysis button
    analyze_button.click(
        fn=get_gemini_analysis,
        inputs=[
            gemini_key, summary_text_output, dataframe_output,
            initial_portfolio_value, initial_cost_basis, annual_spending,
            annual_return, annual_std_dev, margin_rate, margin_rate_std_dev,
            margin_limit, simulation_count, tax_harvesting_profit_threshold,
            return_dist_model, return_dist_df,
            interest_rate_dist_model, interest_rate_dist_df
        ],
        outputs=[gemini_analysis_output],
        show_progress='full'
    )

if __name__ == "__main__":
    demo.launch()
