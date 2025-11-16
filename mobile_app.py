
import os
# Set environment variables to configure Kivy before importing other modules
os.environ['KIVY_METRICS_DENSITY'] = '1'
os.environ['KIVY_DPI'] = '96'

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.checkbox import CheckBox
from kivy.uix.slider import Slider

from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.clock import Clock, mainthread
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import threading

from simulation import run_simulation

class RetirementSimApp(App):
    def build(self):
        self.title = "Retirement Simulator"
        self.inputs = {}

        # --- Main Layout ---
        root = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # --- Scrollable Input Area ---
        scroll_view = ScrollView(size_hint_y=1)
        
        main_grid = GridLayout(cols=1, spacing=10, size_hint_y=None)
        main_grid.bind(minimum_height=main_grid.setter('height'))

        # --- Header 1 ---
        main_grid.add_widget(Label(text="Step 1: Core Financial Inputs", size_hint_y=None, height=40, font_size='18sp', bold=True))
        
        # --- Core Inputs Grid ---
        core_grid = GridLayout(cols=2, spacing=10, size_hint_y=None)
        core_grid.bind(minimum_height=core_grid.setter('height'))
        self._create_core_input_widgets(core_grid)
        main_grid.add_widget(core_grid)

        # --- Header 2 ---
        main_grid.add_widget(Label(text="Step 2: Advanced Settings", size_hint_y=None, height=40, font_size='18sp', bold=True))

        # --- Advanced Inputs Grid ---
        advanced_grid = GridLayout(cols=2, spacing=10, size_hint_y=None)
        advanced_grid.bind(minimum_height=advanced_grid.setter('height'))
        self._create_advanced_input_widgets(advanced_grid)
        main_grid.add_widget(advanced_grid)
        
        scroll_view.add_widget(main_grid)
        root.add_widget(scroll_view)

        # --- Run Button ---
        self.run_button = Button(text="Run Simulation", size_hint_y=None, height=50, background_color=(0.2, 0.6, 0.8, 1))
        self.run_button.bind(on_press=self.run_simulation_callback)
        root.add_widget(self.run_button)

        # --- Results Area ---
        self.summary_label = Label(text="Run the simulation to see results.", size_hint_y=0.1, markup=True)
        root.add_widget(self.summary_label)

        # --- Tabbed Panel for Plot and Data ---
        results_panel = TabbedPanel(do_default_tab=False, size_hint_y=0.9)
        
        # Plot Tab
        plot_tab = TabbedPanelItem(text='Plot')
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasKivyAgg(self.fig)
        plot_tab.add_widget(self.canvas)
        results_panel.add_widget(plot_tab)

        # Data Tab
        data_tab = TabbedPanelItem(text='Data Table')
        data_scroll = ScrollView()
        self.data_grid = GridLayout(cols=9, size_hint_y=None, spacing=2)
        self.data_grid.bind(minimum_height=self.data_grid.setter('height'))
        data_scroll.add_widget(self.data_grid)
        data_tab.add_widget(data_scroll)
        results_panel.add_widget(data_tab)

        root.add_widget(results_panel)

        return root

    def _add_widget_pair(self, grid, label_text, widget_key, widget):
        label = Label(text=label_text, size_hint_y=None, height=40)
        widget.size_hint_y = None
        widget.height = 40
        grid.add_widget(label)
        grid.add_widget(widget)
        self.inputs[widget_key] = widget

    def _create_core_input_widgets(self, grid):
        self._add_widget_pair(grid, "Portfolio Value ($)", 'initial_portfolio_value', TextInput(text='2000000', multiline=False, input_filter='float'))
        self._add_widget_pair(grid, "Cost Basis ($)", 'initial_cost_basis', TextInput(text='1500000', multiline=False, input_filter='float'))
        self._add_widget_pair(grid, "Annual Spending ($)", 'annual_spending', TextInput(text='120000', multiline=False, input_filter='float'))
        self._add_widget_pair(grid, "Monthly Passive Income ($)", 'monthly_passive_income', TextInput(text='1000', multiline=False, input_filter='float'))
        self._add_widget_pair(grid, "Annual Dividend Yield (%)", 'annual_dividend_yield', TextInput(text='4.0', multiline=False, input_filter='float'))
        self._add_widget_pair(grid, "Federal Tax-Free Gain Limit ($)", 'federal_tax_free_gain_limit', TextInput(text='123250', multiline=False, input_filter='float'))
        self._add_widget_pair(grid, "Avg. Annual Return (%)", 'annual_return', TextInput(text='10', multiline=False, input_filter='float'))
        self._add_widget_pair(grid, "Annual Std. Dev. (%)", 'annual_std_dev', TextInput(text='19', multiline=False, input_filter='float'))
        self._add_widget_pair(grid, "Avg. Margin Rate (%)", 'margin_rate', TextInput(text='6', multiline=False, input_filter='float'))
        self._add_widget_pair(grid, "Margin Rate Std. Dev. (%)", 'margin_rate_std_dev', TextInput(text='5', multiline=False, input_filter='float'))
        self._add_widget_pair(grid, "Margin Borrow Limit (%)", 'margin_limit', TextInput(text='55', multiline=False, input_filter='float'))
        self._add_widget_pair(grid, "# of Simulations", 'simulation_count', TextInput(text='1000', multiline=False, input_filter='int'))
        self._add_widget_pair(grid, "Tax Harvest Profit Threshold (%)", 'tax_harvesting_profit_threshold', TextInput(text='30', multiline=False, input_filter='float'))

    def _create_advanced_input_widgets(self, grid):
        # Return Distribution
        self._add_widget_pair(grid, "Return Distribution", 'return_dist_model', Spinner(text='Normal', values=('Normal', "Student's t", 'Laplace')))
        self.inputs['return_dist_model'].bind(text=self.toggle_return_df_slider)
        
        # DF Return Slider
        self.return_df_label = Label(text="Degrees of Freedom (Return)", size_hint_y=None, height=40)
        df_return_layout = BoxLayout(size_hint_y=None, height=40)
        self.return_df_slider = Slider(min=2.1, max=30, value=5, step=0.1)
        self.return_df_value_label = Label(text=f'{self.return_df_slider.value:.1f}', size_hint_x=0.3)
        self.return_df_slider.bind(value=self.update_slider_label)
        df_return_layout.add_widget(self.return_df_slider)
        df_return_layout.add_widget(self.return_df_value_label)
        grid.add_widget(self.return_df_label)
        grid.add_widget(df_return_layout)
        self.inputs['return_dist_df'] = self.return_df_slider
        self.toggle_return_df_slider(None, 'Normal')

        # Interest Rate Distribution
        self._add_widget_pair(grid, "Interest Rate Distribution", 'interest_rate_dist_model', Spinner(text='Normal', values=('Normal', "Student's t", 'Laplace')))
        self.inputs['interest_rate_dist_model'].bind(text=self.toggle_interest_df_slider)

        # DF Interest Slider
        self.ir_df_label = Label(text="Degrees of Freedom (Interest)", size_hint_y=None, height=40)
        df_interest_layout = BoxLayout(size_hint_y=None, height=40)
        self.ir_df_slider = Slider(min=2.1, max=30, value=5, step=0.1)
        self.ir_df_value_label = Label(text=f'{self.ir_df_slider.value:.1f}', size_hint_x=0.3)
        self.ir_df_slider.bind(value=self.update_slider_label)
        df_interest_layout.add_widget(self.ir_df_slider)
        df_interest_layout.add_widget(self.ir_df_value_label)
        grid.add_widget(self.ir_df_label)
        grid.add_widget(df_interest_layout)
        self.inputs['interest_rate_dist_df'] = self.ir_df_slider
        self.toggle_interest_df_slider(None, 'Normal')

        # Margin Investing
        self._add_widget_pair(grid, "Enable Margin Investing", 'enable_margin_investing', CheckBox())
        self.inputs['enable_margin_investing'].bind(active=self.toggle_margin_buffer)

        # Margin Buffer Slider
        self.margin_buffer_label = Label(text="Margin Investing Buffer (%)", size_hint_y=None, height=40)
        margin_buffer_layout = BoxLayout(size_hint_y=None, height=40)
        self.margin_buffer_slider = Slider(min=0, max=50, value=10, step=1)
        self.margin_buffer_value_label = Label(text=f'{int(self.margin_buffer_slider.value)}%', size_hint_x=0.3)
        self.margin_buffer_slider.bind(value=self.update_slider_label)
        margin_buffer_layout.add_widget(self.margin_buffer_slider)
        margin_buffer_layout.add_widget(self.margin_buffer_value_label)
        grid.add_widget(self.margin_buffer_label)
        grid.add_widget(margin_buffer_layout)
        self.inputs['margin_investing_buffer'] = self.margin_buffer_slider
        self.toggle_margin_buffer(None, False)

    def update_slider_label(self, instance, value):
        if instance == self.return_df_slider:
            self.return_df_value_label.text = f'{value:.1f}'
        elif instance == self.ir_df_slider:
            self.ir_df_value_label.text = f'{value:.1f}'
        elif instance == self.margin_buffer_slider:
            self.margin_buffer_value_label.text = f'{int(value)}%'

    def toggle_return_df_slider(self, spinner, text):
        is_visible = (text == "Student's t")
        self.return_df_label.opacity = 1 if is_visible else 0
        self.return_df_slider.opacity = 1 if is_visible else 0
        self.return_df_value_label.opacity = 1 if is_visible else 0
        self.return_df_label.height = 40 if is_visible else 0
        self.return_df_slider.height = 40 if is_visible else 0
        self.return_df_value_label.height = 40 if is_visible else 0


    def toggle_interest_df_slider(self, spinner, text):
        is_visible = (text == "Student's t")
        self.ir_df_label.opacity = 1 if is_visible else 0
        self.ir_df_slider.opacity = 1 if is_visible else 0
        self.ir_df_value_label.opacity = 1 if is_visible else 0
        self.ir_df_label.height = 40 if is_visible else 0
        self.ir_df_slider.height = 40 if is_visible else 0
        self.ir_df_value_label.height = 40 if is_visible else 0

    def toggle_margin_buffer(self, checkbox, is_active):
        self.margin_buffer_label.opacity = 1 if is_active else 0
        self.margin_buffer_slider.opacity = 1 if is_active else 0
        self.margin_buffer_value_label.opacity = 1 if is_active else 0
        self.margin_buffer_label.height = 40 if is_active else 0
        self.margin_buffer_slider.height = 40 if is_active else 0
        self.margin_buffer_value_label.height = 40 if is_active else 0

    def run_simulation_callback(self, instance):
        self.run_button.disabled = True
        self.summary_label.text = "Running simulation, please wait..."
        threading.Thread(target=self._execute_simulation).start()

    def _execute_simulation(self):
        try:
            sim_inputs = {
                key: float(widget.text) for key, widget in self.inputs.items()
                if isinstance(widget, TextInput)
            }
            sim_inputs.update({
                'return_distribution_model': self.inputs['return_dist_model'].text,
                'return_distribution_df': self.inputs['return_dist_df'].value,
                'interest_rate_distribution_model': self.inputs['interest_rate_dist_model'].text,
                'interest_rate_distribution_df': self.inputs['interest_rate_dist_df'].value,
                'enable_margin_investing': self.inputs['enable_margin_investing'].active,
                'margin_investing_buffer': self.inputs['margin_investing_buffer'].value,
            })

            # Convert percentages to decimals
            for key in ['annual_return', 'annual_std_dev', 'annual_dividend_yield', 'margin_rate', 'margin_rate_std_dev', 'margin_limit', 'tax_harvesting_profit_threshold', 'margin_investing_buffer']:
                if key in sim_inputs:
                    sim_inputs[key] /= 100
            
            # Rename keys for simulation function
            sim_inputs['portfolio_annual_return'] = sim_inputs.pop('annual_return')
            sim_inputs['portfolio_annual_std_dev'] = sim_inputs.pop('annual_std_dev')
            sim_inputs['annual_dividend_yield'] = sim_inputs.pop('annual_dividend_yield')
            sim_inputs['margin_loan_annual_avg_interest_rate'] = sim_inputs.pop('margin_rate')
            sim_inputs['margin_loan_annual_interest_rate_std_dev'] = sim_inputs.pop('margin_rate_std_dev')
            sim_inputs['brokerage_margin_limit'] = sim_inputs.pop('margin_limit')
            sim_inputs['num_simulations'] = int(sim_inputs['simulation_count'])


            results, _ = run_simulation(sim_inputs)

            # --- Create DataFrame ---
            df = pd.DataFrame({
                'Month': range(1, len(results['avg_net_worth']) + 1),
                'Min Net Worth': results['min_net_worth'],
                '1st Percentile': results['p01_net_worth'],
                '25th Percentile': results['p25_net_worth'],
                'Median Net Worth': results['median_net_worth'],
                'Avg Net Worth': results['avg_net_worth'],
                '75th Percentile': results['p75_net_worth'],
                '99th Percentile': results['p99_net_worth'],
                'Max Net Worth': results['max_net_worth']
            })

            self._update_ui_with_results(results, df)

        except Exception as e:
            self.summary_label.text = f"An error occurred: {e}"
            self.run_button.disabled = False

    @mainthread
    def _update_ui_with_results(self, results, df):
        # --- Create Summary ---
        stop_month = -1
        if results and results.get('avg_net_worth') is not None:
            for i, avg_net_worth in enumerate(results['avg_net_worth']):
                if avg_net_worth < 0:
                    stop_month = i + 1
                    break

            if stop_month != -1:
                summary_text = f"Strategy Failed: Average net worth dropped below zero in month {stop_month}."
            else:
                final_avg_net_worth = results['avg_net_worth'][-1]
                summary_text = f"Strategy Survived: Average net worth after 10 years is ${final_avg_net_worth:,.0f}."
            self.summary_label.text = summary_text

            # --- Create Plot ---
            self.ax.clear()
            months = range(1, len(results['avg_net_worth']) + 1)
            self.ax.plot(months, results['median_net_worth'], label='Median Net Worth', color='purple')
            self.ax.plot(months, results['avg_net_worth'], label='Average Net Worth', color='blue')
            self.ax.fill_between(months, results['p25_net_worth'], results['p75_net_worth'], color='gray', alpha=0.4, label='Interquartile Range')
            self.ax.fill_between(months, results['p01_net_worth'], results['p99_net_worth'], color='gray', alpha=0.2, label='1-99 Percentile Range')
            self.ax.set_title('Simulated Net Worth Over 10 Years')
            self.ax.set_xlabel('Month')
            self.ax.set_ylabel('Net Worth ($)')
            self.ax.legend()
            self.ax.grid(True, linestyle='--', alpha=0.6)
            self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:,.0f}k'))
            self.canvas.draw()

            # --- Populate Data Table ---
            self.data_grid.clear_widgets()
            # Add headers
            for col_name in df.columns:
                self.data_grid.add_widget(Label(text=col_name, bold=True, size_hint_y=None, height=40))
            # Add data
            for index, row in df.iterrows():
                for col in df.columns:
                    if col == 'Month':
                        self.data_grid.add_widget(Label(text=str(row[col]), size_hint_y=None, height=30))
                    else:
                        self.data_grid.add_widget(Label(text=f'${row[col]:,.0f}', size_hint_y=None, height=30))

        else:
            self.summary_label.text = "Simulation failed to produce results."

        self.run_button.disabled = False

if __name__ == '__main__':
    RetirementSimApp().run()
