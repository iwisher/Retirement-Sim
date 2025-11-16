# Product Requirements Document (PRD)

## 1. Introduction

This document outlines the product requirements for the Monte Carlo Retirement Simulation project. The project is a Python-based tool designed to model and analyze a specific retirement strategy. The core idea is to simulate a scenario where a retiree with a substantial stock portfolio lives by borrowing against it on margin, rather than by selling assets and realizing capital gains.

The simulation runs for a 10-year period and models various financial events, providing detailed statistical analysis to help users assess the viability and risks of this unconventional retirement strategy.

## 2. Core Features

The simulation models the following financial events:

- **Randomized Monthly Market Returns:** Simulates the monthly returns on the user's portfolio using a specified distribution model (Normal, Student's t, or Laplace).
- **Quarterly Dividend Payouts:** Calculates and applies dividend payments to the user's portfolio.
- **Monthly Living Expenses:** Funds the user's monthly living expenses by taking a margin loan.
- **Margin Loan Interest:** Accrues interest on the margin loan based on a specified interest rate and distribution model.
- **Tax-Gain Harvesting:** Implements an annual tax-gain harvesting strategy to take advantage of the federal tax-free gain limit.
- **State Taxes:** Calculates and applies state taxes (simplified for California).
- **Risk Management:** Includes a forced selling mechanism to prevent the margin loan from exceeding a defined limit relative to the portfolio value.

## 3. User Interface

The project provides two interfaces for running the simulation:

- **Command-Line Interface (CLI):** A script that runs the simulation with a default set of parameters and outputs the results to the console.
- **Web-Based User Interface (UI):** A Gradio-based web UI that allows users to interactively set a wide range of simulation parameters and visualize the results.

## 4. Input Parameters

The simulation is highly configurable, allowing users to specify the following parameters:

- **Initial Portfolio Value:** The initial value of the user's stock portfolio.
- **Initial Cost Basis:** The initial cost basis of the user's portfolio.
- **Annual Spending:** The user's estimated annual spending in retirement.
- **Monthly Passive Income:** The user's monthly passive income.
- **Portfolio Annual Return:** The expected annual return of the user's portfolio.
- **Portfolio Annual Standard Deviation:** The expected annual standard deviation of the user's portfolio.
- **Annual Dividend Yield:** The annual dividend yield of the user's portfolio.
- **Margin Loan Annual Average Interest Rate:** The average annual interest rate on the margin loan.
- **Margin Loan Annual Interest Rate Standard Deviation:** The standard deviation of the annual interest rate on the margin loan.
- **Brokerage Margin Limit:** The brokerage's margin limit as a percentage of the portfolio value.
- **Federal Tax-Free Gain Limit:** The federal tax-free gain limit for the user's filing status.
- **Tax Harvesting Profit Threshold:** The profit threshold for the tax-gain harvesting strategy.
- **Number of Simulations:** The number of Monte Carlo simulations to run.
- **Return Distribution Model:** The distribution model to use for simulating market returns (Normal, Student's t, or Laplace).
- **Interest Rate Distribution Model:** The distribution model to use for simulating margin loan interest rates (Normal, Student's t, or Laplace).
- **Enable Margin Investing:** Whether to enable margin investing.
- **Margin Investing Buffer:** The buffer to maintain below the margin limit when investing on margin.

## 5. Simulation Logic

The simulation is executed on a monthly basis for a period of 10 years. The following steps are performed in each monthly iteration:

1. **Asset Aging:** The oldest short-term asset bucket is aged into the long-term portfolio.
2. **Market Returns:** The market return for the month is calculated and applied to the portfolio.
3. **Margin Investing (Optional):** If enabled, the simulation borrows up to a user-defined buffer below the margin limit and invests the funds into the newest short-term bucket.
4. **Quarterly Dividends:** If it is a dividend month, the dividend payment is calculated and applied to the portfolio.
5. **Expenses and Margin Loan:** The user's monthly expenses are covered by taking a margin loan, and interest is accrued on the loan.
6. **Forced Selling:** If the margin loan exceeds the brokerage's limit, the simulation forces the sale of assets to bring the loan back within the limit.
7. **End-of-Year Tax Strategy:** At the end of each year, the simulation executes the tax-gain harvesting strategy and calculates and "pays" state taxes.
8. **Record Net Worth:** The user's net worth is recorded at the end of each month.

## 6. Output and Analysis

The simulation provides the following outputs:

- **Tabular Results:** A table showing the minimum, maximum, average, median, and various percentile outcomes for the user's net worth over the simulation period.
- **Graphical Results:** A plot showing the full range of percentile results, including the interquartile range.
- **Data Table:** A data table of the monthly net worth projections.

These outputs help users visualize the range of potential outcomes and assess the risks associated with the retirement strategy.
