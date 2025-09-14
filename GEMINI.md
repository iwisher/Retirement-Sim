# GEMINI.md

## Project Overview

This project is a Python-based Monte Carlo simulation designed to model and analyze a specific retirement strategy. The core idea is to simulate a scenario where a retiree with a substantial stock portfolio lives by borrowing against it on margin, rather than by selling assets and realizing capital gains.

The simulation runs for a 10-year period and models various financial events, including:
-   Randomized monthly market returns on the portfolio.
-   Quarterly dividend payouts.
-   Monthly living expenses funded by a margin loan.
-   Accrual of interest on the margin loan.
-   An annual tax-gain harvesting strategy to take advantage of the federal tax-free gain limit.
-   Calculation of state taxes (simplified for California).
-   Risk management, including forced selling if the margin loan exceeds a defined limit relative to the portfolio value.

The project consists of two main components:
1.  `simulation.py`: A command-line script that runs the core Monte Carlo simulation with a default set of parameters. It outputs the results as a table to the console and generates a plot using `matplotlib`.
2.  `app.py`: A web-based user interface built with Gradio. This allows users to interactively set the simulation parameters and visualizes the results, including a summary of the outcome, a detailed plot, and a data table of the monthly net worth projections.

The goal is to provide a tool to assess the viability and risks of this unconventional retirement strategy over thousands of potential market scenarios.

## Building and Running

This project is written in Python and its dependencies are listed in `requirements.txt`.

### 1. Install Dependencies

To install the necessary packages, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

### 2. Running the Simulation

There are two ways to run the project:

**A) Command-Line Simulation**

To run the simulation with the default parameters defined within the script, execute:

```bash
python3 simulation.py
```

This will print a month-by-month breakdown of the simulated net worth (minimum, average, and maximum) to the console and display a `matplotlib` chart.

**B) Interactive Web UI**

To launch the user-friendly Gradio web interface, run:

```bash
python3 app.py
```

This will start a local web server. You can access the interface by opening the URL provided in the terminal (usually `http://127.0.0.1:7860`) in your web browser. The web UI provides an intuitive way to change inputs and visualize the results.

## Development Conventions

Based on the existing code, the following conventions are in place:

*   **Modularity:** The core simulation logic is encapsulated in the `run_simulation` function within `simulation.py`. The user interface in `app.py` is kept separate and imports the simulation logic. This is a good practice for separating concerns.
*   **Configuration:** Simulation parameters are managed via a Python dictionary named `inputs`. In `simulation.py`, this is defined directly in the `main()` function. In `app.py`, it is constructed from the user inputs in the Gradio interface.
*   **Clarity and Comments:** The simulation logic in `simulation.py` is well-structured with comments explaining each step of the monthly and annual cycles, making it easier to understand and maintain.
*   **Data Handling:** The `pandas` library is used in `app.py` to structure the results into a DataFrame for easy display in the Gradio UI. `numpy` is used extensively for numerical calculations and random number generation in the simulation.
*   **Visualization:** `matplotlib` is the chosen library for plotting the simulation results, both in the standalone script and within the Gradio application.
