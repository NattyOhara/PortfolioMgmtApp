# Stock Portfolio Management Web Application

## Overview
A stock portfolio management web application built with Python and Streamlit.
Loads portfolio data from CSV files, fetches real-time stock price data,
and performs profit/loss calculations and risk analysis.

## Features
- Portfolio data import from CSV files
- Real-time stock price fetching (Yahoo Finance)
- Multi-currency support (all evaluated in Japanese Yen base)
- Profit/loss calculation and visualization
- Risk metrics calculation (VaR, CVaR, volatility, etc.)
- Interactive dashboard

## Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/portfolio-management-app.git
cd portfolio-management-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

## Usage
```bash
# Start the application
streamlit run app.py
```

Access http://localhost:8501 in your browser.

## CSV File Format
```csv
Ticker,Shares,AvgCostJPY
AAPL,100,15000
MSFT,50,25000
7203.T,1000,800
```

## License
MIT License# PortfolioMgmtApp
