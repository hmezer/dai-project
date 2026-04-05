# DAI Project: Data Extraction & Financial Analysis

Welcome to the DAI Project! This repository showcases a comprehensive financial analysis workflow that demonstrates the process of extracting institutional data, preprocessing it, performing sentiment analysis on press releases, and developing predictive machine learning models to analyze stock market behavior.

## 📋 Overview

This project combines **institutional data extraction**, **financial feature engineering**, and **machine learning** to predict stock market metrics. The workflow is divided into three main phases:

1. **Data Extraction** – Non-reproducible extraction from institutional servers (source code provided as reference)
2. **Data Preprocessing** – Cleaning, feature engineering, and sentiment analysis on press releases
3. **Financial Analysis & Modeling** – Exploratory analysis and predictive modeling using market fundamentals and sentiment indicators

**Note:** While the original data extraction process cannot be replicated, the processed and compressed dataset is available in the [`data/`](./data) folder, enabling you to explore the analysis and models directly.

## 📁 Repository Structure

```
dai-project/
├── data/                           # Processed datasets and compressed data archives
│   ├── extracted_data_all.zip      # Full extracted dataset (compressed)
│   ├── stock_with_features_polarities_chunk_*.csv  # Processed stock data with features
│   ├── test_set.csv                # Test set for model evaluation
│   └── ...
├── data-extraction/                # Original data extraction scripts and documentation
│   ├── data_extraction_nasdaq_only.py  # Python script for institutional data extraction
│   ├── extraction-terminal-initial.PNG # Screenshot of extraction process start
│   └── extraction-terminal-final.PNG   # Screenshot of extraction process completion
├── preprocessing/                  # Data preprocessing and feature engineering notebooks
│   ├── dai_data_extraction.ipynb       # Guides extraction pipeline and data loading
│   ├── dai_sentiment_analysis.ipynb    # Sentiment analysis on press releases
│   ├── press_release_ticker_match.ipynb # Matching press releases to stock tickers
│   └── tickers.txt                 # List of NASDAQ tickers used in analysis
├── models/                         # Trained LightGBM models (compressed archives)
│   ├── lgb_models.zip              # Feature importance exclusion study models
│   └── models_past_vs_sentiment.zip # Base vs. sentiment-only comparison models
├── explanatory_data_analysis.ipynb # EDA on financial features and correlations
├── model_test_features_excluded.ipynb # Feature importance analysis (126 models)
├── model_test_past_vs_sentiment.ipynb # Sentiment vs. past features comparison
├── prediction_with_sentiment.ipynb    # Full prediction pipeline with sentiment integration
├── environment.yml                 # Conda environment configuration
├── requirements.txt                # Python package dependencies
└── README.md                       # This file
```

## 🔄 Workflow

### Phase 1: Data Extraction (`data-extraction/`)

The [`data_extraction_nasdaq_only.py`](./data-extraction/data_extraction_nasdaq_only.py) script extracts financial data from institutional servers. This process is **non-reproducible** due to institutional access requirements, but the script is provided for reference to understand the data collection methodology.

**Key outputs:**
- Raw historical stock price data
- Extracted institutional data archives (stored in `data/extracted_data_all.zip`)
- Screenshots documenting the extraction process

### Phase 2: Data Preprocessing (`preprocessing/`)

Three Jupyter notebooks orchestrate the preprocessing pipeline:

1. **[`dai_data_extraction.ipynb`](./preprocessing/dai_data_extraction.ipynb)**  
   - Loads and validates extracted data
   - Performs initial data cleaning and transformation
   - Structures data for downstream analysis

2. **[`dai_sentiment_analysis.ipynb`](./preprocessing/dai_sentiment_analysis.ipynb)**  
   - Analyzes sentiment from press releases and institutional communications
   - Generates sentiment polarity scores (diluted, pure, and immediate)
   - Integrates sentiment metrics with stock price data

3. **[`press_release_ticker_match.ipynb`](./preprocessing/press_release_ticker_match.ipynb)**  
   - Matches press releases to corresponding stock tickers
   - Aligns temporal records between news events and market data
   - Ensures data integrity across sources

**Output:** `stock_with_features_polarities_chunk_*.csv` – Combined datasets with:
- OHLC price data (Open, High, Low, Close, Volume)
- Historical returns and volatility metrics
- Momentum and drawdown indicators
- Sentiment polarity scores (3 variants)

### Phase 3: Financial Analysis & Modeling

#### Exploratory Data Analysis
**[`explanatory_data_analysis.ipynb`](./explanatory_data_analysis.ipynb)**
- Descriptive statistics and distribution analysis
- Correlation matrices between features
- Time series trends and seasonality
- Feature relationships and dependencies

#### Model Development

**[`model_test_features_excluded.ipynb`](./model_test_features_excluded.ipynb)**
- Feature importance analysis with **126 LightGBM models**
- Each model trained with one feature excluded at a time
- Evaluates impact of each feature on predictions
- Targets analyzed:
  - `future_1D_Return_log` – Daily returns (log scale)
  - `future_1W_Return_CUML_log` – Weekly cumulative returns
  - `future_1M_Return_CUML_log` – Monthly cumulative returns
  - `future_1W_momentum` – Weekly momentum
  - `future_1M_momentum` – Monthly momentum
  - `future_1W_volatility` – Weekly volatility
  - `future_1M_volatility` – Monthly volatility

**[`model_test_past_vs_sentiment.ipynb`](./model_test_past_vs_sentiment.ipynb)**
- Compares predictive power of **historical features vs. sentiment alone**
- For each target, trains two models:
  - **Base model:** Uses all historical features (past returns, momentum, volatility, drawdowns, gains, direction)
  - **Sentiment-only model:** Uses only polarity scores (diluted, pure, immediate)
- Validates the contribution of sentiment indicators to market predictions

**Key Results:**
- Historical features (especially volatility and momentum metrics) are the strongest predictors
- Sentiment metrics add incremental predictive value for certain targets
- Feature importance rankings provide actionable insights for model optimization

#### Full Prediction Pipeline
**[`prediction_with_sentiment.ipynb`](./prediction_with_sentiment.ipynb)**
- Integrated end-to-end prediction system
- Combines all engineered features with trained models
- Generates predictions with confidence metrics
- Ready for production deployment or backtesting

## 🎯 Key Features Engineered

| Feature Category | Examples | Description |
|---|---|---|
| **Returns** | `past_1D_Return_log`, `past_1W_Return_CUML_log`, `past_1M_Return_CUML_log` | Log-normalized returns at different time horizons |
| **Volatility** | `past_1W_volatility`, `past_1M_volatility` | Price volatility estimates |
| **Momentum** | `past_1W_momentum`, `past_1M_momentum` | Rate of change indicators |
| **Risk Metrics** | `past_1W_drawdown`, `past_1M_max_gain`, `past_1W_direction` | Risk and reversal indicators |
| **Sentiment** | `Polarity_diluted`, `Polarity_pure`, `Polarity_immediate` | Extracted sentiment from press releases |

## 🤖 Models

All models are trained using **LightGBM** (Light Gradient Boosting Machine) and are stored in the `models/` directory:

- **`lgb_models.zip`** – 126 models for feature importance exclusion analysis
- **`models_past_vs_sentiment.zip`** – 14 models (7 targets × 2 model types) comparing historical vs. sentiment features

### Model Performance Highlights

From [`model_test_past_vs_sentiment.ipynb`](./model_test_past_vs_sentiment.ipynb):

| Target | Base Model R² | Sentiment-Only R² | Best Predictor |
|---|---|---|---|
| `future_1M_volatility` | **0.326** | -0.197 | Historical volatility and returns |
| `future_1W_volatility` | 0.202 | -0.121 | Historical volatility |
| `future_1W_Return_CUML_log` | -0.070 | 0.005 | Weak; requires ensemble approach |
| `future_1M_Return_CUML_log` | -0.018 | -0.018 | Challenging target |
| `future_1M_momentum` | -0.045 | -0.009 | Marginal sentiment contribution |
| `future_1W_momentum` | 0.002 | — | Near-zero predictability |
| `future_1D_Return_log` | -0.020 | 0.004 | High-frequency data required |

## 📊 Data Specifications

- **Time Period:** 2007 – Present (extended historical coverage)
- **Securities:** NASDAQ stocks (see [`preprocessing/tickers.txt`](./preprocessing/tickers.txt))
- **Records:** 1,000+ stocks × 15+ years of daily OHLC data
- **Total Rows:** ~2.5 million stock-day observations
- **Features:** 46 engineered features + target variables

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Conda or pip for dependency management
- ~5GB free disk space for extracted datasets and models

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hmezer/dai-project.git
   cd dai-project
   ```

2. **Create environment from file:**
   ```bash
   conda env create -f environment.yml
   conda activate dai-project
   ```

   Or install via pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and extract datasets:**
   ```bash
   cd data/
   unzip extracted_data_all.zip
   unzip ../models/lgb_models.zip
   unzip ../models/models_past_vs_sentiment.zip
   ```

### Running the Analysis

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Explore in order:**
   - Start with [`explanatory_data_analysis.ipynb`](./explanatory_data_analysis.ipynb) for data overview
   - Review [`model_test_past_vs_sentiment.ipynb`](./model_test_past_vs_sentiment.ipynb) for model comparison insights
   - Examine [`model_test_features_excluded.ipynb`](./model_test_features_excluded.ipynb) for detailed feature importance
   - Run [`prediction_with_sentiment.ipynb`](./prediction_with_sentiment.ipynb) for end-to-end predictions

## 📈 Analysis Results (To Be Updated)

This section will be updated with final analysis results and key findings:

- [ ] Feature importance rankings and insights
- [ ] Model performance metrics across all targets
- [ ] Sentiment contribution analysis
- [ ] Backtest results and strategy recommendations
- [ ] Key market insights from the analysis

## 🛠 Technologies Used

- **Data Processing:** pandas, numpy
- **Machine Learning:** LightGBM, scikit-learn
- **Sentiment Analysis:** NLP libraries (TBD per [`dai_sentiment_analysis.ipynb`](./preprocessing/dai_sentiment_analysis.ipynb))
- **Visualization:** matplotlib, seaborn
- **Notebooks:** Jupyter, Google Colab

## 📝 License

This project is provided for educational and research purposes.

## 👤 Author

**Hmezer** – [GitHub Profile](https://github.com/hmezer)

## 📬 Contact & Contributions

For questions, suggestions, or contributions, please feel free to open an issue or pull request on the GitHub repository.

---

**Last Updated:** April 5, 2026  
**Project Status:** Active Development (awaiting final analysis results)