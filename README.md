# Sales & Customer Analytics AI Agent

**An AI-powered business analytics platform for sales performance analysis, customer intelligence, predictive analytics and data-driven decision-making.**

[![Production App](https://img.shields.io/badge/Streamlit-Production-brightgreen)](https://ai-agent-sales-and-customer-analytics-xxrti5gfmqxy5appjyeanyg.streamlit.app/)
[![Testing App](https://img.shields.io/badge/Streamlit-Testing-blue)](https://ai-agent-sales-and-customer-analytics-bgc2yk7uer9endoffe54n6.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red)](https://streamlit.io/)
[![AI](https://img.shields.io/badge/AI-Google%20Gemini-orange)](https://ai.google.dev/)

## 🚀 Live Applications

| Environment | Application |
|---|---|
| Production | [Launch Sales & Customer Analytics AI Agent](https://ai-agent-sales-and-customer-analytics-xxrti5gfmqxy5appjyeanyg.streamlit.app/) |
| Testing | [Launch Testing Application](https://ai-agent-sales-and-customer-analytics-bgc2yk7uer9endoffe54n6.streamlit.app/) |

## 📌 Project Overview

The Sales & Customer Analytics AI Agent is an interactive, AI-powered analytics application designed to help business users understand sales performance, identify customer patterns, evaluate business trends and perform advanced analytical tasks using natural-language queries.

Built using Python, Streamlit, Google Gemini and analytical libraries, the application combines traditional data analysis techniques with AI-driven query interpretation.

Users can upload business datasets and interact with the application through natural-language questions instead of relying entirely on manual analysis, coding or predefined reports.

The objective is to bridge the gap between business intelligence, customer analytics and AI-powered decision support.

## 🎯 Key Capabilities

### 1. Sales Performance Analytics
- Analyse overall sales and revenue performance.
- Evaluate regional and category-level performance.
- Identify sales trends and performance changes.
- Investigate business performance through natural-language queries.

### 2. Customer Analytics & Segmentation
- Analyse customer behaviour and purchasing patterns.
- Perform RFM (Recency, Frequency and Monetary) segmentation.
- Identify inactive and high-value customers.

### 3. Predictive Analytics
- Sales forecasting.
- Regression analysis.
- Churn modelling.
- Analytical evaluation of business trends.

### 4. Market Basket Analysis
- Analyse product purchase combinations.
- Identify product associations and purchasing patterns.
- Support cross-selling and product affinity analysis.

### 5. AI-Powered Business Analysis
- Ask business questions using natural language.
- Interpret analytical queries through AI-powered query intelligence.
- Execute relevant analytical operations.
- Present results in business-friendly formats.

### 6. Data Understanding & Validation
- Upload business datasets.
- Understand dataset structure and characteristics.
- Perform data-quality checks.
- Validate analytical results.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Application Framework | Streamlit |
| AI / LLM | Google Gemini |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Data Visualisation | Plotly |
| Testing | Pytest |
| Development Environment | VS Code |
| Version Control | Git & GitHub |
| Deployment | Streamlit Community Cloud |

## 🏗️ Application Architecture

The application follows a modular architecture that separates data handling, analytical operations, AI interpretation and response presentation.

```text
User
  |
  v
Streamlit Application (app.py)
  |
  v
AI Agent & Query Intelligence
  |
  +---------------------------+
  |                           |
  v                           v
Google Gemini             Query Parser
  |                           |
  +-------------+-------------+
                |
                v
       Analytical Tool Layer
                |
     +----------+----------+
     |          |          |
     v          v          v
  Sales     Customer    Predictive
 Analytics  Analytics   Analytics
     |          |          |
     +----------+----------+
                |
                v
       Result Validation
                |
                v
       Response & Presentation
                |
                v
          User Output
```

### Core Analytical Modules

The project uses a modular codebase with dedicated components for:

- Data loading and data quality.
- Query parsing and query intelligence.
- Sales and rule-based analysis.
- Customer segmentation and RFM.
- Churn modelling.
- Forecasting and regression.
- Market Basket Analysis.
- Result validation.
- Response generation and presentation.

## 📊 Example Business Questions

Users can interact with the application using questions such as:

- What is the overall sales performance?
- Which regions are experiencing a decline in sales?
- Identify inactive high-value customers.
- Perform RFM customer segmentation.
- Forecast future sales.
- Analyse customer churn.
- Which products are frequently purchased together?
- Perform regression analysis on the uploaded dataset.

The actual analytical questions available depend on the uploaded dataset and its supported structure.

## ⚙️ Installation & Setup

### Prerequisites
- Python
- Git
- Google Gemini API key

### 1. Clone the repository

```bash
git clone https://github.com/SiteshAnand/AI-Agent-Sales-and-Customer-Analytics.git
```

### 2. Navigate to the project directory

```bash
cd AI-Agent-Sales-and-Customer-Analytics
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```bash
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_actual_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Never commit your actual API key or `.env` file to GitHub.

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🧪 Testing & Reliability

The upgraded application includes automated tests covering analytical modules, query handling, data understanding, forecasting, customer analytics, regression, RFM and result validation.

**Latest recorded test result:**

- 136 automated tests passed.
- Heavy-dataset testing completed successfully.
- Separate testing and production deployments operational.

Testing was performed using Pytest.

```bash
python -m pytest -v
```

## ☁️ Deployment & Version Control

The project uses separate Git branches and Streamlit deployments:

| Branch | Purpose |
|---|---|
| `master` | Production application |
| `upgrade-testing` | Upgrade testing and validation |

This approach allows new functionality to be tested independently before production updates.

## 💼 Business Value

The application demonstrates how AI and analytics can be applied to practical business problems.

Its capabilities are relevant to:

- Revenue Operations.
- Sales Performance Management.
- Customer Success Operations.
- Customer Analytics.
- Business Intelligence.
- Predictive Business Analytics.

It brings together business understanding, analytical methods and AI-powered interaction in one application.

## 🔮 Future Development

Potential areas for further development include:

- Additional CRM and business-system integrations.
- Advanced customer lifetime value analytics.
- Deeper revenue and pipeline analytics.
- More sophisticated forecasting and predictive models.
- Enhanced analytical automation.
- Expanded business performance reporting.

## 👨‍💻 Author

**Sitesh Anand**

Business Analytics | Revenue Operations | Customer Analytics | AI-Powered Business Analysis

This project reflects the application of business domain knowledge, analytical thinking and AI engineering to solve practical business problems.

**GitHub:** [SiteshAnand](https://github.com/SiteshAnand)

---

*Built to connect business intelligence, customer analytics and artificial intelligence.*
