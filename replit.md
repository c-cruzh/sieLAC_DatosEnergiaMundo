# sieLAC Global Energy Data Dashboard

## Overview
Interactive Streamlit dashboard displaying global energy data from the sieLAC dataset. The application visualizes energy supply, consumption, and renewable energy data across different world regions.

**Current State**: Fully functional and deployed
**Last Updated**: November 29, 2025

## Project Architecture

### Technology Stack
- **Language**: Python 3.11
- **Framework**: Streamlit 1.51.0
- **Data Processing**: Pandas 2.3.3
- **Visualization**: Plotly 6.5.0

### Project Structure
```
.
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                # Streamlit server configuration
├── Infraestructura/               # Infrastructure data
│   ├── Eléctrico/                # Electrical infrastructure
│   └── Hidrocarburos/            # Hydrocarbon infrastructure
└── Oferta y demanda/              # Supply and demand data
    ├── Carbón mineral/           # Mineral coal data
    ├── Eléctrico/                # Electrical data
    ├── Hidrocarburos/            # Hydrocarbon data
    ├── Renovables/               # Renewable energy data
    └── Todos/                    # All energy sources combined
```

### Key Features
1. **World Tab**: Global energy overview for 2023
   - Total energy supply and final consumption metrics
   - Renewable energy participation percentage
   - Distribution charts by energy source
   - Top 5 regions comparison

2. **Regional Comparison Tab**: Multi-region analysis
   - Interactive region selection
   - Stacked bar charts for energy supply by source
   - Scatter plots for final consumption
   - Detailed data tables

3. **Renewables Tab**: Renewable energy focus
   - Biomass and biofuel consumption data
   - Multi-region comparison charts
   - Data in thousands of oil barrel equivalents (10³ bep)

## Configuration

### Development Environment
- **Server**: Streamlit runs on `0.0.0.0:5000`
- **Proxy Compatible**: Configured for Replit's iframe proxy with `enableCORS=false` and `enableXsrfProtection=false`
- **Workflow**: Automatically starts Streamlit app on port 5000

### Deployment
- **Target**: Autoscale deployment (stateless web application)
- **Command**: `streamlit run app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true`
- **Suitable for**: Public deployment with automatic scaling based on traffic

## Recent Changes

### November 29, 2025 - Initial Replit Setup
- Installed Python 3.11 and all dependencies
- Configured Streamlit for Replit environment (port 5000, proxy compatibility)
- Fixed data type handling in `_strip_quotes()` function to accept int/float values
- Fixed default region selection in renewables tab to match available data
- Created `.gitignore` for Python project
- Configured deployment for production (autoscale)
- Set up workflow for automatic app startup

### Bug Fixes
1. **Type Error Fix**: Updated `_strip_quotes()` to handle integer and float values from CSV files
2. **Region Selection Fix**: Updated renewables tab default values to only include regions present in the dataset

## Data Sources
All CSV files are sourced from the sieLAC (Sistema de Información Energética de América Latina y el Caribe) dataset, containing energy sector information from 1970 onwards for major world regions:
- África
- América Latina y el Caribe
- Asia & Australia (in some datasets)
- Europa
- Ex URSS
- Medio Oriente
- Norte América
- Mundo
- OCDE

## Running the Application

### Development
The application automatically starts via the configured workflow. It loads and caches data from CSV files on first run.

### Notes
- Data is cached using `@st.cache_data` decorator for optimal performance
- The app includes data cleaning utilities to handle various CSV formatting issues
- Energy values are normalized to common units (10⁶ tep or 10³ bep) for comparison

## Dependencies
See `requirements.txt` for complete list:
- streamlit >= 1.32
- pandas >= 2.1
- plotly >= 5.18
