Here's the main README.md file for the DeFi Trading Pattern Analysis Tool:

```markdown
# DeFi Trading Pattern Analysis Tool

A comprehensive tool for analyzing trading patterns on decentralized exchanges using transaction data from DexTools API. This tool detects wave patterns in trading activities and generates detailed reports for traders and analysts.

## Project Overview

This tool connects to the DexTools API to gather trading data from decentralized exchanges, stores it in a PostgreSQL database, and applies specialized algorithms to detect various trading patterns. It provides insights through detailed reports and visualizations.

### Key Features

- **Data Collection**: Seamless integration with DexTools API to fetch real-time and historical trading data
- **Data Storage**: Efficient PostgreSQL database storage with proper indexing for quick access
- **Pattern Analysis**: Advanced algorithms for detecting Elliott Wave patterns and other market movements
- **Reporting**: Comprehensive reports and visualizations of detected patterns
- **Caching System**: Optimized data caching to minimize API calls and improve performance

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL 13+
- DexTools API access (key required)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/defi-trading-pattern-analyzer.git
   cd defi-trading-pattern-analyzer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

4. Create a `.env` file by copying the example:
   ```bash
   cp .env.example .env
   ```

5. Edit the `.env` file with your settings, including your DexTools API key.

6. Initialize the database:
   ```bash
   alembic upgrade head
   ```

## Usage

### Basic Usage

```bash
# Run the main analysis tool
python -m src.main --token_address 0x123456789abcdef --days 30

# Generate a report for a specific token
python -m src.reports.generator --token_address 0x123456789abcdef --output report.pdf
```

### Configuration

The application is configured through environment variables and/or a `.env` file. See `.env.example` for all available configuration options.

## Development

### Project Structure

```
├── alembic/              # Database migration scripts
├── config/               # Configuration management
├── src/                  # Source code
│   ├── api/              # API clients (DexTools, etc.)
│   ├── database/         # Database connection and repository
│   ├── models/           # Data models and schemas
│   ├── analysis/         # Pattern analysis algorithms
│   ├── reports/          # Report generation
│   └── utils/            # Utility functions
├── tests/                # Test suite
└── docs/                 # Documentation
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src
```

## License

[MIT License](LICENSE)

## Acknowledgements

- [DexTools](https://www.dextools.io/) for providing the API
- [Elliott Wave Theory](https://en.wikipedia.org/wiki/Elliott_wave_principle) for the foundational pattern analysis concepts
```

This README.md provides a comprehensive overview of the project, installation instructions, usage examples, and development guidance. It clearly communicates the purpose and features of the DeFi Trading Pattern Analysis Tool.