# Contributing to ASNU

Thank you for your interest in contributing to ASNU!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/asnu.git
cd asnu
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode with dev dependencies:
```bash
pip install -e ".[dev]"
```

## Running Tests

Run the test suite:
```bash
pytest tests/ -v
```

Run tests with coverage:
```bash
pytest tests/ --cov=asnu --cov-report=html
```

## Code Style

We follow PEP 8 style guidelines. Before submitting:

1. Format your code:
```bash
black asnu/ tests/
```

2. Check for style issues:
```bash
flake8 asnu/ tests/
```

## Submitting Changes

1. Fork the repository
2. Create a new branch for your feature: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite to ensure everything passes
6. Commit your changes: `git commit -am 'Add feature description'`
7. Push to your fork: `git push origin feature-name`
8. Submit a pull request

## Reporting Issues

When reporting issues, please include:
- ASNU version
- Python version
- Operating system
- Minimal example to reproduce the issue
- Expected vs actual behavior

## Questions?

Feel free to open an issue for questions or discussions about ASNU.
