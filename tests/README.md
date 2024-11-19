# Test Suite Structure

This directory contains the test suite for the project. The structure is organized as follows:

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for end-to-end workflows
├── data/             # Test data and fixtures
├── models/           # Model test artifacts
├── pipelines/        # Pipeline test cases
└── conftest.py       # Shared pytest fixtures and configurations
```

## Test Categories

### Unit Tests (`unit/`)
- Test individual components in isolation
- Fast execution
- No external dependencies
- One test file per source file

### Integration Tests (`integration/`)
- Test multiple components working together
- End-to-end pipeline tests
- May require external resources
- Organized by feature or workflow

### Test Data (`data/`)
- Sample datasets for testing
- Mock data fixtures
- Test case inputs and expected outputs

### Model Tests (`models/`)
- Model-specific test cases
- Saved model artifacts for testing
- Model evaluation test cases

### Pipeline Tests (`pipelines/`)
- Data pipeline test cases
- Training pipeline tests
- Inference pipeline tests

## Running Tests

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/
pytest tests/integration/

# Run with coverage report
pytest --cov=src

# Run with verbose output
pytest -v
```

## Writing Tests

1. Follow the existing structure
2. Use fixtures from conftest.py
3. Write descriptive test names
4. Include docstrings explaining test purpose
5. Keep tests independent and idempotent

## Best Practices

1. Test both success and failure cases
2. Use meaningful assertions
3. Keep tests simple and focused
4. Use appropriate fixtures
5. Clean up test artifacts
6. Document test requirements
