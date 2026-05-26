# Pool Management in Kidney Exchange Programs

## Requirements

- The project was done using Python 3.12.0, so Python 3.11 or higher is necessary.

- A valid installation and license of Gurobi Optimizer (required by gurobipy).

See the official Gurobi documentation for installation instructions:
https://www.gurobi.com/

## Installation

This project uses `pyproject.toml` for dependency management. If `pip` is available, you can install the package and all its required dependencies automatically by running the following command in the root directory:

```
pip install .
```

For development mode:

```
pip install -e .
```

## Dependencies

All dependencies are defined in `pyproject.toml` and will be installed automatically when installing the package:

```
numpy
gurobipy
pandas
matplotlib
seaborn
```

## Usage

### To generate synthetic data:

```
python -m scripts.run_generate
```

### To run the main program:

```
python -m scripts.run_kep
```

### To process and visualize results:

```
python -m scripts.process_results
python -m scripts.plot_results
```

## Running Tests

To install testing dependencies execute the command:

```
pip install .[test]
```

Then run:

```
pytest
```

This will run all the tests implemented and generate a test coverage report.
