# PyPipeline

A Python project for pipeline processing and analysis.

## Project Structure

```text
pypipeline/
├── data/           # Data and test data files
├── src/            # Source code
│   ├── cactus.py   # Cactus algorithm implementation
│   ├── partition.py # Partition processing
│   └── __init__.py
└── tests/          # Test files
```

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dependencies

Main dependencies are listed in `requirements.txt`:

- numpy: Scientific computing
- scipy: Scientific algorithms
- matplotlib: Plotting and visualization
- networkx: Graph operations
- shapely: Geometric operations
- opencv-python: Computer vision
- loguru: Logging
