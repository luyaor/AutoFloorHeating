#!/bin/bash

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python first."
    exit 1
fi

# Check if virtualenv is installed
if ! command -v virtualenv &> /dev/null; then
    echo "virtualenv is not installed. Installing virtualenv..."
    pip install virtualenv
fi

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    virtualenv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip first to avoid warnings
# python -m pip install --upgrade pip > /dev/null 2>&1

# Check and install missing requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Checking for missing dependencies..."
    missing_deps=false
    
    # Cache pip list output
    pip_packages=$(pip list 2>/dev/null)
    
    while IFS= read -r requirement; do
        # Skip empty lines and comments
        [[ -z "$requirement" || "$requirement" =~ ^#.*$ ]] && continue
        
        # Extract package name without version
        package_name=$(echo "$requirement" | cut -d'>' -f1 | cut -d'<' -f1 | cut -d'=' -f1 | cut -d'[' -f1 | tr -d ' ')
        
        if ! echo "$pip_packages" | grep -i "^${package_name}\s" > /dev/null; then
            echo "Missing dependency: ${package_name}"
            missing_deps=true
        fi
    done < requirements.txt
    
    if [ "$missing_deps" = true ]; then
        echo "Installing missing dependencies..."
        pip install -r requirements.txt
    else
        echo "All dependencies are already installed."
    fi
fi

# Run the main script
PYTHONPATH=$PYTHONPATH:. python src/main.py

# Deactivate virtual environment
deactivate