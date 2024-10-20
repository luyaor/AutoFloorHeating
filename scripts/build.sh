#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Go to the project root directory
cd "$SCRIPT_DIR/.."

# Create build directory if it doesn't exist
mkdir -p build

# Go to build directory
cd build

# Clean the build directory
echo "Cleaning build directory..."
rm -rf *

# Run CMake
echo "Configuring CMake..."
cmake ..

# Build the project
echo "Building the project..."
make

# Run tests
echo "Running tests..."
make test

echo "Build completed successfully!"
