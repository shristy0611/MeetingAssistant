#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running code quality checks...${NC}"

# Run Ruff linter
echo "Running Ruff..."
ruff check .

# Run Black formatter check
echo "Running Black..."
black --check .

echo -e "${YELLOW}Running tests...${NC}"

# Create coverage directory if it doesn't exist
mkdir -p coverage

# Run pytest with coverage
pytest \
    --verbose \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html:coverage/html \
    --cov-report=xml:coverage/coverage.xml \
    --junitxml=coverage/junit.xml \
    tests/

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Tests failed!${NC}"
    exit 1
fi 