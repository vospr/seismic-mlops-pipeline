#!/bin/bash
# Seismic MLOps Pipeline - Docker Quick Start
# Usage: ./scripts/docker-start.sh

set -e

echo "=============================================="
echo "Seismic MLOps Pipeline - Docker Setup"
echo "=============================================="

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker Desktop first."
    exit 1
fi

echo "Docker is running..."

# Build the image
echo ""
echo "Building Docker image..."
docker-compose build

# Start services
echo ""
echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo ""
echo "Waiting for services to start..."
sleep 10

# Check service health
echo ""
echo "Checking service status..."
docker-compose ps

# Run the pipeline
echo ""
echo "=============================================="
echo "Services are ready!"
echo "=============================================="
echo ""
echo "Access points:"
echo "  - API Server: http://localhost:8000"
echo "  - MLflow UI:  http://localhost:5000"
echo "  - Metrics:    http://localhost:8001/metrics"
echo ""
echo "To run the full pipeline:"
echo "  docker-compose exec mlops python run_all_stages.py"
echo ""
echo "To run quick validation:"
echo "  docker-compose exec mlops python src/stage8_cicd.py"
echo ""
echo "To stop services:"
echo "  docker-compose down"
