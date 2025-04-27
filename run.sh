#!/bin/bash

# Build the image if not built yet
echo "🔧 Building the Docker image (if needed)..."
docker compose build

# Run the container
echo "🐳 Starting the container with GPU access..."
docker compose run --rm offroad
