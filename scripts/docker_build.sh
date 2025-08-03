#!/bin/bash

# Define the default image name and tag
IMAGE_NAME="aqi-prediction-image"
DEFAULT_TAG="latest"

# Use the first command-line argument as the tag, or use the default
if [ -n "$1" ]; then
  IMAGE_TAG="$1"
else
  IMAGE_TAG="$DEFAULT_TAG"
fi

echo "=========================================="
echo "Starting Docker image build..."
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo "=========================================="

# Run the Docker build command
docker build -t "$IMAGE_NAME:$IMAGE_TAG" .

# Check the exit status
if [ $? -eq 0 ]; then
  echo "=========================================="
  echo "Docker image build completed successfully!"
  echo "Image '$IMAGE_NAME:$IMAGE_TAG' is ready."
  echo "=========================================="
else
  echo "=========================================="
  echo "Error: Docker image build failed."
  echo "Please check the output above for errors."
  echo "=========================================="
  exit 1
fi