#!/bin/bash

# Define the image name and tag
IMAGE_NAME="aqi-prediction-image"
IMAGE_TAG="latest"

# Check if a Python script name was provided as an argument
if [ -z "$1" ]; then
  echo "Error: No Python script specified."
  echo "Usage: $0 <path_to_script>"
  echo "Example: $0 scripts/fetch_data.py"
  exit 1
fi

# The path to the Python script is the first argument
PYTHON_SCRIPT="$1"

# Check if the required environment variables are set
if [ -z "$AQI_TOKEN" ]; then
  echo "Error: AQI_TOKEN environment variable is not set."
  exit 1
fi

if [ -z "$HOPSWORKS_AQI_TOKEN" ]; then
  echo "Error: HOPSWORKS_AQI_TOKEN environment variable is not set."
  exit 1
fi

echo "=========================================="
echo "Running Python script '$PYTHON_SCRIPT' inside Docker container..."
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo "=========================================="

# Run the container with the specified environment variables and script
docker run --rm \
  -e AQI_TOKEN="${AQI_TOKEN}" \
  -e HOPSWORKS_AQI_TOKEN="${HOPSWORKS_AQI_TOKEN}" \
  "$IMAGE_NAME:$IMAGE_TAG" \
  python "$PYTHON_SCRIPT"

# Check the exit status of the docker run command
if [ $? -eq 0 ]; then
  echo "=========================================="
  echo "Python script '$PYTHON_SCRIPT' ran successfully."
  echo "=========================================="
else
  echo "=========================================="
  echo "Error: The Docker command or Python script failed."
  echo "Please check the output above for errors."
  echo "=========================================="
  exit 1
fi