FROM python:3.12-slim-bookworm
WORKDIR /app

# Image is lean + Code always runs from its source, not cached bytecode.
ENV PYTHONDONTWRITEBYTECODE=1
# Logs are emitted in real-time
ENV PYTHONUNBUFFERED=1

# Install system dependencies (build-essential for some pip packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    # Add any other system libraries if needed by your specific Python packages
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy your requirements.txt file into the container
# It's good practice to do this before copying the rest of your code
# to leverage Docker's build cache.
COPY requirements.txt .
# Install Python dependencies from requirements.txt
# --no-cache-dir: prevents pip from storing cache files, reducing image size
# -U: upgrade all specified packages to the newest available version
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# TODO: cron jobs fail when application code is stale on docker image but we do not rebuild the image on code changes
# Either don't copy the code and instead mount the code directory at runtime
# or rebuild the image on code changes

# Copy the rest of your application code into the container 
# Ignore everything in .dockerignore
COPY . .