# Use an official Python runtime as a parent image
# python:3.9-slim-buster is a good choice for smaller image size and compatibility
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir to save space during image build
# pip install --upgrade pip is good practice to ensure pip is up-to-date
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Cloud Run injects the listening port via the PORT environment variable.
# We set a default for local testing, but Uvicorn will pick up the env var.
ENV PORT 8080

# Command to run your FastAPI application with Uvicorn
# --host 0.0.0.0 is crucial for containerized apps to listen on all available network interfaces
# api:app refers to the 'app' object in 'api.py'
CMD uvicorn api:app --host 0.0.0.0 --port $PORT