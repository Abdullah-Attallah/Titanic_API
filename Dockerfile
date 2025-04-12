# Use official Python image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && rm -rf ~/.cache

# Copy the app directory to the container
COPY app/ ./app

# Expose port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
