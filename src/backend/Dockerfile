# Dockerfile for the backend in Flask
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    alsa-utils \
    libasound2 \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy the project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the backend port (for example 5000)
EXPOSE 4000

# Command to start Flask
CMD ["python", "app.py"]
