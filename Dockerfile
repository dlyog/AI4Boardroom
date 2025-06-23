# Dockerfile

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Declare build arguments for secrets
ARG USERS_B64
ARG GOOGLE_API_KEY
EXPOSE 8080
# Write secrets into the container file system
RUN echo "$USERS_B64" | base64 -d > /app/users.json \
    && echo "GOOGLE_API_KEY=$GOOGLE_API_KEY" > /app/.env

# Run the application
CMD ["python", "app_adk.py"]
