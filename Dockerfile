# Dockerfile

FROM python:3.11-slim

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

# Copy application code
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose Cloud Run default port
EXPOSE 8080

# Runtime secret injection
CMD bash -c '\
  echo "$USERS_B64" | base64 -d > /app/users.json && \
  echo "GOOGLE_API_KEY=$GOOGLE_API_KEY" > /app/.env && \
  python app_adk.py'
