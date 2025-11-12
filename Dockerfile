FROM python:3.12-slim

WORKDIR /app

# Install Google Cloud SDK (for gsutil)
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y google-cloud-sdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY . /app

# Install Python dependencies
RUN uv sync --frozen

# Make startup script executable
RUN chmod +x /app/download_and_train.sh

# Use startup script as entrypoint
ENTRYPOINT ["/app/download_and_train.sh"]
