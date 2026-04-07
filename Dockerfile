# Use a very standard, cached base image
FROM python:3.10-slim

# Set environment variables to prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Increase timeout for pip to handle slow network connections during build
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the standard port
EXPOSE 7860

# Use the command that was working for you in Phase 1
CMD ["python", "-m", "server.app"]
