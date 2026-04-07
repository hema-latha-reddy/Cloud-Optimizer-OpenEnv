# We change from 'slim' to a more common version that is likely cached
FROM python:3.10

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy and install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the yaml file (Crucial for the "Not enough tasks" error)
COPY openenv.yaml .

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 7860

# Use the python module command you know works
CMD ["python", "-m", "server.app"]
