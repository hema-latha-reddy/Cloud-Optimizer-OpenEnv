FROM python:3.10

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install uv tool
RUN pip install --no-cache-dir uv

# 1. Copy dependency files
COPY pyproject.toml requirements.txt* ./

# 2. Install dependencies (FIXED with --system)
RUN uv pip install --system --no-cache-dir -r pyproject.toml

# 3. Copy the rest of the application
COPY . .

# Expose port
EXPOSE 7860

# Run using the module path
CMD ["python", "-m", "server.app"]
