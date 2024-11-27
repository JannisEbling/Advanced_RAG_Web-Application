FROM python:3.10-slim

# Install system dependencies including PostgreSQL client
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional PostgreSQL dependencies
RUN pip install --no-cache-dir \
    sqlalchemy>=1.4.0 \
    psycopg2-binary>=2.9.0

# Copy the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "src/app/main.py"]