# Use the official dolfinx Docker image as base
FROM dolfinx/dolfinx:stable

# Install additional dependencies (if needed)
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev 

# Install Python dependencies
RUN pip3 install --upgrade pip

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ /app/src/

# Copy requirements and install if necessary
COPY requirements.txt /app/
RUN pip3 install -r /app/requirements.txt

# Run the solver by default
CMD ["python3", "/app/src/coupled_solver.py"]
