# Dockerfile for Spark Master/Worker with NumPy

# Use the base image provided
FROM bde2020/spark-master:3.0.0-hadoop3.2
# Or use FROM bde2020/spark-worker:3.0.0-hadoop3.2 for a separate worker Dockerfile

# Switch to root user to install packages
USER root

# Install necessary build tools (gcc, python3-dev, build-base) and pip using apk
RUN apk update && \
    apk add --no-cache gcc python3-dev build-base py3-pip

# Check python and pip versions being used
RUN echo "--- Checking Python and Pip ---"
RUN which python3
RUN python3 --version
RUN which pip3
RUN pip3 --version
RUN echo "------------------------------"

# Install NumPy using python3 -m pip to ensure it uses the correct pip
RUN python3 -m pip install --no-cache-dir numpy

# Verify NumPy installation within the build
RUN echo "--- Verifying NumPy installation ---"
RUN python3 -c "import numpy; print('NumPy successfully imported, version:', numpy.__version__)"
RUN echo "------------------------------------"

# Copy your application code
COPY ./src /app

# No USER spark or chown needed

# The original entrypoint/command should still be inherited