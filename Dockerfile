FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY src/ ./src/

# Expose the port for the API or web interface
EXPOSE 8501  # For Streamlit or Gradio

# Command to run the application
CMD ["python", "src/main.py"]