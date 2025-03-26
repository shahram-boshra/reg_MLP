# Use a Python base image with PyTorch
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy requirements.txt if it exists
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the command to run your script. Replace "your_script_name.py" with the actual filename.
CMD ["python", "your_script_name.py"]