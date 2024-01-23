
# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app/training

# Copy the current directory contents into the container at /app
COPY . .
#COPY training/models /app/training/models

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variables
ENV MODEL_PATH="C:/Users/Vishnu Kumar/Plant Disease/training/Dockerfile"

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Run app.py when the container launches
CMD ["python", "app.py"]
