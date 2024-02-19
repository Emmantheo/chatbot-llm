# Use an official Python runtime as a parent image
FROM python:3.10.9-slim

# Port to expose
EXPOSE 5000

# Set the working directory
WORKDIR /rag

# Copy the contents of the current directory into the container at /app
COPY . /rag

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run app.py when the container launches
CMD ["python", "rag.py", "--server.port=5000", "--server.address=0.0.0.0"]
