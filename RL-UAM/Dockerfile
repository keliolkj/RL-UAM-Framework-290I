# Use official Python 3.10 runtime as a parent image. Use the slim version to reduce the image size.
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

ENV PYTHONPATH "${PYTHONPATH}:/app"

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 6006 available to the world outside this container
EXPOSE 6006

# Run main.py when the container launches
CMD ["python", "./src/main.py"]
