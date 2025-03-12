# Step 1: Use a Python base image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements file into the container
# You should create a requirements.txt file listing all necessary dependencies
COPY requirements.txt /app/

# Step 4: Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the Flask app and model files into the container
COPY . /app/

# Step 6: Expose port 5000 (default for Flask)
EXPOSE 5000

# Step 7: Set the environment variable for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Step 8: Run the Flask app when the container starts
CMD ["flask", "run"]
