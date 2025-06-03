FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "schedule_tracker_gui.py", "--server.port=8501", "--server.address=0.0.0.0"]
