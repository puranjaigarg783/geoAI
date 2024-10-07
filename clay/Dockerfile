# Use Miniforge3 base image with Mamba
FROM condaforge/miniforge3:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the environment.yml to the container
COPY environment.yml /app

# Install dependencies using Mamba
RUN mamba env create -f environment.yml

# Activate the environment
ENV PATH /opt/conda/envs/your_env_name/bin:$PATH

# Copy the rest of the application code
COPY . /app

# Set the default command to run the application
CMD ["python", "your_main_script.py"]
