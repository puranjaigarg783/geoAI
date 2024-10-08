# Use Miniforge3 base image with Mamba
FROM condaforge/miniforge3:latest

# Set the working directory inside the container
WORKDIR /app

# Install git and other required tools
RUN apt-get update && apt-get install -y git


# Clone the Clay model repository
RUN git clone https://github.com/Clay-foundation/model /app

# Navigate to the repository folder
WORKDIR /app/model

# Copy your custom environment.yml file into the container
COPY environment.yml /app/model/environment.yml

# Install dependencies using Mamba and create the environment from your environment.yml
RUN mamba env create --file environment.yml

# Activate the environment by updating the PATH
ENV PATH /opt/conda/envs/claymodel/bin:$PATH

# Install Jupyter kernel and run JupyterLab
# Install Jupyter kernel
RUN python -m ipykernel install --user --name claymodel


RUN mamba update -n claymodel -c conda-forge --all

# Expose the port for JupyterLab
EXPOSE 8889

# Set the working directory inside the container
WORKDIR /app

# Set the default command to run JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8889", "--no-browser", "--allow-root"]
