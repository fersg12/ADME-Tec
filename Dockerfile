FROM continuumio/miniconda3:latest

WORKDIR /app

COPY . /app
COPY environment.yml /app/environment.yml

# Usamos shell por defecto, así las variables de entorno se expanden
SHELL ["/bin/bash", "-c"]

# Ensure conda is up-to-date and create the environment non-interactively
RUN conda update -n base -c defaults conda -y && \
	conda env create -f environment.yml

EXPOSE 10000

# Aquí Render inyecta $PORT y se expande correctamente
CMD conda run --no-capture-output -n Adme-Tec streamlit run ui/main.py --server.port=$PORT --server.address=0.0.0.0
