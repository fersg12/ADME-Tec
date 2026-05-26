FROM continuumio/miniconda3:latest

WORKDIR /app

COPY . /app
COPY environment.yml /app/environment.yml

ENV CONDA_NO_PLUGINS=true

RUN conda env create -f environment.yml

# Usamos shell por defecto, así las variables de entorno se expanden
SHELL ["/bin/bash", "-c"]

EXPOSE 10000

# Aquí Render inyecta $PORT y se expande correctamente
CMD conda run --no-capture-output -n Adme-Tec streamlit run ui/main.py --server.port=$PORT --server.address=0.0.0.0
