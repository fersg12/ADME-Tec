FROM continuumio/miniconda3:24.1.2-0

WORKDIR /app

COPY . .

ENV CONDA_NO_PLUGINS=true

RUN conda env create --solver=classic -f environment.yml

SHELL ["/bin/bash", "-c"]

EXPOSE 10000

# Aquí Render inyecta $PORT y se expande correctamente
CMD conda run --no-capture-output -n Adme-Tec streamlit run ui/main.py --server.port=$PORT --server.address=0.0.0.0
