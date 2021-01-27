# To build this container, go to ESMValCore root folder and execute:
# docker build -t esmvalcore:latest . -f docker/Dockerfile
FROM continuumio/miniconda3

WORKDIR /src/ESMValCore
COPY environment.yml .
RUN conda update -y conda && conda env create --name esmvaltool --file environment.yml && conda clean --all -y

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--name", "esmvaltool", "/bin/bash", "-c"]

COPY . .
RUN pip install . && pip cache purge

ENTRYPOINT ["conda", "run", "--name", "esmvaltool", "esmvaltool"]
