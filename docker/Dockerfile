# To build this container, go to ESMValCore root folder and execute:
# docker build -t esmvalcore:latest . -f docker/Dockerfile
FROM condaforge/mambaforge

WORKDIR /src/ESMValCore
COPY environment.yml .
RUN mamba env create --name esmvaltool --file environment.yml && conda clean --all -y

COPY . .
RUN conda run --name esmvaltool pip install --no-cache .

ENTRYPOINT ["conda", "run", "--name", "esmvaltool", "esmvaltool"]
