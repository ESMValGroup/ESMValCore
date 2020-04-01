FROM continuumio/miniconda3

# update the conda packages
RUN conda update -y conda pip

# install development tools
RUN apt-get update -y && apt-get install -y \
    build-essential \
    curl \
    unzip

# install environment packages
RUN conda env update --name base --file environment.yml
RUN pip install .

# run tests
RUN esmvaltool -h

ENTRYPOINT ["esmvaltool"]
CMD ["-h"]
