# To build this container, go to ESMValCore root folder and execute:
# docker build -t esmvalcore:latest . -f docker/Dockerfile
FROM continuumio/miniconda3

# update the conda packages
RUN conda update -y conda pip


# install environment packages
COPY . /home/root/source
WORKDIR /home/root/source
RUN conda env update --name base --file environment.yml && conda clean --all -y
RUN pip install .
RUN rm -r /home/root/source

# run tests
RUN esmvaltool -h

ENTRYPOINT ["esmvaltool"]
