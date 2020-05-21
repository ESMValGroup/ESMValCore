# To build this container, go to ESMValCore root folder and execute :
# docker build -t ${TAG_OF_YOUR_CHOICE} . -f docker/Dockerfile
FROM continuumio/miniconda3:4.8.2-alpine

# update the conda packages
RUN conda update -y conda pip


# install environment packages
COPY . /home/root/source
WORKDIR /home/root/source
RUN ls
RUN conda env update --name base --file environment.yml
RUN pip install .
RUN conda clean --all -y
RUN rm -r /home/root/source

# run tests
RUN esmvaltool -h

ENTRYPOINT ["esmvaltool"]
CMD ["-h"]
