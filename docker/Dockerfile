FROM continuumio/miniconda3

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
