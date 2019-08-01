FROM nvcr.io/nvidia/tensorflow:18.06-py3

WORKDIR /home

COPY *.py ./
COPY LICENSE .
COPY bbbc014_labels.npy .

RUN mkdir images_bbbc014 && mkdir images_bbbc021 && mkdir act_max_output

