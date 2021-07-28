FROM nvidia/cuda:9.0-cudnn7-devel

RUN ln -sf /usr/local/cuda-9.2 /usr/local/cuda
# Add some dependencies
RUN apt-get clean && apt-get update -y -qq

RUN apt-get update && apt-get install -y \
	wget \
	vim \
	bzip2

RUN apt-get install -y curl git build-essential

RUN apt-get update && apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev
RUN apt-get install -y --no-install-recommends libboost-all-dev


ENV LATEST_CONDA "5.2.0"
ENV PATH="/root/anaconda3/bin:${PATH}"

RUN curl --silent -O https://repo.anaconda.com/archive/Anaconda3-$LATEST_CONDA-Linux-x86_64.sh \
    && bash Anaconda3-$LATEST_CONDA-Linux-x86_64.sh -b -p /root/anaconda3

RUN /bin/bash -c "source /root/anaconda3/etc/profile.d/conda.sh"
RUN conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
#RUN conda install -y cudnn=7.1 -c anaconda
RUN pip install tensorflow-gpu==1.12.0
RUN conda install -y protobuf
RUN pip install easydict
RUN pip install opencv-python
RUN pip install jsonrpcserver
RUN pip install flask-cors
RUN pip install pytorch-transformers
RUN pip install git+https://github.com/JiahuiYu/neuralgym
RUN apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
RUN pip install pydicom
RUN pip install tqdm
RUN pip install dicom2nifti
RUN pip install nibabel
RUN pip install monai==0.2.0
RUN pip install pytorch-ignite==0.3.0
#RUN dpkg -i /vqa-server/nccl-repo-ubuntu1604-2.5.6-ga-cuda9.0_1-1_amd64.deb && apt-get update && apt-get install  -y libnccl2 libnccl-dev
RUN pip install scikit-image==0.17.2
RUN pip install --ignore-installed pyradiomics
RUN pip install scikit-learn==0.23.2
RUN pip install pynrrd
RUN pip install flask_caching
RUN pip install flask_cachecontrol

#RUN apt-get install -y openssh-server
#RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test
#RUN echo 'test:test' | chpasswd
#RUN service ssh start
#EXPOSE 22

ENV CUDA_VISIBLE_DEVICES="4"
ENV FLASK_ENV="development"

EXPOSE 5010

COPY ./ /usr/src/app/
WORKDIR /usr/src/app/
