ARG PYTORCH="2.6.0"
ARG CUDA="11.8"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
# RUN rm /etc/apt/sources.list.d/cuda.list \
#     && rm /etc/apt/sources.list.d/nvidia-ml.list \
#     && apt-key del 7fa2af80 \
#     && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
#     && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install the required packages
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev htop vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install packages
RUN python3 -m pip install --upgrade pip

# Install MMEngine and MMCV
RUN pip install openmim ipykernel kaggle seaborn && \
    mim install "mmengine>=0.7.1" "mmcv>=2.0.0rc4"

# Install MMDetection
RUN conda clean --all \
    && git clone https://github.com/open-mmlab/mmdetection.git /mmdetection \
    && cd /mmdetection \
    && pip install --no-cache-dir -e .

# Patch MMDetection to allow MMCV==2.2.0 and MMEngine>=0.7.1,<1.0.0
RUN sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" /mmdetection/mmdet/__init__.py

RUN python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda)"


# Install SSH server and configure it to use only key-based login
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir -p /var/run/sshd /root/.ssh && \
    chmod 700 /root/.ssh && \
    sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

EXPOSE 22

RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate base" >> /root/.bashrc

CMD ["/usr/sbin/sshd", "-D"]
