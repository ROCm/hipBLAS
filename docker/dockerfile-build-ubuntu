# Parameters related to building hip
ARG base_image

FROM ${base_image}
LABEL maintainer="kent.knox@amd"

ARG user_uid

# Install dependent packages
# Dependencies:
# * hcc-config.cmake: pkg-config
# * tensile: python2.7, python-yaml
# * hipblas-test: gfortran, googletest
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    sudo \
    ca-certificates \
    git \
    make \
    cmake \
    pkg-config \
    gfortran \
    libboost-program-options-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# docker pipeline runs containers with particular uid
# create a jenkins user with this specific uid so it can use sudo priviledges
# Grant any member of sudo group password-less sudo privileges
RUN useradd --create-home -u ${user_uid} -o -G sudo --shell /bin/bash jenkins && \
    mkdir -p /etc/sudoers.d/ && \
    echo '%sudo   ALL=(ALL) NOPASSWD:ALL' | tee /etc/sudoers.d/sudo-nopasswd

ARG HIPBLAS_SRC_ROOT=/usr/local/src/hipBLAS

# Clone hipblas repo
RUN mkdir -p ${HIPBLAS_SRC_ROOT} && cd ${HIPBLAS_SRC_ROOT} && \
    git clone -b develop --depth=1 https://github.com/ROCmSoftwarePlatform/hipBLAS . && \

# Build client dependencies and install into /usr/local (LAPACK & GTEST)
    mkdir -p build/deps && cd build/deps && \
    cmake -DBUILD_BOOST=OFF ${HIPBLAS_SRC_ROOT}/deps && \
    make -j $(nproc) install && \
    rm -rf ${HIPBLAS_SRC_ROOT}
