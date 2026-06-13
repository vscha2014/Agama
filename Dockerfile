FROM ubuntu:24.04

LABEL description="AGAMA Galactic Dynamics Framework"

ENV DEBIAN_FRONTEND=noninteractive
# TZ убираем отсюда — зададим в конце один раз

# ─── 1. Системные зависимости ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    libopenblas-dev \
    liblapack-dev \
    libsuitesparse-dev \
    libeigen3-dev \
    wget \
    curl \
    unzip \
    git \
    gnupg2 \
    tar \
    gzip \
    ca-certificates \
    gosu \
    && ln -sf /usr/sbin/gosu /usr/local/bin/gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ─── 2. Python-пакеты ДО сборки AGAMA ────────────────────────────────────────
RUN pip3 install --break-system-packages \
    cvxopt==1.3.2

RUN pip3 install --break-system-packages \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install --break-system-packages \
    botorch \
    ax-platform

RUN pip3 install --break-system-packages \
    mgefit \
    powerbin \
    requests \
    scikit-learn

RUN python3 -c "import numpy, scipy, torch, botorch, requests, sklearn; \
    print('OK numpy:', numpy.__version__); \
    print('OK torch:', torch.__version__); \
    print('OK botorch:', botorch.__version__); \
    print('OK requests:', requests.__version__)"

# ─── 3. Структура директорий ──────────────────────────────────────────────────
RUN mkdir -p /opt/agama-deps/lib \
             /opt/agama-deps/include \
             /opt/agama-deps/bin \
             /tmp/build

WORKDIR /tmp/build

# ─── 4. GLPK 5.0 ─────────────────────────────────────────────────────────────
RUN wget -q https://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz && \
    tar -zxf glpk-5.0.tar.gz && \
    cd glpk-5.0 && \
    ./configure \
        --prefix=/opt/agama-deps \
        --disable-shared \
        --enable-static \
        CFLAGS="-O2 -fPIC" \
        CXXFLAGS="-O2 -fPIC" && \
    make -j$(nproc) && \
    make install && \
    cd /tmp/build && rm -rf glpk-5.0*

# ─── 5. GSL 2.7 ──────────────────────────────────────────────────────────────
RUN wget -q https://ftp.gnu.org/gnu/gsl/gsl-2.7.tar.gz && \
    tar -zxf gsl-2.7.tar.gz && \
    cd gsl-2.7 && \
    ./configure \
        --prefix=/opt/agama-deps \
        --disable-shared \
        --enable-static \
        CFLAGS="-O2 -fPIC" && \
    make -j$(nproc) && \
    make install && \
    cd /tmp/build && rm -rf gsl-2.7*

# ─── 6. CVXOPT заголовок ─────────────────────────────────────────────────────
RUN wget -q https://pypi.org/packages/source/c/cvxopt/cvxopt-1.3.2.tar.gz \
        -O /opt/agama-deps/cvxopt-1.3.2.tar.gz && \
    tar xf /opt/agama-deps/cvxopt-1.3.2.tar.gz -C /opt/agama-deps/ && \
    rm /opt/agama-deps/cvxopt-1.3.2.tar.gz && \
    test -f /opt/agama-deps/cvxopt-1.3.2/src/C/cvxopt.h && \
    echo "OK: cvxopt.h найден"

# ─── 7. UNSIO ────────────────────────────────────────────────────────────────
RUN git clone https://github.com/GalacticDynamics-Oxford/unsio \
        /opt/agama-deps/unsio && \
    cd /opt/agama-deps/unsio && \
    CXXFLAGS="-O2 -fPIC" CFLAGS="-O2 -fPIC" make -j$(nproc)

# ─── 8. Проверка зависимостей ────────────────────────────────────────────────
RUN test -f /opt/agama-deps/lib/libgsl.a      && echo "OK: libgsl.a"      && \
    test -f /opt/agama-deps/lib/libgslcblas.a && echo "OK: libgslcblas.a" && \
    test -f /opt/agama-deps/lib/libglpk.a     && echo "OK: libglpk.a"     && \
    test -f /opt/agama-deps/cvxopt-1.3.2/src/C/cvxopt.h && \
                                                 echo "OK: cvxopt.h"       && \
    test -f /opt/agama-deps/unsio/libunsio.a  && echo "OK: libunsio.a"    && \
    test -f /opt/agama-deps/unsio/libnemo.a   && echo "OK: libnemo.a"     && \
    test -d /usr/include/eigen3/Eigen          && echo "OK: eigen3"        && \
    which gfortran                             && echo "OK: gfortran"      && \
    python3 -c "import numpy; print('OK: numpy', numpy.__version__)"

# ─── 9. Makefile.local ────────────────────────────────────────────────────────
COPY Makefile.local /opt/agama-deps/Makefile.local

# ─── 10. Сборка AGAMA ────────────────────────────────────────────────────────
RUN git clone --depth=1 \
        https://github.com/GalacticDynamics-Oxford/Agama.git \
        /opt/agama && \
    cp /opt/agama-deps/Makefile.local /opt/agama/ && \
    cd /opt/agama && \
    make -j$(nproc) agama.so agama.a && \
    make

RUN test -f /opt/agama/agama.so && echo "OK: agama.so собран"

# ─── 11. Установка AGAMA через pip ───────────────────────────────────────────
RUN cd /opt/agama && \
    pip3 install --break-system-packages \
        --no-build-isolation \
        --config-settings="--build-option=--yes" \
        . \
    || python3 setup.py install --prefix=/usr/local --yes

# ─── 12. Проверка AGAMA ──────────────────────────────────────────────────────
COPY check_agama.py /tmp/check_agama.py
RUN python3 /tmp/check_agama.py

# ─── 13. Проверка всех импортов ──────────────────────────────────────────────
COPY check_imports.py /tmp/check_imports.py
RUN python3 /tmp/check_imports.py

# ─── 14. rclone ──────────────────────────────────────────────────────────────
RUN curl -fsSL https://rclone.org/install.sh | bash && \
    rclone version

# ─── 15. Права доступа ───────────────────────────────────────────────────────
RUN chmod -R 755 /opt/agama /opt/agama-deps

# ─── Очистка ─────────────────────────────────────────────────────────────────
RUN rm -rf /tmp/build /tmp/check_agama.py /tmp/check_imports.py

# ─── Переменные окружения ─────────────────────────────────────────────────────
ENV AGAMA_DIR=/opt/agama
ENV LD_LIBRARY_PATH=/opt/agama-deps/lib
ENV PATH=/opt/agama-deps/bin:/usr/local/bin:/usr/bin:/bin
ENV TZ=Europe/Moscow
# PYTHONPATH намеренно не задан — pip установил agama как пакет

# ─── entrypoint ───────────────────────────────────────────────────────────────
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
