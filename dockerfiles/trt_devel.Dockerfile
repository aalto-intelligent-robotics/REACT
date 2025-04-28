FROM react:trt AS trt_devel

# non-root username
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Developmental env stuffs
USER root
RUN apt-get update \
    && apt-get install -y unzip curl wget \
    clang cmake tmux \
    python3-venv xclip xsel libxml2-utils \
    libxcb1-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev \
    iputils-ping iproute2 net-tools \
    && rm -rf /var/lib/apt/lists/*

USER ${USERNAME}
# Get Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

RUN source $HOME/.cargo/env \
    && cargo install --locked starship zellij eza bat ripgrep du-dust zoxide yazi-fm yazi-cli jless

RUN python3 -m pip install pynvim black flake8 cmakelang

RUN python3 -m pip install progressbar loguru

USER root
RUN apt-get update \
    && apt-get install -y zsh \
    ros-noetic-rqt ros-noetic-rqt-common-plugins \
    gdb eog mpv pcl-tools libstdc++-13-dev \
    && rm -rf /var/lib/apt/lists/*
RUN chsh -s $(which zsh)

USER ${USERNAME}
SHELL ["/bin/zsh", "-c"]
COPY ./install_dev.sh ${HOME}/install_dev.sh

RUN source ${HOME}/install_dev.sh


RUN mkdir -p ${HOME}/.cache/zsh \
    && touch ${HOME}/.cache/zsh/history

RUN mkdir ${HOME}/.config \
    && chown -R ${USERNAME} ${HOME}/.config

RUN python3 -m pip install sphinx sphinx_rtd_theme
