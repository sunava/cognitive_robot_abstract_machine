# Dockerfile of the base image: https://github.com/IntEL4CoRo/jupyter-ros2/blob/main/Dockerfile
FROM intel4coro/jupyter-ros2:jazzy-py3.12

USER root

# %% system packages
# pygraphviz, which krrood pulls in, compiles against the graphviz headers
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# %% the repository
ENV REPO_DIR=/home/repo
RUN mkdir -p ${REPO_DIR}
COPY . ${REPO_DIR}/
RUN git config --global --add safe.directory ${REPO_DIR}
WORKDIR ${REPO_DIR}

# %% the workspace members cramera imports
# Only these: the viewer serves scenes, the knowledge graph and EQL, and needs nothing
# of the simulation, perception or demo stack to do it. mujoco and the bullet bindings
# are import-time dependencies of semantic_digital_twin that it does not declare.
RUN python3 -m pip install \
        ./random_events \
        ./probabilistic_model \
        ./krrood && \
    python3 -m pip install mujoco giskardpy_bullet_bindings && \
    python3 -m pip install \
        ./semantic_digital_twin \
        ./coraplex \
        ./giskardpy && \
    python3 -m pip install -e ./cramera

# %% serving the viewer instead of a notebook interface
RUN python3 -m pip install jupyter-server-proxy
COPY binder/jupyter_server_config.py /tmp/cramera_jupyter_config.py
RUN mkdir -p /etc/jupyter && \
    cat /tmp/cramera_jupyter_config.py >> /etc/jupyter/jupyter_server_config.py && \
    rm /tmp/cramera_jupyter_config.py
