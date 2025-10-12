# --- STAGE 1: Builder ---
# This stage does all the "messy" work and will be discarded.
FROM mambaorg/micromamba:1.4.2 AS builder

# Set the final destination path for the environments
ENV CONDA_env_PATH=/home/micromamba/RNAsign/envs

# Create the final destination directory
RUN mkdir -p ${CONDA_env_PATH}

# Create your temporary directory for the archives
RUN mkdir ./temp_archives

# Copy the archives into your temporary directory
COPY --chown=micromamba:micromamba bedtools_env.tar.gz featurecounts_env.tar.gz python_env_gpu.tar.gz ./temp_archives/

# Unpack the archives from ./temp_archives into the final destination.

RUN \
    mkdir -p ${CONDA_env_PATH}/python_env_gpu && tar -xzvf ./temp_archives/python_env_gpu.tar.gz --no-same-owner -C ${CONDA_env_PATH}/python_env_gpu && \
    mkdir -p ${CONDA_env_PATH}/bedtools_env && tar -xzvf ./temp_archives/bedtools_env.tar.gz --no-same-owner -C ${CONDA_env_PATH}/bedtools_env && \
    mkdir -p ${CONDA_env_PATH}/featurecounts_env && tar -xzvf ./temp_archives/featurecounts_env.tar.gz --no-same-owner -C ${CONDA_env_PATH}/featurecounts_env


# --- STAGE 2: The Clean Image ---
# Start fresh from the same clean base image.
FROM mambaorg/micromamba:1.4.2

# Set the environment variable again for the final image
ENV CONDA_env_PATH=/home/micromamba/RNAsign/envs

# This is the key step: Copy ONLY the unpacked environments from the 'builder' stage.
# The ./temp_archives directory and the large .tar.gz files are left behind and discarded.
COPY --from=builder ${CONDA_env_PATH} ${CONDA_env_PATH}

# Copy the entrypoint script and make it executable.
COPY --chmod=755 entrypoint.sh /usr/local/bin/entrypoint.sh

# Set the entrypoint to the full path of the script.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]