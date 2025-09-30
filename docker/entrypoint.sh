#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# Base path for all environments
ENV_BASE_PATH="/home/micromamba/RNAsign/envs"

# The first argument from the 'docker run' command
COMMAND="$1"

case "$COMMAND" in
  cpu)
    echo "--- Testing python_env_cpu ---"
    micromamba run -p "${ENV_BASE_PATH}/python_env_cpu" python --version
    ;;
  gpu)
    echo "--- Testing python_env_gpu ---"
    micromamba run -p "${ENV_BASE_PATH}/python_env_gpu" python --version
    ;;
  bedtools)
    echo "--- Testing bedtools_env ---"
    micromamba run -p "${ENV_BASE_PATH}/bedtools_env" bedtools --version
    ;;
  featurecounts)
    echo "--- Testing featurecounts_env ---"
    # Note: featureCounts uses -v for version
    micromamba run -p "${ENV_BASE_PATH}/featurecounts_env" featureCounts -v
    ;;
  bash | shell)
    # Allows you to get an interactive shell for debugging
    # The 'exec' command replaces the script process with the shell
    exec /bin/bash
    ;;
  *)
    # Default action: Print a help message
    echo "Usage: docker run <image> [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  cpu            Test the python_env_cpu environment."
    echo "  gpu            Test the python_env_gpu environment."
    echo "  bedtools       Test the bedtools_env environment."
    echo "  featurecounts  Test the featurecounts_env environment."
    echo "  bash | shell   Start an interactive bash shell."
    ;;
esac