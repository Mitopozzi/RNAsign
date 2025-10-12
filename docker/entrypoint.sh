#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# Base path for all environments
ENV_BASE_PATH="/home/micromamba/RNAsign/envs"

# The first argument from the 'docker run' command
COMMAND="$1"

case "$COMMAND" in
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
  "")
    # If no command is provided, print help
    echo "Usage: docker run <image> [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  gpu            Test the python_env_gpu environment."
    echo "  bedtools       Test the bedtools_env environment."
    echo "  featurecounts  Test the featurecounts_env environment."
    echo "  bash | shell   Start an interactive bash shell."
    ;;
  *)
    # CRITICAL FIX: Execute any other command passed to the container
    # This allows Nextflow to run arbitrary commands like 'bedtools genomecov ...'
    exec "$@"
    ;;
esac