#!/bin/bash

# Filename: setup_and_install.sh

# --------------------------- Configuration ---------------------------

# Name of the Conda environment to create
ENV_NAME="hpcSeqMemVenv2" 

# Python version to use in the environment
PYTHON_VERSION="3.10"

# Path to the requirements file
REQUIREMENTS_FILE="requirementsShort.txt"

# Log files
LOG_FILE="setup_and_install.log"
ERROR_LOG_FILE="setup_and_install_errors.log"

# ---------------------------------------------------------------------

# Function to print error messages
error_exit() {
    echo "Error: $1" | tee -a "$ERROR_LOG_FILE" >&2
}

# Function to log informational messages
log_info() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Initialize log files
: > "$LOG_FILE"
: > "$ERROR_LOG_FILE"

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    error_exit "Conda is not installed or not in the PATH. Please install Conda first."
    # Continue script execution
else
    log_info "Conda is installed."
fi

# Check if the environment already exists
if conda info --envs | grep -w "$ENV_NAME" > /dev/null; then
    log_info "Conda environment '$ENV_NAME' already exists. Activating it..."
    ENV_EXISTS=true
else
    ENV_EXISTS=false
fi

# Create the environment if it doesn't exist
if [ "$ENV_EXISTS" = false ]; then
    log_info "Creating a new Conda environment named '$ENV_NAME' with Python $PYTHON_VERSION..."
    if conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION" pip; then
        log_info "Conda environment '$ENV_NAME' created successfully."
    else
        error_exit "Failed to create Conda environment '$ENV_NAME'."
        # Continue script execution
    fi
else
    log_info "Skipping environment creation since '$ENV_NAME' already exists."
fi

# Activate the Conda environment
log_info "Activating the Conda environment '$ENV_NAME'..."
# Initialize Conda for the current shell
source "$(conda info --base)/etc/profile.d/conda.sh" || {
    error_exit "Failed to source Conda."
    # Continue script execution
}

if conda activate "$ENV_NAME"; then
    log_info "Conda environment '$ENV_NAME' is now active."
else
    error_exit "Failed to activate Conda environment '$ENV_NAME'."
    # Continue script execution
fi

# Ensure pip is installed
if ! command -v pip &> /dev/null; then
    log_info "pip not found in the Conda environment. Installing pip..."
    if conda install -y pip; then
        log_info "pip installed successfully."
    else
        error_exit "Failed to install pip."
        # Continue script execution
    fi
else
    log_info "pip is already installed."
fi

# Upgrade pip to the latest version
log_info "Upgrading pip..."
if pip install --upgrade pip; then
    log_info "pip upgraded successfully."
else
    error_exit "Failed to upgrade pip."
    # Continue script execution
fi

# Check if requirements file exists
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    error_exit "Requirements file '$REQUIREMENTS_FILE' not found in the current directory."
    # Continue script execution
fi

log_info "Starting installation of Python packages from '$REQUIREMENTS_FILE'..."

# Read the requirements file line by line
while IFS= read -r package || [[ -n "$package" ]]; do
    # Trim leading/trailing whitespace
    package=$(echo "$package" | xargs)

    # Skip empty lines and comments
    if [[ -z "$package" || "$package" == \#* ]]; then
        continue
    fi

    # Extract package name and version (if specified)
    if [[ "$package" == *"=="* ]]; then
        pkg_name=$(echo "$package" | cut -d '=' -f 1)
        pkg_version=$(echo "$package" | cut -d '=' -f 3)
    else
        pkg_name="$package"
        pkg_version=""
    fi

    # Check if the package is already installed
    installed_version=$(pip show "$pkg_name" 2>/dev/null | grep ^Version: | awk '{print $2}')

    if [[ -n "$installed_version" ]]; then
        if [[ -z "$pkg_version" ]]; then
            log_info "Package '$pkg_name' is already installed (version $installed_version). Skipping installation."
            continue
        elif [[ "$installed_version" == "$pkg_version" ]]; then
            log_info "Package '$pkg_name==$pkg_version' is already installed. Skipping installation."
            continue
        else
            log_info "Package '$pkg_name' is installed with version $installed_version, but version $pkg_version is required. Upgrading..."
        fi
    else
        log_info "Package '$pkg_name' is not installed. Installing..."
    fi

    # Install the package
    if pip install "$package"; then
        log_info "Successfully installed '$package'."
    else
        error_exit "Failed to install '$package'."
        # Continue with the next package
    fi

done < "$REQUIREMENTS_FILE"

log_info "All package installations attempted."

# Extra clean up
# Check and install 'hdmf==3.8.1' only if not installed or different version
hdmf_version_required="3.8.1"
installed_hdmf_version=$(pip show hdmf 2>/dev/null | grep ^Version: | awk '{print $2}')

if [[ "$installed_hdmf_version" != "$hdmf_version_required" ]]; then
    log_info "Installing hdmf==$hdmf_version_required..."
    if pip install "hdmf==$hdmf_version_required"; then
        log_info "Successfully installed 'hdmf==$hdmf_version_required'."
    else
        error_exit "Failed to install 'hdmf==$hdmf_version_required'."
    fi
else
    log_info "hdmf==$hdmf_version_required is already installed. Skipping."
fi

# Check and install 'ipython' via conda only if not installed
if ! conda list ipython | grep -q "^ipython "; then
    log_info "Installing ipython via Conda..."
    if conda install -y ipython; then
        log_info "Successfully installed 'ipython'."
    else
        error_exit "Failed to install 'ipython' via Conda."
    fi
else
    log_info "ipython is already installed in the Conda environment. Skipping."
fi

log_info "Setup and installation completed. Check '$LOG_FILE' for details and '$ERROR_LOG_FILE' for any errors."

