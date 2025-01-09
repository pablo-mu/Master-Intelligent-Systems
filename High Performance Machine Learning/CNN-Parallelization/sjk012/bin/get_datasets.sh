#!/usr/bin/env bash

# Bash options
set -o errexit
set -o nounset

# Script dir
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DST_DIR=$(readlink -f "${SCRIPT_DIR}/../datasets")

error() {
    echo $* >&2
    exit 1
}

[ -d ${DST_DIR} ] || error "Directory '${DST_DIR}' not found. Aborting!"

cd ${DST_DIR}

if [ ! -d "cifar-10-batches-py" ]; then
  wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O cifar-10-python.tar.gz
  tar -xzvf cifar-10-python.tar.gz
  rm cifar-10-python.tar.gz
else
  echo "The dataset has been already downloaded. Nothing to be done."
fi
