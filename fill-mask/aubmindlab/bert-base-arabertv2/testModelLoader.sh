#!/bin/sh

pip install -r /server/requirements.txt

# IMPORTANT: This locks in the dependencies so that the production image's copy never changes over time

# NOTE this may run several times during loading debugging

# Run #1: converts from unversioned dependencies to versioned dependencies and adds anything that's present
# from the base container, forcing any new dependencies to try to avoid upgrading or downgrading packages for compatability

# Run #2: then adds whatever the version number is for the package resolved by the error resolver

# Subsequent runs: then repeat this pattern

# NOTE We gain the benefit of other packages not forcing an updgrade unless necessary
# Also, by not explicitly versioning packages we ensure we're using the latest packages when the error loop runs
# Some models will likely require logic that downgrades or upgrades packages
# pip freeze -r /server/requirements.txt > /server/requirements.txt

python3 /server/model_loader.py