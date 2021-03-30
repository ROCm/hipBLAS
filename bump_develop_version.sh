#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_HIPBLAS_VERSION="0.44.0"
NEW_HIPBLAS_VERSION="0.45.0"

OLD_MINIMUM_ROCBLAS_VERSION="2.38.0"
NEW_MINIMUM_ROCBLAS_VERSION="2.39.0"

OLD_MINIMUM_ROCSOLVER_VERSION="3.12.0"
NEW_MINIMUM_ROCSOLVER_VERSION="3.13.0"

sed -i "s/${OLD_HIPBLAS_VERSION}/${NEW_HIPBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_MINIMUM_ROCBLAS_VERSION}/${NEW_MINIMUM_ROCBLAS_VERSION}/g" library/CMakeLists.txt
sed -i "s/${OLD_MINIMUM_ROCSOLVER_VERSION}/${NEW_MINIMUM_ROCSOLVER_VERSION}/g" library/CMakeLists.txt
