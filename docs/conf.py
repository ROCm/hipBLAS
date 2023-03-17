# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess
from rocm_docs import ROCmDocs

getVersionCmd = r'sed -n -e "s/^.*VERSION_STRING.* \"\([0-9\.]\{1,\}\).*/\1/p" ../CMakeLists.txt'
version = subprocess.getoutput(getVersionCmd)
if len(version) == 0:
    version = '1.0.0' # optional version number override if CMakeLists.txt does not have the version

docs_core = ROCmDocs("hipBLAS {}".format(version))
docs_core.run_doxygen()
docs_core.setup()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
