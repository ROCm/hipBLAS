.. meta::
  :description: hipBLAS documentation and API reference library
  :keywords: hipBLAS, rocBLAS, BLAS, ROCm, API, Linear Algebra, documentation

.. _hipblas:

********************************************************************
hipBLAS documentation
********************************************************************

hipBLAS is a BLAS marshaling library with multiple supported backends. It sits between the application and a 'worker' BLAS library, marshalling inputs into the backend library and marshalling results back to the application.
hipBLAS exports an interface that does not require the client to change, regardless of the chosen backend. Currently, it supports `rocBLAS <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_ and `cuBLAS <https://developer.nvidia.com/cublas>`_ as backends.

The code is open and hosted at: https://github.com/ROCm/hipBLAS

The hipBLAS documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

    * :ref:`install`

  .. grid-item-card:: How to

    * :ref:`hipblas-orga`
    * :ref:`hipblas-clients`
    * :ref:`contribute`

  .. grid-item-card:: API Reference

    * :ref:`api_label`
    * :ref:`deprecations`

To contribute to the documentation refer to `Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.

