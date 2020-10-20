/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef _HIPBLAS_PARSE_DATA_H
#define _HIPBLAS_PARSE_DATA_H

#include <string>

// Parse --data and --yaml command-line arguments
bool hipblas_parse_data(int& argc, char** argv, const std::string& default_file = "");

#endif
