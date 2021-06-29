/* ************************************************************************
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "blis.h"
#include "omp.h"

void setup_blis()
{
    bli_init();
}

static int initialize_blis = (setup_blis(), 0);
