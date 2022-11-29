#include <iostream>

#include "include/ze_api.h"
#include "CL/sycl/backend/level_zero.hpp"
#include "CL/sycl.hpp"
#include "oneapi/mkl.hpp"

#include <sycl_w.h>

void* get_current_context() {

}

void set_current_context() {

}

void print_me() {
  std::cout<<"From sycl_wrapper library"<<std::endl;
}