#!/usr/bin/python3
"""Copyright 2020-2021 Advanced Micro Devices, Inc.
Manage build and installation"""

import re
import sys
import os
import platform
import subprocess
import argparse
import pathlib
from fnmatch import fnmatchcase

args = {}
param = {}
OS_info = {}

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="""Checks build arguments""")
    # common
    parser.add_argument('-g', '--debug', required=False, default = False,  action='store_true',
                        help='Generate Debug build (optional, default: False)')
    parser.add_argument(      '--build_dir', type=str, required=False, default = "build",
                        help='Build directory path (optional, default: build)')
    parser.add_argument(      '--skip_ld_conf_entry', required=False, default = False)
    parser.add_argument(      '--static', required=False, default = False, dest='static_lib', action='store_true',
                        help='Generate static library build (optional, default: False)')
    parser.add_argument('-c', '--clients', required=False, default = False, dest='build_clients', action='store_true',
                        help='Generate all client builds (optional, default: False)')
    parser.add_argument('-i', '--install', required=False, default = False, dest='install', action='store_true',
                        help='Install after build (optional, default: False)')
    parser.add_argument(      '--cmake-darg', required=False, dest='cmake_dargs', action='append', default=[],
                        help='List of additional cmake defines for builds (optional, e.g. CMAKE)')
    parser.add_argument('-v', '--verbose', required=False, default = False, action='store_true',
                        help='Verbose build (optional, default: False)')
    # hipblas
    parser.add_argument(      '--rocm_dev', type=str, required=False, default = "",
                        help='Set specific rocm-dev version')
    parser.add_argument(      '--cpu_ref_lib', type=str, required=False, default = "blis",
                        help='Specify library to use for CPU reference code in testing (blis or lapack)')
    # rocblas/rocsolver
    parser.add_argument('-n', '--no-solver', dest='build_solver', required=False, default=True, action='store_false')
    parser.add_argument('-b', '--rocblas', dest='rocbls_version', type=str, required=False, default="",
                        help='Set a specific rocBLAS vesrion (optional)')
    parser.add_argument('--hipsdk-path', dest='hipsdk_path', type=str, required=False, default="C:/hipSDK",
                        help='Set specific path to custom build rocBLAS (optional)')

    return parser.parse_args()

def os_detect():
    global OS_info
    if os.name == "nt":
        OS_info["ID"] = platform.system()
    else:
        inf_file = "/etc/os-release"
        if os.path.exists(inf_file):
            with open(inf_file) as f:
                for line in f:
                    if "=" in line:
                        k,v = line.strip().split("=")
                        OS_info[k] = v.replace('"','')
    OS_info["NUM_PROC"] = os.cpu_count()
    print(OS_info)

def create_dir(dir_path):
    full_path = ""
    if os.path.isabs(dir_path):
        full_path = dir_path
    else:
        full_path = os.path.join( os.getcwd(), dir_path )
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
    return

def delete_dir(dir_path) :
    if (not os.path.exists(dir_path)):
        return
    if os.name == "nt":
        run_cmd( "RMDIR" , f"/S /Q {dir_path}")
    else:
        run_cmd( "rm" , f"-rf {dir_path}")

def cmake_path(os_path):
    if os.name == "nt":
        return os_path.replace("\\", "/")
    else:
        return os_path
    
def config_cmd():
    global args
    global OS_info
    cwd_path = os.getcwd()
    cmake_executable = ""
    cmake_options = []
    src_path = cmake_path(cwd_path)
    cmake_platform_opts = []
    if os.name == "nt":
        # not really rocm path as none exist, HIP_DIR set in toolchain is more important
        rocm_path = os.getenv( 'ROCM_CMAKE_PATH', "C:/github/rocm-cmake-master/share/rocm")
        cmake_executable = "cmake"
        #set CPACK_PACKAGING_INSTALL_PREFIX= defined as blank as it is appended to end of path for archive creation
        cmake_platform_opts.append( f"-DCPACK_PACKAGING_INSTALL_PREFIX=" )
        cmake_platform_opts.append( f"-DCMAKE_INSTALL_PREFIX=\"{args.hipsdk_path}\"" )
        generator = f"-G Ninja"
        cmake_options.append( generator )
        toolchain = os.path.join( src_path, "toolchain-windows.cmake" )
    else:
        rocm_path = os.getenv( 'ROCM_PATH', "/opt/rocm")
        if (OS_info["ID"] in ['centos', 'rhel']):
          cmake_executable = "cmake" # was cmake3 but now we built cmake
        else:
          cmake_executable = "cmake"
        cmake_platform_opts.append( f"-DROCM_DIR:PATH={rocm_path} -DCPACK_PACKAGING_INSTALL_PREFIX={rocm_path}" )
        cmake_platform_opts.append( f"-DCMAKE_INSTALL_PREFIX=\"hipblas-install\"" )
        toolchain = "toolchain-linux.cmake"

    print( f"Build source path: {src_path}")

    tools = f"-DCMAKE_TOOLCHAIN_FILE={toolchain}"
    cmake_options.append( tools )

    cmake_options.extend( cmake_platform_opts )

    cmake_base_options = f"-DROCM_PATH={rocm_path} -DCMAKE_PREFIX_PATH:PATH={rocm_path}" 
    cmake_options.append( cmake_base_options )

    # packaging options
    cmake_pack_options = f"-DCPACK_SET_DESTDIR=OFF" 
    cmake_options.append( cmake_pack_options )

    if os.getenv('CMAKE_CXX_COMPILER_LAUNCHER'):
        cmake_options.append( f"-DCMAKE_CXX_COMPILER_LAUNCHER={os.getenv('CMAKE_CXX_COMPILER_LAUNCHER')}" )

    cmake_options.append("-DBUILD_TESTING=OFF")

    print( cmake_options )

    # build type
    cmake_config = ""
    build_dir = os.path.abspath(args.build_dir)
    if not args.debug:
        build_path = os.path.join(build_dir, "release")
        cmake_config="Release"
    else:
        build_path = os.path.join(build_dir, "debug")
        cmake_config="Debug"

    cmake_options.append( f"-DCMAKE_BUILD_TYPE={cmake_config}" ) 

    # clean
    delete_dir( build_path )

    create_dir( os.path.join(build_path, "clients") )
    os.chdir( build_path )

    if args.static_lib:
        cmake_options.append( f"-DBUILD_SHARED_LIBS=OFF" )

    if args.skip_ld_conf_entry:
        cmake_options.append( f"-DROCM_DISABLE_LDCONFIG=ON" )

    if args.build_clients:
        cmake_build_dir = cmake_path(build_dir)
        cmake_options.append( f"-DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_DIR={cmake_build_dir} " )

    if args.build_solver:
        cmake_options.append (f"-DBUILD_WITH_SOLVER=ON")
    else:
        cmake_options.append(f"-DBUILD_WITH_SOLVER=OFF")

    # currently windows build only works with blis
    if args.cpu_ref_lib == 'blis':
        cmake_options.append( f"-DLINK_BLIS=ON" )

    cmake_options.append( f"-DHIPSDK_PATH={args.hipsdk_path}")

    if args.cmake_dargs:
        for i in args.cmake_dargs:
          cmake_options.append( f"-D{i}" )

    cmake_options.append( f"{src_path}")
    cmd_opts = " ".join(cmake_options)

    return cmake_executable, cmd_opts


def make_cmd():
    global args
    global OS_info

    make_options = []

    nproc = OS_info["NUM_PROC"]
    if os.name == "nt":
        make_executable = f"cmake.exe --build . " # ninja
        if args.verbose:
          make_options.append( "--verbose" )
        make_options.append( "--target all" )
        if args.install:
          make_options.append( "--target package --target install" )
    else:
        make_executable = f"make -j{nproc}"
        if args.verbose:
          make_options.append( "VERBOSE=1" )
        if True: # args.install:
         make_options.append( "install" )
    cmd_opts = " ".join(make_options)

    return make_executable, cmd_opts

def run_cmd(exe, opts):
    program = f"{exe} {opts}"
    print(program)
    proc = subprocess.run(program, check=True, stderr=subprocess.STDOUT, shell=True)
    return proc.returncode

def main():
    global args
    os_detect()
    args = parse_args()

    # configure
    exe, opts = config_cmd()
    run_cmd(exe, opts)

    # make
    exe, opts = make_cmd()
    run_cmd(exe, opts)

if __name__ == '__main__':
    main()

