# ########################################################################
# Copyright 2021-2022 Advanced Micro Devices, Inc.
# ########################################################################
cmake_minimum_required(VERSION 3.16.8)

#Include Directory Paths for - reorg file/folder wrapper, and symlink wrapper resp.
set(PROJECT_FILE_REORG_INC_DIR    ${PROJECT_SOURCE_DIR}/library/include/hipblas)
set(PROJECT_WRAPPER_TEMPLATE_FILE ${PROJECT_SOURCE_DIR}/header.hpp.in)
set(PROJECT_WRAPPER_INC_DIR       ${PROJECT_BINARY_DIR}/hipblas/include)
set(PROJECT_SLINK_WRAPPER_INC_DIR ${PROJECT_BINARY_DIR}/include)

#Relative Path for wrapper and symlink wrapper (hipblas/include and include/ resp.) files generated 
set(PROJECT_WRAPPER_INC_RELATIVE_PATH     "${include_statements}#include \"../../include/hipblas")
set(PROJECT_SLINK_INC_RELATIVE_PATH       "${include_statements}#include \"./hipblas")


# Function for Generating Wrapper Headers for Backward Compatibilty
# Generates wrapper for all *.h files in include/hipblas folder.
# No Arguments to function.
# Wrapper are generated under hipblas/include and symlink under include/.
function (package_gen_bkwdcomp_hdrs)
	# Get list of *.h files in folder include/hipblas 
	file(GLOB include_files ${PROJECT_FILE_REORG_INC_DIR}/*.h)
	# Convert the list of files into #includes
	foreach(include_file ${include_files})
		get_filename_component(file_name ${include_file} NAME)
		set(include_statements "${PROJECT_WRAPPER_INC_RELATIVE_PATH}/${file_name}\"\n")
		configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_INC_DIR}/${file_name}")
		unset(include_statements)
		#Generate Symlink Wrapper
		set(include_statements "${PROJECT_SLINK_INC_RELATIVE_PATH}/${file_name}\"\n")
		configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_SLINK_WRAPPER_INC_DIR}/${file_name}")
		unset(include_statements)
	endforeach()
endfunction()


# Function for Generating Wrapper Header for Backward Compatibilty
# Generates wrapper for gen_file_name given as input 
# gen_file_name -  Arguments to function (absolute file name) to give as input file for which wrapper header is generated
# Wrapper generated for the input file under hipblas/include and symlink under include/.
function (package_gen_bkwdcomp_hdrfile gen_file_name)
        set(include_file ${gen_file_name})  
	get_filename_component( file_name ${include_file} NAME)
	set(include_statements "${PROJECT_WRAPPER_INC_RELATIVE_PATH}/${file_name}\"\n")
	configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_WRAPPER_INC_DIR}/${file_name}")
	unset(include_statements)
	set(include_statements "${PROJECT_SLINK_INC_RELATIVE_PATH}/${file_name}\"\n")
	configure_file(${PROJECT_WRAPPER_TEMPLATE_FILE} "${PROJECT_SLINK_WRAPPER_INC_DIR}/${file_name}")
	unset(include_statements)
endfunction()

