cmake_minimum_required (VERSION 3.8)

project("GigaLearnCPP")

include_directories("${PROJECT_SOURCE_DIR}/src/")

# LibTorch: prefer CMAKE_PREFIX_PATH from parent, then project-local folders
# Order: 1) env LIBTORCH_PATH 2) <project_root>/libtorch 3) <GigaLearnCPP>/libtorch
if(DEFINED ENV{LIBTORCH_PATH} AND EXISTS "$ENV{LIBTORCH_PATH}")
	list(APPEND CMAKE_PREFIX_PATH "$ENV{LIBTORCH_PATH}")
	message(STATUS "Using LibTorch from LIBTORCH_PATH: $ENV{LIBTORCH_PATH}")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/libtorch")
	list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
	message(STATUS "Using local libtorch: ${CMAKE_SOURCE_DIR}/libtorch")
elseif(EXISTS "${PROJECT_SOURCE_DIR}/libtorch")
	list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/libtorch")
	message(STATUS "Using local libtorch: ${PROJECT_SOURCE_DIR}/libtorch")
endif()

# Workaround: CUDA 12+/13 no longer provides CUDA::nvToolsExt (replaced by NVTX3),
# but libtorch's cuda.cmake expects it. Create stub target; libtorch bundles nvToolsExt64_1.dll.
if(NOT TARGET CUDA::nvToolsExt)
	add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
	message(STATUS "Created stub CUDA::nvToolsExt for libtorch CUDA 13 compatibility")
endif()

# CUDA arches: 12.0 = RTX 5090 (sm_120); compute_125 not in CUDA 12.9
if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
	set(TORCH_CUDA_ARCH_LIST "7.5;8.0;8.6;8.9;9.0;9.0a;12.0" CACHE STRING "CUDA arches (12.0 for RTX 5090)")
endif()

# Add libtorch (https://pytorch.org/cppdocs/installing.html#minimal-example)
find_package(Torch REQUIRED)
# Remove --Werror cross-execution-space-call so our device-only .cu files compile (libtorch cuda.cmake adds it for MSVC)
string(REPLACE "--Werror cross-execution-space-call" "" TORCH_CXX_FLAGS_FILTERED "${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS_FILTERED}")
string(REPLACE " --Werror cross-execution-space-call" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" CACHE STRING "" FORCE)

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp" "libsrc/*.cpp" "libsrc/.h" "libsrc/.hpp")

if (TORCH_CUDA_LIBRARIES)
	file(GLOB_RECURSE FILES_CUDA "src/*.cu")
	set_source_files_properties(${FILES_CUDA} PROPERTIES LANGUAGE CUDA)
	list(APPEND FILES_SRC ${FILES_CUDA})
endif()

add_library(GigaLearnCPP SHARED ${FILES_SRC})
target_compile_definitions(GigaLearnCPP PRIVATE -DWITHIN_GGL)
target_include_directories(GigaLearnCPP PUBLIC "src/public")
target_include_directories(GigaLearnCPP PRIVATE "src/private")

# Include libtorch (filter out NOTFOUND entries that break linking on some Linux/CUDA setups)
set(_torch_libs ${TORCH_LIBRARIES})
list(FILTER _torch_libs EXCLUDE REGEX "NOTFOUND")
target_link_libraries(GigaLearnCPP PRIVATE ${_torch_libs})

if (TORCH_CUDA_LIBRARIES)
	message("Enabling CUDA support...")
	target_compile_definitions(GigaLearnCPP PRIVATE -DRG_CUDA_SUPPORT)
	# Note: LibTorch adds --Werror cross-execution-space-call. --Wno-error is not a valid nvcc option.
	# CUDA 13.0 MSBuild targets require CudaToolkitDir with trailing slash (else v13.0+bin = v13.0bin)
	# CudaToolkitDir is Windows/VS only; skip on Linux to avoid regex errors
	if(MSVC AND CUDA_TOOLKIT_ROOT_DIR AND NOT "${CUDA_TOOLKIT_ROOT_DIR}" STREQUAL "")
		string(REGEX REPLACE "/+$" "" _cuda_dir "${CUDA_TOOLKIT_ROOT_DIR}")
		set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_GLOBAL_CudaToolkitDir "${_cuda_dir}/")
	endif()
	if(MSVC AND CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR AND NOT "${CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR}" STREQUAL "")
		string(REGEX REPLACE "/+$" "" _cuda_dir "${CMAKE_VS_PLATFORM_TOOLSET_CUDA_CUSTOM_DIR}")
		set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_GLOBAL_CudaToolkitDir "${_cuda_dir}/")
	endif()
endif()

# Set C++ version to 20
set_target_properties(GigaLearnCPP PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnCPP PROPERTIES CXX_STANDARD 20)

# Optional: Redis for scalable distributed (3000+ workers)
option(GIGL_USE_REDIS "Use Redis for distributed training" ON)
if(GIGL_USE_REDIS)
	include(FetchContent)
	FetchContent_Declare(hiredis
		GIT_REPOSITORY https://github.com/redis/hiredis
		GIT_TAG v1.3.0
	)
	set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
	set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Required for linking into shared lib" FORCE)
	FetchContent_MakeAvailable(hiredis)
	target_compile_definitions(GigaLearnCPP PRIVATE GIGL_REDIS)
	target_link_libraries(GigaLearnCPP PRIVATE hiredis)
	target_include_directories(GigaLearnCPP PRIVATE ${hiredis_SOURCE_DIR})
	message(STATUS "Redis (hiredis) enabled for distributed training")
endif()

# Include RLGymCPP
add_subdirectory(RLGymCPP)
target_link_libraries(GigaLearnCPP PUBLIC RLGymCPP)

# Include JSON
target_include_directories(GigaLearnCPP PUBLIC "${PROJECT_SOURCE_DIR}/libsrc/json")

# Include python
find_package(Python COMPONENTS Interpreter Development)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(GigaLearnCPP PUBLIC ${PYTHON_LIBRARIES})
message("Found Python:")
message("PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
message("PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
message("Python_RUNTIME_LIBRARY_DIRS: ${Python_RUNTIME_LIBRARY_DIRS}")
message("Python_EXECUTABLE: ${Python_EXECUTABLE}")
add_definitions(-DPY_EXEC_PATH="${Python_EXECUTABLE}") # Give C++ access to the executable path

# Include pybind11
add_subdirectory(pybind11)
target_link_libraries(GigaLearnCPP PUBLIC pybind11::embed)

# MSVC fails to find python DLLs even through they are in my path. Good job MSVC. Well done.
# This copies the the python DLLs to the output directory
if (MSVC)
    file(GLOB PYTHON_DLLS "${Python_RUNTIME_LIBRARY_DIRS}/*.dll")
	message("Adding Python DLLS: ${PYTHON_DLLS}")
    add_custom_command(TARGET GigaLearnCPP
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${PYTHON_DLLS}
                 $<TARGET_FILE_DIR:GigaLearnCPP>)
endif (MSVC)

# Make our python files copy over to our build dir
configure_file("./python_scripts/__init__.py" "../python_scripts/__init__.py" COPYONLY)
configure_file("./python_scripts/metric_receiver.py" "../python_scripts/metric_receiver.py" COPYONLY)
configure_file("./python_scripts/render_receiver.py" "../python_scripts/render_receiver.py" COPYONLY)

# MSVC sometimes won't link to the libtorch DLLs unless you do this
# This is also from https://pytorch.org/cppdocs/installing.html#minimal-example
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
	message("Adding TORCH_DLLS: ${TORCH_DLLS}")
    add_custom_command(TARGET GigaLearnCPP
                 POST_BUILD
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                 ${TORCH_DLLS}
                 $<TARGET_FILE_DIR:GigaLearnCPP>)
endif (MSVC)