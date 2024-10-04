cmake_minimum_required(VERSION 3.18)

project(SwirlCraft LANGUAGES CXX)


option(SWIRL_CRAFT_ENABLE_GPU "enable gpu support" OFF)

set(ENABLE_GPU_SUCCESS 0)

include(GNUInstallDirs)
include(CheckLanguage)


if(${SWIRL_CRAFT_ENABLE_GPU})
	check_language(CUDA)	
	if(CMAKE_CUDA_COMPILER)
		enable_language(CUDA)
		find_package(CUDAToolkit REQUIRED)
		set(ENABLE_GPU_SUCCESS 1)

		set(CMAKE_CUDA_STANDARD 17)
		set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	else()
		message("Failed to find CUDA compiler. GPU will not be used.")
	endif()
endif()

add_library(swirl_craft INTERFACE)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(swirl_craft INTERFACE OpenMP::OpenMP_CXX)
endif()


if(${ENABLE_GPU_SUCCESS})
	add_library(swirl_craft_gpu src/PressureSolve/jacobi_gpu.cu SwirlCraft/PressureSolve/jacobi_gpu.h SwirlCraft/grid.h SwirlCraft/arch.h)
	target_include_directories(swirl_craft_gpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
	target_compile_options(swirl_craft_gpu PRIVATE "-DSWIRL_CRAFT_GPU_ENABLED=1")
	target_compile_options(swirl_craft INTERFACE "-DSWIRL_CRAFT_GPU_ENABLED=1")
	target_link_libraries(swirl_craft INTERFACE swirl_craft_gpu)
	target_link_libraries(swirl_craft INTERFACE CUDA::cudart)
endif()

target_include_directories(swirl_craft INTERFACE 
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
	"$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_compile_features(swirl_craft INTERFACE cxx_std_17)


install(TARGETS swirl_craft
	EXPORT swirl_craft_Targets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
)

install(DIRECTORY SwirlCraft/
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/SwirlCraft"
)
