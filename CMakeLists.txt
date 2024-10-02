# NOTE: CMake support is currently a work in progress.
# At the moment it is recommended to just follow the
# instructions in the README.
cmake_minimum_required(VERSION 3.15)

project(SwirlCraft LANGUAGES CXX)

include(GNUInstallDirs)

add_library(swirl_craft INTERFACE)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(swirl_craft INTERFACE OpenMP::OpenMP_CXX)
endif()

target_include_directories(swirl_craft INTERFACE 
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
	"$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_compile_features(swirl_craft INTERFACE cxx_std_17)


install(TARGETS swirl_craft
	EXPORT swirl_craft_Targets
)

install(DIRECTORY SwirlCraft/
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/SwirlCraft"
)
