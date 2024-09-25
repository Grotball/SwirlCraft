cmake_minimum_required(VERSION 3.15)

project(SwirlCraft LANGUAGES CXX)

add_library(swirl_craft INTERFACE)

target_include_directories(swirl_craft INTERFACE .)

target_compile_features(swirl_craft INTERFACE cxx_std_17)

