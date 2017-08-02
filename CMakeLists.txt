cmake_minimum_required(VERSION 3.8)
project(ADMM)

set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src/)

set(SOURCE_FILES
    ${SOURCE_DIR}/BP.cpp
    ${SOURCE_DIR}/LAD.cpp
    ${SOURCE_DIR}/Lasso.cpp
    ${SOURCE_DIR}/ParLasso.cpp
    ${SOURCE_DIR}/Enet.cpp)

message("[ Source files for board_platformer ]")
foreach(MAIN_FILE ${MAIN_SOURCE_FILES})
    message(STATUS "${MAIN_FILE}")
endforeach()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(BUILD_FLAGS  "-Wall -Wextra -O3")
message("C++ flags = ${BUILD_FLAGS}")
set_target_properties(${PROJECT_NAME}
    PROPERTIES COMPILE_FLAGS "${BUILD_FLAGS}")

target_include_directories(${PROJECT_NAME}
    PRIVATE
    include/eigen/Eigen)
