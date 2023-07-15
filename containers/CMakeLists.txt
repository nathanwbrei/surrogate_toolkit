

find_program(Docker_EXECUTABLE docker)
if(NOT Docker_EXECUTABLE)
    message(STATUS "Skipping targets 'phasm_dev_env','phasm_cuda_env' because CMake cannot find `docker`")
    return()
endif()

message(STATUS "Including optional targets 'phasm_dev_env','phasm_cuda_env'")
add_custom_target(phasm_dev_env
    COMMAND PHASM_USE_CUDA=0 ${CMAKE_SOURCE_DIR}/scripts/download_deps.sh ${CMAKE_SOURCE_DIR}/downloads && ${Docker_EXECUTABLE} build --progress=plain -t phasm_dev_env -f ${CMAKE_SOURCE_DIR}/containers/phasm_dev_env/Dockerfile ${CMAKE_SOURCE_DIR}
)

add_custom_target(phasm_cuda_env
    COMMAND PHASM_USE_CUDA=1 ${CMAKE_SOURCE_DIR}/scripts/download_deps.sh ${CMAKE_SOURCE_DIR}/downloads && ${Docker_EXECUTABLE} build --progress=plain -t phasm_cuda_env -f ${CMAKE_SOURCE_DIR}/containers/phasm_cuda_env/Dockerfile ${CMAKE_SOURCE_DIR}
)
