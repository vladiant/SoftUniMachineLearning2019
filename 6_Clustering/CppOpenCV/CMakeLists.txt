project(OpenCV_ML_Clustering)
cmake_minimum_required(VERSION 3.1)
find_package( OpenCV REQUIRED )

aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
