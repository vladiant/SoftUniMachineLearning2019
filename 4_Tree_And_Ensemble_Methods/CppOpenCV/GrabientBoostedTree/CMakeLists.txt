project(OpenCV_ML_Boosting)
cmake_minimum_required(VERSION 3.1)
find_package( OpenCV REQUIRED )

configure_file(adult.data . COPYONLY)

aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
