CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

set (CMAKE_CXX_STANDARD 11)

PROJECT(jump)
INCLUDE_DIRECTORIES(inc)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
MESSAGE(STATUS "OpenCV version: ${OpenCV_VERSION}")

ADD_LIBRARY(utils src/utils.cpp)
TARGET_LINK_LIBRARIES(utils ${OpenCV_LIBS} -lglog)


ADD_EXECUTABLE(jump src/main.cpp)
TARGET_LINK_LIBRARIES(jump ${OpenCV_LIBS} -lglog utils)

ADD_EXECUTABLE(test_pic src/test_pic.cpp)
TARGET_LINK_LIBRARIES(test_pic ${OpenCV_LIBS} -lglog utils)
