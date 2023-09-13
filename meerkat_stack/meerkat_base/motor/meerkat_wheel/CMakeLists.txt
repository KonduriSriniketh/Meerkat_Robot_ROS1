cmake_minimum_required(VERSION 3.0.2)
project(meerkat_wheel)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  oriental_driver
  roscpp
  std_msgs
  nav_msgs
  geometry_msgs
  tf
  meerkat_msgs
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)
catkin_package(
  INCLUDE_DIRS include
  #  LIBRARIES claw_drive
  CATKIN_DEPENDS roscpp rospy
  #  DEPENDS system_lib
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)



file(GLOB source
  include/meerkat_wheel/*.h
  src/DragonPaw.cpp
)


add_library(${PROJECT_NAME}
  ${source}
)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} )

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


add_executable(${PROJECT_NAME}_node src/DragonAction.cpp)
add_executable(odometry_node src/Odometry.cpp)
add_executable(fused_odometry_node src/Odometry.cpp)


target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}
)

target_link_libraries(odometry_node
  ${PROJECT_NAME}
)

target_link_libraries(fused_odometry_node
  ${PROJECT_NAME}
)



install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)


install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
PATTERN ".svn" EXCLUDE)
