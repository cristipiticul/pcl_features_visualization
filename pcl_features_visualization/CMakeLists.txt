cmake_minimum_required(VERSION 2.8.3)
project(pcl_features_visualization)

find_package(PCL REQUIRED COMPONENTS common io features visualization)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  geometry_msgs
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS geometry_msgs roscpp  sensor_msgs std_msgs pcl_ros pcl_conversions dynamic_reconfigure
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake
)


link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pcl_features_visualizer
	src/pcl_features_visualizer.cpp
)
add_dependencies(pcl_features_visualizer
	${${PROJECT_NAME}_EXPORTED_TARGETS}
)
target_link_libraries(pcl_features_visualizer
  ${catkin_LIBRARIES}
  ${PCL_COMMON_LIBRARIES}
  ${PCL_IO_LIBRARIES}
  ${PCL_VISUALIZATION_LIBRARIES}
)
