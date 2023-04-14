cmake_minimum_required(VERSION 3.8)
project(robot_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(OpenCV_DIR /usr/local/lib/cmake/opencv)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenCV REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/LoadMap.srv"
  "srv/ResetMap.srv"
  "srv/SetRobotPose.srv"
  "action/NavigateToPose.action"
  DEPENDENCIES builtin_interfaces std_msgs action_msgs geometry_msgs nav_msgs
)

ament_export_dependencies(rosidl_default_runtime)

add_executable(robot_sim_server src/robot_sim_server.cpp)
target_compile_features(robot_sim_server PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(robot_sim_server rclcpp OpenCV tf2_ros builtin_interfaces std_msgs action_msgs geometry_msgs nav_msgs)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

target_link_libraries(robot_sim_server "${cpp_typesupport_target}")

install(TARGETS robot_sim_server
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
 DESTINATION share/${PROJECT_NAME}/)

# Install rviz folder
install(DIRECTORY rviz
 DESTINATION share/${PROJECT_NAME}/)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
