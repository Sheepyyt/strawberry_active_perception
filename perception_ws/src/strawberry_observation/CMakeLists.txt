cmake_minimum_required(VERSION 3.8)
project(strawberry_observation)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS calib3d core imgproc)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(strawberry_perception_interfaces REQUIRED)

add_library(observation_processing src/observation_processor.cpp)
target_include_directories(observation_processing
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(observation_processing
  cv_bridge
  rclcpp
  sensor_msgs
)
target_link_libraries(observation_processing ${OpenCV_LIBS})

add_executable(strawberry_observation_node src/strawberry_observation_node.cpp)
target_include_directories(strawberry_observation_node
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(strawberry_observation_node
  builtin_interfaces
  geometry_msgs
  rclcpp
  sensor_msgs
  strawberry_perception_interfaces
)
target_link_libraries(strawberry_observation_node observation_processing)

install(
  TARGETS observation_processing strawberry_observation_node
  EXPORT export_strawberry_observation
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_export_targets(export_strawberry_observation HAS_LIBRARY_TARGET)
ament_export_dependencies(cv_bridge rclcpp sensor_msgs)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_observation_processor test/test_observation_processor.cpp)
  if(TARGET test_observation_processor)
    target_include_directories(test_observation_processor
      PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    ament_target_dependencies(test_observation_processor rclcpp sensor_msgs)
    target_link_libraries(test_observation_processor observation_processing ${OpenCV_LIBS})
  endif()

  ament_add_pytest_test(test_launch_contract test/test_launch_contract.py TIMEOUT 60)
endif()

ament_package()
