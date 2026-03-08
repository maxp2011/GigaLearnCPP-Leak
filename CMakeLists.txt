cmake_minimum_required (VERSION 3.8)

project("GigaLearnBot")

file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp")
list(FILTER FILES_SRC EXCLUDE REGEX ".*[\\\\/]Rewards[\\\\/].*")
add_executable(GigaLearnBot ${FILES_SRC})

# Set C++ version to 20
set_target_properties(GigaLearnBot PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnBot PROPERTIES CXX_STANDARD 20)

# Make sure GigaLearnCPP is going to build in the same directory as us
# Otherwise, we won't be able to import it at runtime
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# Include RLGymSim_PPO
add_subdirectory(GigaLearnCPP)
target_link_libraries(GigaLearnBot GigaLearnCPP)

# Include RLBot
add_subdirectory(RLBotCPP)
target_link_libraries(GigaLearnBot RLBotCPP)

# Copy python_scripts to exe dir so MetricSender/RenderSender can import them when run from build/Release/
add_custom_command(TARGET GigaLearnBot POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		${CMAKE_BINARY_DIR}/python_scripts
		$<TARGET_FILE_DIR:GigaLearnBot>/python_scripts
	COMMENT "Copying python_scripts for metrics/render receivers"
)

# RLBot inference executable (separate from training)
add_executable(GigaLearnRLBot
	rlbot_build/src/RLBotMain.cpp
	src/RLBotClient.cpp
	src/RLBotClient.h
)
set_target_properties(GigaLearnRLBot PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnRLBot PROPERTIES CXX_STANDARD 20)
target_link_libraries(GigaLearnRLBot GigaLearnCPP)
target_link_libraries(GigaLearnRLBot RLBotCPP)