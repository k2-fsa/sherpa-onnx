set(srcs
  ./utils.cc
  ./offline-sense-voice-model-ascend.cc
)
add_executable(test-acl ./test-acl.cc ${srcs})
target_include_directories(test-acl PRIVATE ${ASCEND_TOOLKIT_HOME}/include)
target_link_libraries(test-acl
  -L${ASCEND_TOOLKIT_HOME}/lib64
  -lascendcl
)

