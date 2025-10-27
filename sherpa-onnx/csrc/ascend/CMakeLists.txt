add_executable(test-acl ./acl-test.cc ./utils.cc)
target_include_directories(test-acl PRIVATE ${ASCEND_TOOLKIT_HOME}/include)
target_link_libraries(test-acl
  -L${ASCEND_TOOLKIT_HOME}/lib64
  -lascendcl
)

