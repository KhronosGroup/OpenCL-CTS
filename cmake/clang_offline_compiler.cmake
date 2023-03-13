#[[
# Clang offline compiler wrapper

Usage: `cmake -D CLANG_EXE=<path_to_clang> -D CLANG_ARGS=<extra_args_to_clang> -P <path_to_this_script> -- <unparsed_args>`

- `CLANG_EXE` specifies the full path to a Clang compiler which may not be on the path
- `CLANG_ARGS` a semi-colon delimited list (CMake format) of args that should be passed to Clang
- unparsed arguments are given to the test executables verbatim
]]

macro(DEFINE_CLI_ARG_PARSING_FUNCTION CLI_ARG_NAME CLI_ARG_REGEX CLI_ARG_HELP)
  string(TOUPPER ${CLI_ARG_NAME} CLI_ARG_NAME_UPPER)
  function(GET_${CLI_ARG_NAME_UPPER} OUT_VAR CLI_ARGS)
  foreach(ARG IN LISTS CLI_ARGS)
    if(ARG MATCHES "${CLI_ARG_REGEX}")
      set(MODE ${CMAKE_MATCH_1})
    endif()
  endforeach()
  if(DEFINED MODE)
    set(${OUT_VAR} ${MODE} PARENT_SCOPE)
  else()
    message(FATAL_ERROR "${CLI_ARG_HELP}")
  endif()
  endfunction()
endmacro(DEFINE_CLI_ARG_PARSING_FUNCTION)

define_cli_arg_parsing_function(mode   [[^--mode=(spir-v|binary)]] "No compilation mode provided. Use --mode=<binary|spir-v>")
define_cli_arg_parsing_function(source [[^--source=(.+)]]          "No source file provided. Use --source=<path>")
define_cli_arg_parsing_function(output [[^--output=(.+)]]          "No output file provided. Use --output=<path>")

if(NOT DEFINED CLANG_EXE)
  message(FATAL_ERROR "No clang executable provided. Use -D CLANG_EXE=<path> (before -P)")
endif()

message(DEBUG "CMAKE_ARGC: ${CMAKE_ARGC}")

math(EXPR ARG_INDEX_STOP "${CMAKE_ARGC} - 1")
foreach(ARG_INDEX RANGE 1 ${ARG_INDEX_STOP})
  message(DEBUG "CMAKE_ARGV${ARG_INDEX}: ${CMAKE_ARGV${ARG_INDEX}}")
  list(APPEND CMAKE_ARGS "${CMAKE_ARGV${ARG_INDEX}}")
endforeach()

message(DEBUG "CMAKE_ARGS: ${CMAKE_ARGS}")

get_mode(MODE "${CMAKE_ARGS}")
message(VERBOSE "MODE: ${MODE}")

get_source(SOURCE "${CMAKE_ARGS}")
message(VERBOSE "SOURCE: ${SOURCE}")

get_output(OUTPUT "${CMAKE_ARGS}")
message(VERBOSE "OUTPUT: ${OUTPUT}")

message(VERBOSE "CLANG_EXE: ${CLANG_EXE}")

list(FIND CMAKE_ARGS "--" FIRST_DASHDASH)
math(EXPR AFTER_FIRST_DASHDASH "${FIRST_DASHDASH} + 1")
list(SUBLIST CMAKE_ARGS ${AFTER_FIRST_DASHDASH} "-1" CMAKE_ARGS_AFTER_FIRST_DASHDASH)
list(FIND CMAKE_ARGS_AFTER_FIRST_DASHDASH "--" SECOND_DASHDASH)
math(EXPR AFTER_SECOND_DASHDASH "${SECOND_DASHDASH} + 1")
list(SUBLIST CMAKE_ARGS_AFTER_FIRST_DASHDASH ${AFTER_SECOND_DASHDASH} "-1" CMAKE_ARGS_AFTER_SECOND_DASHDASH)

execute_process(
  COMMAND "${CLANG_EXE}"
    ${CMAKE_ARGS_AFTER_SECOND_DASHDASH}
    ${CLANG_ARGS}
    "${SOURCE}"
    -o "${OUTPUT}"
)