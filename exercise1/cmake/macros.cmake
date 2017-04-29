
include(CMakeParseArguments)

macro(INSEG_SET_OUTPUT_DIR)
  set(INSEG_COMPILER_DIR)

  if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} "-dumpversion" OUTPUT_VARIABLE GCC_VERSION)
    string(REGEX MATCHALL "[0-9]+" GCC_VERSION_COMPONENTS ${GCC_VERSION})
    list(GET GCC_VERSION_COMPONENTS 0 GCC_MAJOR)
    list(GET GCC_VERSION_COMPONENTS 1 GCC_MINOR)

    set(INSEG_COMPILER_DIR "gcc${GCC_MAJOR}_${GCC_MINOR}")
  endif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)

  if(MSVC60) # Good luck!
    set(INSEG_COMPILER_DIR "vc6") # yes, this is correct
  elseif(MSVC70) # Good luck!
    set(INSEG_COMPILER_DIR "vc7") # yes, this is correct
  elseif(MSVC71)
    set(INSEG_COMPILER_DIR "vc71")
  elseif(MSVC80)
    set(INSEG_COMPILER_DIR "vc8")
  elseif(MSVC90)
    set(INSEG_COMPILER_DIR "vc9")
  elseif(MSVC10)
    if (CMAKE_CL_64)
		set(INSEG_COMPILER_DIR "vc10/x64")
	else (CMAKE_CL_64)
		set(INSEG_COMPILER_DIR "vc10/win32")
	endif(CMAKE_CL_64)
  elseif(MSVC)
    set(INSEG_COMPILER_DIR "vc") # ??
  endif(MSVC60)

  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/${INSEG_COMPILER_DIR})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/${INSEG_COMPILER_DIR})
  IF(WIN32)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/${INSEG_COMPILER_DIR})
  ELSE(WIN32)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/${INSEG_COMPILER_DIR})
  ENDIF(WIN32)
endmacro(INSEG_SET_OUTPUT_DIR)

macro(INSEG_CHECK_FOR_SSE)
    set(INSEG_SSE_FLAGS)

    if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} "-dumpversion" OUTPUT_VARIABLE GCC_VERSION)
        if(GCC_VERSION VERSION_GREATER 4.2)
          set(INSEG_SSE_FLAGS "${INSEG_SSE_FLAGS} -march=native")
          message(STATUS "Using CPU native flags for SSE optimization: ${INSEG_SSE_FLAGS}")
        endif()
    endif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)

    if(NOT INSEG_SSE_FLAGS)
      include(CheckCXXSourceRuns)
      set(CMAKE_REQUIRED_FLAGS)

      check_cxx_source_runs("
          #include <mm_malloc.h>
          int main()
          {
            void* mem = _mm_malloc (100, 16);
            return 0;
          }"
          HAVE_MM_MALLOC)

      check_cxx_source_runs("
          #include <stdlib.h>
          int main()
          {
            void* mem;
            return posix_memalign (&mem, 16, 100);
          }"
          HAVE_POSIX_MEMALIGN)

      if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
          set(CMAKE_REQUIRED_FLAGS "-msse4.1")
      endif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)

      check_cxx_source_runs("
          #include <smmintrin.h>
          int main()
          {
            __m128 a, b;
            float vals[4] = {1, 2, 3, 4};
            const int mask = 123;
            a = _mm_loadu_ps(vals);
            b = a;
            b = _mm_dp_ps (a, a, mask);
            _mm_storeu_ps(vals,b);
            return 0;
          }"
          HAVE_SSE4_1_EXTENSIONS)

      if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
          set(CMAKE_REQUIRED_FLAGS "-msse3")
      endif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)

      check_cxx_source_runs("
          #include <pmmintrin.h>
          int main()
          {
              __m128d a, b;
              double vals[2] = {0};
              a = _mm_loadu_pd(vals);
              b = _mm_hadd_pd(a,a);
              _mm_storeu_pd(vals, b);
              return 0;
          }"
          HAVE_SSE3_EXTENSIONS)

      if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
          set(CMAKE_REQUIRED_FLAGS "-msse2")
      elseif(MSVC AND NOT CMAKE_CL_64)
          set(CMAKE_REQUIRED_FLAGS "/arch:SSE2")
      endif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)

      check_cxx_source_runs("
          #include <emmintrin.h>
          int main()
          {
              __m128d a, b;
              double vals[2] = {0};
              a = _mm_loadu_pd(vals);
              b = _mm_add_pd(a,a);
              _mm_storeu_pd(vals,b);
              return 0;
          }"
          HAVE_SSE2_EXTENSIONS)

      if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
          set(CMAKE_REQUIRED_FLAGS "-msse")
      elseif(MSVC AND NOT CMAKE_CL_64)
          set(CMAKE_REQUIRED_FLAGS "/arch:SSE")
      endif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)

      check_cxx_source_runs("
          #include <xmmintrin.h>
          int main()
          {
              __m128 a, b;
              float vals[4] = {0};
              a = _mm_loadu_ps(vals);
              b = a;
              b = _mm_add_ps(a,b);
              _mm_storeu_ps(vals,b);
              return 0;
          }"
          HAVE_SSE_EXTENSIONS)

      set(CMAKE_REQUIRED_FLAGS)

      if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
          if (HAVE_SSE4_1_EXTENSIONS)
              SET(INSEG_SSE_FLAGS "${INSEG_SSE_FLAGS} -msse4.1 -mfpmath=sse")
              message(STATUS "Found SSE4.1 extensions, using flags: ${INSEG_SSE_FLAGS}")
          elseif(HAVE_SSE3_EXTENSIONS)
              SET(INSEG_SSE_FLAGS "${INSEG_SSE_FLAGS} -msse3 -mfpmath=sse")
              message(STATUS "Found SSE3 extensions, using flags: ${INSEG_SSE_FLAGS}")
          elseif(HAVE_SSE2_EXTENSIONS)
              SET(INSEG_SSE_FLAGS "${INSEG_SSE_FLAGS} -msse2 -mfpmath=sse")
              message(STATUS "Found SSE2 extensions, using flags: ${INSEG_SSE_FLAGS}")
          elseif(HAVE_SSE_EXTENSIONS)
              SET(INSEG_SSE_FLAGS "${INSEG_SSE_FLAGS} -msse -mfpmath=sse")
              message(STATUS "Found SSE extensions, using flags: ${INSEG_SSE_FLAGS}")
          else (HAVE_SSE4_1_EXTENSIONS)
              message(STATUS "No SSE extensions found")
          endif(HAVE_SSE4_1_EXTENSIONS)
      elseif (MSVC)
          if(HAVE_SSE2_EXTENSIONS)
              SET(INSEG_SSE_FLAGS "${INSEG_SSE_FLAGS} /arch:SSE2")
              message(STATUS "Found SSE2 extensions, using flags: ${INSEG_SSE_FLAGS}")
          elseif(HAVE_SSE_EXTENSIONS)
              SET(INSEG_SSE_FLAGS "${INSEG_SSE_FLAGS} /arch:SSE")
              message(STATUS "Found SSE extensions, using flags: ${INSEG_SSE_FLAGS}")
          endif(HAVE_SSE2_EXTENSIONS)
      endif ()

    endif()

    set(INSEG_CXX_FLAGS_RELEASE "${INSEG_CXX_FLAGS_RELEASE} ${INSEG_SSE_FLAGS}")
endmacro(INSEG_CHECK_FOR_SSE)


# Add a test target
# _name the test name
# _exename the executable name
# FILES _file_names the source files
# ARGUMENTS _cmd_line the exec args
# LINK_WITH _lib_names which libs must be linked
macro(INSEG_ADD_TEST _name _exename)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs FILES ARGUMENTS LINK_WITH)
  cmake_parse_arguments(INSEG_ADD_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  add_executable(${_exename} ${INSEG_ADD_TEST_FILES})
  if(NOT WIN32)
    set_target_properties(${_exename} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif(NOT WIN32)
  target_link_libraries(${_exename} ${Boost_SYSTEM_LIBRARY} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${INSEG_ADD_TEST_LINK_WITH})

  if(${CMAKE_VERSION} VERSION_LESS 2.8.4)
    add_test(${_name} ${_exename} ${INSEG_ADD_TEST_ARGUMENTS})
  else(${CMAKE_VERSION} VERSION_LESS 2.8.4)
    add_test(NAME ${_name} COMMAND ${_exename} ${INSEG_ADD_TEST_ARGUMENTS})
  endif(${CMAKE_VERSION} VERSION_LESS 2.8.4)
endmacro(INSEG_ADD_TEST _name _exename)


# Add a library target
# _name The library name
# FILES _filenames the source files
# LINK_WITH _lib_names which libs must be linked
# DEFINITIONS _definitions
macro(INSEG_ADD_LIBRARY _name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs FILES LINK_WITH DEFINITIONS)
  cmake_parse_arguments(INSEG_ADD_LIBRARY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  add_library(${_name} ${INSEG_LIB_TYPE} ${INSEG_ADD_LIBRARY_FILES})
  target_link_libraries(${_name} ${INSEG_ADD_LIBRARY_LINK_WITH})
  set_property(TARGET ${_name} APPEND PROPERTY COMPILE_DEFINITIONS ${INSEG_ADD_LIBRARY_DEFINITIONS})

  #if(USE_PROJECT_FOLDERS)
    #set_target_properties(${_name} PROPERTIES FOLDER "Libs")
  #endif(USE_PROJECT_FOLDERS)
endmacro(INSEG_ADD_LIBRARY)


# Add an executable target
# _name The executable name
# FILES _filenames the source files
# LINK_WITH _lib_names which libs must be linked
macro(INSEG_ADD_EXECUTABLE _name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs FILES LINK_WITH)
  cmake_parse_arguments(INSEG_ADD_EXECUTABLE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  add_executable(${_name} ${INSEG_ADD_EXECUTABLE_FILES})
  set_target_properties(${_name} PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}"
                                            RELEASE_POSTFIX "${CMAKE_RELEASE_POSTFIX}"
                                            RELWITHDEBINFO_POSTFIX "${CMAKE_RELWITHDEBINFO_POSTFIX}"
                                            MINSIZEREL_POSTFIX "${CMAKE_MINSIZEREL_POSTFIX}")
  target_link_libraries(${_name} ${INSEG_ADD_EXECUTABLE_LINK_WITH})

  #if(USE_PROJECT_FOLDERS)
    #set_target_properties(${_name} PROPERTIES FOLDER "Bin")
  #endif(USE_PROJECT_FOLDERS)
endmacro(INSEG_ADD_EXECUTABLE)

macro(INSEG_PROJECT_DEPS)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(INSEG_PROJECT_DEPS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

  foreach(dep ${INSEG_PROJECT_DEPS_UNPARSED_ARGUMENTS})
    include_directories(${PROJECT_SOURCE_DIR}/${dep}/include)
  endforeach(dep ${INSEG_PROJECT_DEPS_UNPARSED_ARGUMENTS})
endmacro(INSEG_PROJECT_DEPS)
