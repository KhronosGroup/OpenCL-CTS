set(MODULE_NAME CL_KHR_SEMAPHORE)

set(${MODULE_NAME}_SOURCES
         main.cpp
         test_semaphores.cpp
         test_semaphores_negative_wait_signal.cpp
         test_semaphores_negative_release_retain.cpp
         test_semaphores_negative_getinfo.cpp
         test_semaphores_negative_create.cpp
         test_semaphores_cross_queue.cpp
         test_semaphores_queries.cpp
         test_semaphores_payload.cpp
         semaphore_base.h
)

include(../../CMakeCommon.txt)
