set(DIRECTX_WRAPPER_SOURCES
        directx_wrapper.cpp
)

add_library(directx_wrapper STATIC ${DIRECTX_WRAPPER_SOURCES})

target_include_directories(directx_wrapper
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(directx_wrapper
    PUBLIC
        $<$<BOOL:${D3D12_IS_SUPPORTED}>:D3D12_IS_SUPPORTED>)

target_link_libraries(directx_wrapper
    dxgi
    $<$<BOOL:${D3D12_IS_SUPPORTED}>:d3d12>)
