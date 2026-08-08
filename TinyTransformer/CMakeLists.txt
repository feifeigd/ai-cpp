
add_executable(TinyTransformer 
    src/main.cpp
    src/tokenizer.cpp
)

target_include_directories(TinyTransformer PRIVATE include)
