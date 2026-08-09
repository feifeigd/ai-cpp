#pragma once

#include "gpt2.h"

// 加载 model.safetensors
GPT2Weights load_gpt2_weights(std::string const& st_path);
