#include "tokenizer.h"
#include <fstream>
#include <sstream>

namespace {
    std::string read_file(std::string const& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error("Failed to open file: " + path);
        }

        std::stringstream buffer;
        buffer << f.rdbuf();
        return buffer.str();
    }
}

Tokenizer::Tokenizer(std::string const& encoder_json_path, std::string const& vocab_bpe_path)
{
    // Load encoder.json
    auto encoder_json{read_file(encoder_json_path)};
}
