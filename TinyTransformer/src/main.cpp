
#include "tokenizer.h"

#include <print>

#include <string>

// 匿名 namespace（anonymous namespace）在 C++ 里的作用是： 
// 把它里面的所有声明（类型、变量、函数）限制在当前翻译单元（当前 .cpp 文件）内部可见，对外部文件不可见、不可链接 。
namespace {
    struct Args {
        std::string cmd;
        std::string prompt;
        int max_tokens = 50;
        float temp = 0.8f;
        int top_k = 40;
        uint64_t seed = 42;
        bool greedy = false;
        bool debug = false;
        std::string weights_path = "weights";
    };

    char const* opt_value(int argc, char const* argv[], int& i) {
        if (i + 1 >= argc) {
            std::println(stderr, "Error: missing value for option {}", argv[i]);
            return nullptr;
        }
        return argv[++i];
    }

    void usage() {
        std::println(R"(
Usage: 
    TinyTransformer generate "<prompt>" 
        [--max-tokens N] [--temp T] [--top-k K]
        [--seed N] [--greedy] [--debug]
        [--weight-path DIR]

    TinyTransformer logits "<prompt>" [--debug] [--weight-path DIR]
)");
    }
}


int main(int argc, char const* argv[]) {
    
    if(argc < 3) {
        usage();
        return 1;
    }

    Args a{.cmd = argv[1],};
    for(int i = 2; i < argc; ++i) {
        std::string arg{argv[i]};
        if(arg == "--max-tokens") {
            a.max_tokens = std::stoi(opt_value(argc, argv, i));
        } else if(arg == "--temp") {
            a.temp = std::stof(opt_value(argc, argv, i));
        } else if(arg == "--top-k") {
            a.top_k = std::stoi(opt_value(argc, argv, i));
        } else if(arg == "--seed") {
            a.seed = std::stoull(opt_value(argc, argv, i));
        } else if(arg == "--greedy") {
            a.greedy = true;
        } else if(arg == "--debug") {
            a.debug = true;
        } else if(arg == "--weight-path") {
            a.weights_path = opt_value(argc, argv, i);
        } else if(arg[0] != '-') {
            a.prompt = arg;
        }else{
            std::println(stderr, "Error: unknown option {}", arg);
            return 2;
        }
    }

    // safetensors 是 Hugging Face 开发的一种 模型权重文件格式 ，用于安全、快速地保存和加载深度学习模型的张量（权重）数据。
    std::string const st_path{a.weights_path + "/model.safetensors"};
    std::string const enc_path{a.weights_path + "/encoder.json"};
    std::string const bpe_path{a.weights_path + "/vocab.json"};

    std::println(stderr, "loading tokenizer...");
    Tokenizer tokenizer{enc_path, bpe_path};

    return 0;
}
