#include "tokenizer.h"
#include <fstream>
#include <regex>
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

    // Simple JSON parsing for {"token": id, ...} format
    size_t pos = 0;
    auto file_size = encoder_json.size();
    while ((pos = encoder_json.find('"', pos)) != std::string::npos) {
        auto end = encoder_json.find('"', pos + 1);
        if(std::string::npos == end) {
            break;
        }
        std::string token{encoder_json.substr(pos + 1, end - pos - 1)};
        pos = end + 1;

        while(pos < file_size && (encoder_json[pos] == ' ' || encoder_json[pos] == ':')) {
            pos++;
        }
        auto num_start = pos;
        while(pos < file_size && (encoder_json[pos] >= '0' && encoder_json[pos] <= '9')) {
            pos++;
        }
        if(num_start < pos){
            auto id = std::stoi(encoder_json.substr(num_start, pos - num_start));
            token_to_id_[token] = id;
            id_to_token_[id] = token;
        }
        while(pos < file_size && encoder_json[pos] != '"') {
            pos++;
        }
    }

    // Build byte encoder/decoder
    for(int i = 0; i < 256; ++i) {
        auto c = static_cast<char>(i);
        // 33 ~ 126, 161 ~ 172, 174 ~ 255 are printable
        if((c >= '!' && c <= '~') || (c >= 161 && c <= 172) || (c >= 174 && c <= 255)){
            byte_encoder_[i] = std::string(1, c);// 直接用这个字节本身作为字符
        }else{
            // 不可见/控制字符，映射到 128~255 范围的"占位"字节
            byte_encoder_[i] = std::string(1, static_cast<char>(i + 256));// 用 256 开始的字符作为非打印字符
        }
    }

    for (int i = 0; i < 256; ++i) {
        byte_decoder_[byte_encoder_[i]] = i;
    }

    // Load vocab.bpe
    auto bpe_content{read_file(vocab_bpe_path)};
    std::istringstream iss(bpe_content);
    std::string line;

    // Skip header line
    std::getline(iss, line);

    int rank = 0;
    while(std::getline(iss, line)) {
        if(line.empty()) {
            continue;
        }

        auto space_pos = line.find(' ');
        if(space_pos == std::string::npos) {
            continue;
        }
        auto first = line.substr(0, space_pos);
        auto second = line.substr(space_pos + 1);
        bpe_ranks_[first + second] = rank++;
    }
}

std::vector<std::string> Tokenizer::pretokenize(std::string const& text)const
{
    // GPT-2 protokenization regex
    std::regex re{"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}]+|\\s+(?!\\S)|\\s+"};

    std::vector<std::string> tokens;
    auto begin = std::sregex_iterator(text.begin(), text.end(), re);
    auto end = std::sregex_iterator();

    for(auto it = begin; it != end; ++it) {
        auto match = it->str();
        if(!match.empty()){
            tokens.push_back(match);
        }
    }

    return tokens;
}

std::vector<std::string> Tokenizer::bpe(std::string const& token)const
{
    std::vector<std::string> word;
    for(auto c : token) {
        word.push_back(std::string(1, c));
    }

    do{
        if(word.empty())break;
        while(true){
            std::string best_pair;
            int min_rank = std::numeric_limits<int>::max();
            for(int i = 0; i + 1 < word.size(); ++i) {
                auto pair = word[i] + word[i + 1];
                if(auto it = bpe_ranks_.find(pair); it != bpe_ranks_.end()){
                    auto rank = it->second;
                    if(rank < min_rank){
                        min_rank = rank;
                        best_pair = pair;
                    }
                }
            }
            if(best_pair.empty()){
                break;
            }
            
            std::vector<std::string> new_word;
            for(int i = 0; i < word.size();) {
                if(i + 1 < word.size() && word[i] + word[i + 1] == best_pair){
                    new_word.push_back(best_pair);
                    i += 2;
                }else{
                    new_word.push_back(word[i]);
                    i++;
                }
            }
            word = std::move(new_word);
            if(word.size() == 1){
                break;
            }
        }

    }while( false);

    return word;
}

std::vector<int> Tokenizer::encode(std::string const& text)const
{
    std::vector<int> result;

    // Convert to bytes
    std::string btext;
    for(auto c : text) {
        btext += byte_encoder_[c];
    }

    auto pretokens = pretokenize(btext);

    for(auto const& token : pretokens) {
        auto bpe_tokens = bpe(token);
        for(auto const& bpe_token : bpe_tokens) {
            if(auto it = token_to_id_.find(bpe_token); it != token_to_id_.end()){
                result.push_back(it->second);
            }
        }
    }

    return result;
}

std::string Tokenizer::decode(std::vector<int> const& ids)const
{
    std::string result;
    for(auto id : ids) {
        if(auto it = id_to_token_.find(id); it != id_to_token_.end()){
           auto const& token = it->second;
           for(auto c : token) {
               auto uc = static_cast<unsigned char>(c);
               if(uc >= 256)uc -= 256;
               result += static_cast<char>(uc);
           }
        }
    }
    
    return result;
}