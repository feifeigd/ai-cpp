#pragma once

#include <string>
#include <unordered_map>

class Tokenizer{
    
public:
    Tokenizer(std::string const& encoder_json_path, std::string const& vocab_bpe_path);

    std::vector<int> encode(std::string const& text)const;
    std::string decode(std::vector<int> const& ids)const;
    size_t  vocab_size()const{return id_to_token_.size();}
private:
    std::vector<std::string> pretokenize(std::string const& text)const;
    std::vector<std::string> bpe(std::string const& token)const;

    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    std::unordered_map<std::string, int> bpe_ranks_;

    std::string byte_encoder_[256];
    std::unordered_map<std::string, int> byte_decoder_;    
};
