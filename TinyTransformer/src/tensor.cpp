#include "tensor.h"
#include <stdexcept>
#include <sstream>

Tensor::Tensor(std::vector<size_t> shape, size_t data_size, float fill): shape_(std::move(shape)){
    data_.resize(data_size, fill);
    compute_strides();
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<float> data): shape_(std::move(shape)), data_(std::move(data)){
    compute_strides();
}

void Tensor::compute_strides(){
    auto ndim = shape_.size();
    strides_.resize(ndim);
    if(0 == ndim){
        return;
    }

    strides_[ndim - 1] = 1;
    for(int i = ndim - 2; i >= 0; --i){
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

void Tensor::reshape(std::vector<size_t> new_shape){
    size_t new_size = 1;
    for(auto s : new_shape){
        new_size *= s;
    }
    if(new_size != data_.size()){
        throw std::runtime_error("Error: reshape size must match data size" + std::to_string(new_size) + " != " + std::to_string(data_.size()));
    }

    shape_ = std::move(new_shape);
    compute_strides();
}

std::string Tensor::shape_str() const{
    std::ostringstream oss;
    oss << "[";
    for(size_t i = 0; i < shape_.size(); ++i){
        if(i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "]";
    return oss.str();
}
