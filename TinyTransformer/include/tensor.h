#pragma once

#include <string>
#include <vector>

/// 张量
/// 严格定义： 张量是标量、向量、矩阵在更高维度的推广 ，本质是"多维数组"。在工程实现里（PyTorch、TensorFlow、这个 TinyTransformer 项目），无论几维都统称 tensor，底层就是一维连续内存 + shape + strides（你刚问的 compute_strides ）。
/// 数学里"张量"还涉及坐标变换规则（协变/逆变），但深度学习里基本就当"多维数组"用，不深究那套理论。
class Tensor{
public:
    Tensor() = default;
    Tensor(std::vector<size_t> shape, size_t data_size, float fill = 0.0f);
    Tensor(std::vector<size_t> shape, std::vector<float> data);

    std::vector<size_t> const& shape() const { return shape_; }
    std::vector<size_t> const& strides() const { return strides_; }
    size_t ndim() const { return shape_.size(); }
    /// 张量的元素个数，例如 {2, 3, 4} 有 2x3x4 个float元素
    size_t  size() const { return data_.size(); }
    size_t dim(size_t i) const { return shape_[i]; }

    /// 数据
    std::vector<float>& data() { return data_; }
    std::vector<float> const& data() const { return data_; }
    float* ptr() { return data_.data(); }
    float const* ptr() const { return data_.data(); }

    /// 访问数据
    float& operator[](size_t i) { return data_[i]; }
    float operator[](size_t i) const { return data_[i]; }
    float& at(size_t i) { return data_[i]; }
    float at(size_t i) const { return data_[i]; }
    float& at(size_t i, size_t j) { return data_[i * strides_[0] + j * strides_[1]]; }
    float at(size_t i, size_t j) const { return data_[i * strides_[0] + j * strides_[1]]; }
    float& at(size_t i, size_t j, size_t k) { return data_[i * strides_[0] + j * strides_[1] + k * strides_[2]]; }
    float at(size_t i, size_t j, size_t k) const { return data_[i * strides_[0] + j * strides_[1] + k * strides_[2]]; }
    float& at(size_t i, size_t j, size_t k, size_t l) { return data_[i * strides_[0] + j * strides_[1] + k * strides_[2] + l * strides_[3]]; }
    float at(size_t i, size_t j, size_t k, size_t l) const { return data_[i * strides_[0] + j * strides_[1] + k * strides_[2] + l * strides_[3]]; }

    void reshape(std::vector<size_t> new_shape);
    std::string shape_str() const;

private:
    /// 计算张量每个维度的 步幅（stride） 
    /// ——即沿某个维度移动一步，在底层一维数组里要跳过多少个元素。
    /// 这是多维数组用一维存储时的索引映射表。
    void compute_strides();

    /// 张量的形状，例如 {2, 3, 4} 表示 2x3x4 的张量
    std::vector<size_t> shape_;
    /// 底层数据存储，是一维数组
    std::vector<float> data_;
    std::vector<size_t> strides_;
};
