/**
 * @file inferencer.cpp
 * @brief Implementation of ONNX Runtime inferencer for Vision Transformers.
 */

#include "inferencer.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

ONNXInferencer::ONNXInferencer(
    const std::string &onnx_path,
    int image_height,
    int image_width,
    const cv::Scalar &mean,
    const cv::Scalar &std)
    : image_height_(image_height),
      image_width_(image_width),
      mean_(mean),
      std_(std)
{
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXInferencer");

    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options_.DisableMemPattern();
    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    session_ = std::make_unique<Ort::Session>(*env_, onnx_path.c_str(), session_options_);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_input_nodes = session_->GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++)
    {
        auto input_name = session_->GetInputNameAllocated(i, allocator);
        input_names_storage_.push_back(std::string(input_name.get()));
        input_names_.push_back(input_names_storage_.back().c_str());
    }

    size_t num_output_nodes = session_->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++)
    {
        auto output_name = session_->GetOutputNameAllocated(i, allocator);
        output_names_storage_.push_back(std::string(output_name.get()));
        output_names_.push_back(output_names_storage_.back().c_str());
    }

    Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape_ = tensor_info.GetShape();
    if (!input_shape_.empty() && input_shape_[0] == -1)
        input_shape_[0] = 1;

    std::cout << "✅ Model input shape: [";
    for (size_t i = 0; i < input_shape_.size(); ++i)
    {
        std::cout << input_shape_[i];
        if (i + 1 < input_shape_.size())
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

cv::Mat ONNXInferencer::preprocess(const cv::Mat &image)
{
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(image_width_, image_height_));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
    resized = (resized - mean_) / std_;

    return resized;
}

std::vector<float> ONNXInferencer::softmax(const std::vector<float> &logits)
{
    std::vector<float> probs(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;

    for (size_t i = 0; i < logits.size(); i++)
    {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (float &p : probs)
        p /= sum;

    return probs;
}

InferenceResult ONNXInferencer::predict(const std::string &image_path)
{
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty())
        throw std::runtime_error("❌ Failed to load image: " + image_path);

    cv::Mat processed = preprocess(image);

    // Convert HWC (OpenCV) to CHW (ONNX) format
    std::vector<float> input_tensor_values;
    int channels = processed.channels();
    int h = processed.rows;
    int w = processed.cols;
    input_tensor_values.resize(channels * h * w);

    for (int c = 0; c < channels; ++c)
    {
        for (int i = 0; i < h; ++i)
        {
            for (int j = 0; j < w; ++j)
            {
                input_tensor_values[c * h * w + i * w + j] =
                    processed.at<cv::Vec3f>(i, j)[c];
            }
        }
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape_.data(),
        static_cast<uint32_t>(input_shape_.size()));

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        &input_tensor,
        input_names_.size(),
        output_names_.data(),
        output_names_.size());

    float *logits_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t num_classes = static_cast<size_t>(output_shape.back());
    std::vector<float> logits(logits_data, logits_data + num_classes);

    std::vector<float> probs = softmax(logits);
    int index = static_cast<int>(std::distance(probs.begin(), std::max_element(probs.begin(), probs.end())));

    return InferenceResult{probs, index};
}
