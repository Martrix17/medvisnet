/**
 * @file inferencer.h
 * @brief ONNX Runtime inference for Vision Transformer models with CPU execution.
 *
 * Example:
 * @code
 * ONNXInferencer inferencer("model.onnx", 224, 224);
 * InferenceResult result = inferencer.predict("image.png");
 * std::cout << "Predicted class index: " << result.index << std::endl;
 * @endcode
 */

#ifndef INFERENCER_H
#define INFERENCER_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

/**
 * @brief Inference output struct containing class predictions and probabilities.
 */
struct InferenceResult
{
    std::vector<float> probs; ///< Probability distribution over classes
    int index;                ///< Predicted class index
};

/**
 * @brief ONNX Runtime inferencer for Vision Transformer models (CPU only).
 *
 * Loads ONNX models, preprocesses images with normalization, and runs inference
 * using ONNX Runtime with CPU execution provider.
 */
class ONNXInferencer
{
public:
    /**
     * @brief Construct inferencer with ONNX model and preprocessing parameters.
     *
     * @param onnx_path Path to ONNX model file.
     * @param image_height Target image height for resizing.
     * @param image_width Target image width for resizing.
     * @param mean RGB channel means for normalization (ImageNet defaults).
     * @param std RGB channel stds for normalization (ImageNet defaults).
     *
     * @throws Ort::Exception if model loading fails.
     */
    ONNXInferencer(
        const std::string &onnx_path,
        int image_height = 224,
        int image_width = 224,
        const cv::Scalar &mean = cv::Scalar(0.485, 0.456, 0.406),
        const cv::Scalar &std = cv::Scalar(0.229, 0.224, 0.225));

    /**
     * @brief Run inference on a single image.
     *
     * @param image_path Path to input image file.
     * @return InferenceResult containing probabilities and predicted class.
     *
     * @throws cv::Exception if image cannot be loaded.
     * @throws Ort::Exception if inference fails.
     */
    InferenceResult predict(const std::string &image_path);

    /**
     * @brief Get model input shape.
     *
     * @return Input tensor shape [batch_size, channels, height, width].
     */
    std::vector<int64_t> get_input_shape() const { return input_shape_; }

private:
    /**
     * @brief Preprocess image: resize, normalize, and convert to CHW format.
     *
     * @param image Input image in BGR format.
     * @return Preprocessed image ready for inference [C, H, W].
     */
    cv::Mat preprocess(const cv::Mat &image);

    /**
     * @brief Convert logits to probability distribution.
     *
     * @param logits Raw model outputs.
     * @return Softmax probabilities summing to 1.0.
     */

    std::vector<float> softmax(const std::vector<float> &logits);

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions session_options_;

    std::vector<std::string> input_names_storage_;
    std::vector<std::string> output_names_storage_;
    std::vector<const char *> input_names_;
    std::vector<const char *> output_names_;
    std::vector<int64_t> input_shape_;

    int image_height_;
    int image_width_;
    cv::Scalar mean_;
    cv::Scalar std_;
};

#endif // INFERENCER_H
