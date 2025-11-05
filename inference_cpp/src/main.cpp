
/**
 * @file main.cpp
 * @brief CLI application for Vision Transformer inference using ONNX Runtime.
 *
 * Usage:
 *   ./inference <model.onnx> <image_path>
 *
 * Example:
 *   ./inference models/exported/vit_base.onnx data/test_image.png
 */

#include "inferencer.h"
#include <iostream>
#include <iomanip>
#include <vector>

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <image_path>\n"
                  << "Example:\n  " << argv[0]
                  << " models/vit_base.onnx data/test_image.jpg\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    std::vector<std::string> classes = {"COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"};

    try
    {
        std::cout << "=== ONNX Vision Transformer Inference ===\n";
        ONNXInferencer inferencer(model_path);

        auto result = inferencer.predict(image_path);

        std::cout << "\n=== Results ===\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Predicted class index: " << result.index << "\n";
        std::cout << "Predicted class: " << classes.at(result.index) << "\n";
        std::cout << "Probabilities:\n";
        for (size_t i = 0; i < result.probs.size(); ++i)
            std::cout << "  " << classes[i] << ": " << result.probs[i] << "\n";
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "ONNX Runtime Error: " << e.what() << "\n";
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
