//
// Created by cartercpp on 11/6/26.
//

#include "neural_network.hpp"
#include <initializer_list>
#include <vector>
#include <mdspan>
#include <random>
#include <ranges>
#include <algorithm>
#include <utility>
#include <cstddef>
#include <cmath>

std::vector<float> neural_network::MatrixMatrix(Matrix lArg, Matrix rArg)
{
    const std::size_t rows = lArg.extent(0),
                      columns = rArg.extent(1),
                      iterCount = lArg.extent(1);

    std::vector<float> output(rows * columns, 0.0f);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            for (std::size_t i = 0; i < iterCount; ++i)
                std::mdspan(output.data(), rows, columns)[row, column] += lArg[row, i] * rArg[i, column];

    return output;
}

std::vector<float> neural_network::MatrixVector(Matrix mat, Vector vec)
{
    const std::size_t rows = mat.extent(0);

    std::vector<float> output(rows, 0.0f);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t i = 0; i < vec.extent(0); ++i)
            output[row] += mat[row, i] * vec[i];

    return output;
}

std::vector<float> neural_network::VectorMatrix(Vector vec, Matrix mat)
{
    const std::size_t columns = mat.extent(1);

    std::vector<float> output(columns, 0.0f);
    for (std::size_t column = 0; column < columns; ++column)
        for (std::size_t i = 0; i < vec.extent(0); ++i)
            output[column] += mat[i, column] * vec[i];

    return output;
}

std::vector<float> neural_network::OuterProduct(const std::vector<float>& lArg, const std::vector<float>& rArg)
{
    std::vector<float> output(lArg.size() * rArg.size(), 0.0f);

    for (std::size_t row = 0; row < lArg.size(); ++row)
        for (std::size_t column = 0; column < rArg.size(); ++column)
            std::mdspan(output.data(), lArg.size(), rArg.size())[row, column] = lArg[row] * rArg[column];

    return output;
}

neural_network::neural_network(std::initializer_list<std::size_t> layerSizes, float learningRate)
    : m_layerSizes{layerSizes},
    m_learningRate{learningRate}
{
    m_weights.reserve(m_layerSizes.size() - 1);
    m_weightGradients.reserve(m_layerSizes.size() - 1);
    m_biases.reserve(m_layerSizes.size() - 1);
    m_biasGradients.reserve(m_layerSizes.size() - 1);

    std::mt19937_64 rng{std::random_device{}()};
    for (std::size_t i = 0; i < m_layerSizes.size() - 1; ++i)
    {
        const std::size_t layerSize = m_layerSizes[i + 1],
                          prevLayerSize = m_layerSizes[i];

        std::normal_distribution<float> dist(0.0f, std::sqrtf(2.0f / static_cast<float>(prevLayerSize)));

        std::vector<float> weights(layerSize * prevLayerSize, 0.0f);
        for (std::size_t row = 0; row < layerSize; ++row)
            for (std::size_t column = 0; column < prevLayerSize; ++column)
                std::mdspan(weights.data(), layerSize, prevLayerSize)[row, column] = dist(rng);

        m_weights.emplace_back(std::move(weights));
        m_weightGradients.emplace_back(layerSize * prevLayerSize, 0.0f);
        m_biases.emplace_back(layerSize, 0.0f);
        m_biasGradients.emplace_back(layerSize, 0.0f);
    }
}

std::vector<std::vector<float>> neural_network::forward(const std::vector<float>& input) const
{
    std::vector<std::vector<float>> activations;
    activations.reserve(m_layerSizes.size());
    activations.push_back(input);

    for (std::size_t weightIndex = 0; weightIndex < m_weights.size(); ++weightIndex)
    {
        const std::size_t layerSize = m_layerSizes[weightIndex + 1],
                          prevLayerSize = m_layerSizes[weightIndex];

        std::vector<float> layerActivations = MatrixVector(
            Matrix{m_weights[weightIndex].data(), layerSize, prevLayerSize},
            Vector{activations[weightIndex].data(), prevLayerSize}
        );

        for (std::size_t neuron = 0; neuron < layerSize; ++neuron)
        {
            layerActivations[neuron] += m_biases[weightIndex][neuron];

            if (weightIndex + 1 < m_weights.size())
                layerActivations[neuron] = Relu(layerActivations[neuron]);
        }

        activations.emplace_back(std::move(layerActivations));
    }

    return activations;
}

std::vector<float> neural_network::predict(const std::vector<float>& input) const
{
    return forward(input).back();
}

std::vector<float> neural_network::fit(const std::vector<float>& input, const std::vector<float>& target)
{
    const std::vector<std::vector<float>> activations{forward(input)};
    const std::vector<float>& prediction{activations.back()};

    std::vector<float> gradient(m_layerSizes.back(), 0.0f);
    for (std::size_t neuron = 0; neuron < m_layerSizes.back(); ++neuron)
        gradient[neuron] = prediction[neuron] - target[neuron];

    return backward(activations, gradient);
}

std::vector<float> neural_network::backward(
    const std::vector<std::vector<float>>& activations,
    std::vector<float> gradient
)
{
    for (std::size_t weightIndex = m_weights.size(); weightIndex-- > 0;)
    {
        const std::size_t rightSize = m_layerSizes[weightIndex + 1],
                          leftSize = m_layerSizes[weightIndex];

        if (weightIndex < m_weights.size() - 1)
            for (std::size_t neuron = 0; neuron < rightSize; ++neuron)
                gradient[neuron] *= ReluDerivative(activations[weightIndex + 1][neuron]);

        const auto weightGradient = OuterProduct(gradient, activations[weightIndex]);
        for (std::size_t neuron = 0; neuron < rightSize * leftSize; ++neuron)
            m_weightGradients[weightIndex][neuron] += weightGradient[neuron];

        for (std::size_t neuron = 0; neuron < rightSize; ++neuron)
            m_biasGradients[weightIndex][neuron] += gradient[neuron];

        gradient = VectorMatrix(
            Vector{gradient.data(), rightSize},
            Matrix{m_weights[weightIndex].data(), rightSize, leftSize}
        );
    }

    return gradient;
}

void neural_network::update_weights_and_biases()
{
    for (std::size_t weightIndex = 0; weightIndex < m_weights.size(); ++weightIndex)
    {
        const std::size_t rows = m_layerSizes[weightIndex + 1],
                          columns = m_layerSizes[weightIndex];

        for (std::size_t neuron = 0; neuron < rows * columns; ++neuron)
            m_weights[weightIndex][neuron] -= m_learningRate * m_weightGradients[weightIndex][neuron];

        for (std::size_t neuron = 0; neuron < rows; ++neuron)
            m_biases[weightIndex][neuron] -= m_learningRate * m_biasGradients[weightIndex][neuron];
    }

    zero_gradients();
}

void neural_network::zero_gradients()
{
    for (std::size_t weightIndex = 0; weightIndex < m_weights.size(); ++weightIndex)
    {
        std::ranges::for_each(m_weightGradients[weightIndex], [](float& valueRef){valueRef = 0.0f;});
        std::ranges::for_each(m_biasGradients[weightIndex], [](float& valueRef){valueRef = 0.0f;});
    }
}

float neural_network::Relu(float x)
{
    return (x > 0.0f) ? x : 0.0f;
}

float neural_network::ReluDerivative(float x)
{
    return (x > 0.0f) ? 1.0f : 0.0f;
}

std::vector<std::vector<float>>& neural_network::weights() noexcept
{
    return m_weights;
}

const std::vector<std::vector<float>>& neural_network::weights() const noexcept
{
    return m_weights;
}

std::vector<std::vector<float>>& neural_network::biases() noexcept
{
    return m_biases;
}

const std::vector<std::vector<float>>& neural_network::biases() const noexcept
{
    return m_biases;
}