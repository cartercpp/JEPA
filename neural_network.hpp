//
// Created by cartercpp on 11/6/26.
//

#pragma once

#include <initializer_list>
#include <vector>
#include <mdspan>
#include <cstddef>

class neural_network
{
public:

    // CONSTRUCTORS

    explicit neural_network(
        std::initializer_list<std::size_t> layerSizes,
        float learningRate
    );
        //pre(layerSizes.size() >= 3)
        //pre((learningRate > 0) && (learningRate < 1));

    // METHODS

    [[nodiscard]] std::vector<std::vector<float>> forward(const std::vector<float>& input) const;
        //pre(input.size() == m_layerSizes.front());

    [[nodiscard]] std::vector<float> predict(const std::vector<float>& input) const;
        //pre(input.size() == m_layerSizes.front());

    std::vector<float> backward(
        const std::vector<std::vector<float>>& activations,
        std::vector<float> gradient
    );
        //pre(activations.size() == m_layerSizes.size())
        //pre(gradient.size() == m_layerSizes.back());

    std::vector<float> fit(const std::vector<float>& input, const std::vector<float>& target);
        //pre(input.size() == m_layerSizes.front())
        //pre(target.size() == m_layerSizes.back())

    void update_weights_and_biases();
    void zero_gradients();

    [[nodiscard]] std::vector<std::vector<float>>& weights() noexcept;
    [[nodiscard]] const std::vector<std::vector<float>>& weights() const noexcept;
    [[nodiscard]] std::vector<std::vector<float>>& biases() noexcept;
    [[nodiscard]] const std::vector<std::vector<float>>& biases() const noexcept;

private:

    using Matrix = std::mdspan<const float, std::dextents<std::size_t, 2>>;
    using Vector = std::mdspan<const float, std::dextents<std::size_t, 1>>;

    static float Relu(float x);
    static float ReluDerivative(float x);

    [[nodiscard]] static std::vector<float> MatrixMatrix(Matrix lArg, Matrix rArg);
        //pre(lArg.extent(1) == rArg.extent(0));

    [[nodiscard]] static std::vector<float> MatrixVector(Matrix mat, Vector vec);
        //pre(mat.extent(1) == vec.extent(0));

    [[nodiscard]] static std::vector<float> VectorMatrix(Vector vec, Matrix mat);
        //pre(vec.extent(0) == mat.extent(0));

    [[nodiscard]] static std::vector<float> OuterProduct(const std::vector<float>& lArg, const std::vector<float>& rArg);

    std::vector<std::vector<float>> m_weights,
                                    m_weightGradients,
                                    m_biases,
                                    m_biasGradients;
    std::vector<std::size_t> m_layerSizes;
    float m_learningRate;
};