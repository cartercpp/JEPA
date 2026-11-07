//
// Created by cartercpp on 11/6/26.
//

#include "jepa.hpp"
#include <initializer_list>
#include <cstddef>
#include <cmath>
#include "neural_network.hpp"

jepa::jepa(
    std::initializer_list<std::size_t> encoderLayerSizes,
    std::initializer_list<std::size_t> predictorLayerSizes,
    std::size_t contextFrames,
    float learningRate,
    float emaDecay,
    float varianceWeight,
    float minStdDev,
    bool frozenTargetEncoder
) : m_contextEncoder{encoderLayerSizes, learningRate},
    m_targetEncoder{m_contextEncoder},
    m_predictor{predictorLayerSizes, learningRate},
    m_contextFrames{contextFrames},
    m_encodingSize{*(encoderLayerSizes.end() - 1)},
    m_emaDecay{emaDecay},
    m_varianceWeight{varianceWeight},
    m_minStdDev{minStdDev},
    m_frozenTargetEncoder{frozenTargetEncoder}
{}

float jepa::train(const std::vector<std::vector<float>>& frames)
{
    float averageLoss = 0.0f;

    for (std::size_t start = 0; start + m_contextFrames < frames.size(); ++start)
    {
        std::vector<
                std::vector<std::vector<float>>
            > frameEncodingActivations;
        frameEncodingActivations.reserve(m_contextFrames);

        std::vector<float> contextEncoding;
        contextEncoding.reserve(m_contextFrames * m_encodingSize);

        for (std::size_t frameIndex = start; frameIndex < start + m_contextFrames; ++frameIndex)
        {
            const std::vector<float>& frame{frames[frameIndex]};

            frameEncodingActivations.emplace_back(m_contextEncoder.forward(frame));

            contextEncoding.insert(
                contextEncoding.end(),
                frameEncodingActivations.back().back().begin(),
                frameEncodingActivations.back().back().end()
            );
        }

        const std::vector<std::vector<float>> predictionActivations{m_predictor.forward(contextEncoding)};
        const std::vector<float>& predictedNextFrameEncoding{predictionActivations.back()};

        const std::vector<float>& nextFrame{frames[start + m_contextFrames]};
        const std::vector<float> nextFrameEncoding{m_targetEncoder.predict(nextFrame)};

        std::vector<float> incomingGradient(m_encodingSize, 0.0f);
        for (std::size_t i = 0; i < m_encodingSize; ++i)
            incomingGradient[i] = predictedNextFrameEncoding[i] - nextFrameEncoding[i];

        const float loss = [&incomingGradient](){
            float loss = 0.0f;

            for (const float gradient : incomingGradient)
                loss += gradient * gradient / 2.0f;

            return loss;
        }();
        averageLoss += loss / static_cast<float>(frames.size() - m_contextFrames);

        std::vector<float> varianceGradient(m_contextFrames * m_encodingSize, 0.0f);
        float varianceLoss = 0.0f;

        if (m_contextFrames > 1)
        {
            for (std::size_t encodingIndex = 0; encodingIndex < m_encodingSize; ++encodingIndex)
            {
                float meanEncodingValue = 0.0f;
                for (std::size_t i = 0; i < m_contextFrames; ++i)
                    meanEncodingValue += contextEncoding[i * m_encodingSize + encodingIndex]
                                         / static_cast<float>(m_contextFrames);

                float variance = 0.0f;
                for (std::size_t i = 0; i < m_contextFrames; ++i)
                    variance += (contextEncoding[i * m_encodingSize + encodingIndex] - meanEncodingValue)
                                * (contextEncoding[i * m_encodingSize + encodingIndex] - meanEncodingValue)
                                / static_cast<float>(m_contextFrames);

                const float stdDev = std::sqrtf(variance + 1e-4f);

                if (stdDev < m_minStdDev)
                {
                    const float shortfall = m_minStdDev - stdDev;

                    varianceLoss += m_varianceWeight * shortfall * shortfall
                                    / (2.0f * static_cast<float>(m_encodingSize));

                    for (std::size_t i = 0; i < m_contextFrames; ++i)
                        varianceGradient[i * m_encodingSize + encodingIndex]
                            = -m_varianceWeight * shortfall
                            * (contextEncoding[i * m_encodingSize + encodingIndex] - meanEncodingValue)
                            / (static_cast<float>(m_contextFrames * m_encodingSize) * stdDev);
                }
            }
        }
        averageLoss += varianceLoss / static_cast<float>(frames.size() - m_contextFrames);

        const std::vector<float> contextGradient{m_predictor.backward(predictionActivations, incomingGradient)};
        for (std::size_t i = 0; i < m_contextFrames; ++i)
        {
            std::vector<float> frameGradient(
                contextGradient.begin() + i * m_encodingSize,
                contextGradient.begin() + (i + 1) * m_encodingSize
            );

            for (std::size_t encodingIndex = 0; encodingIndex < m_encodingSize; ++encodingIndex)
                frameGradient[encodingIndex] += varianceGradient[i * m_encodingSize + encodingIndex];

            m_contextEncoder.backward(
                frameEncodingActivations[i],
                frameGradient
            );
        }

        m_predictor.update_weights_and_biases();
        m_contextEncoder.update_weights_and_biases();

        if (!m_frozenTargetEncoder)
            for (std::size_t weightIndex = 0; weightIndex < m_targetEncoder.weights().size(); ++weightIndex)
            {
                std::vector<float>& targetWeightsRef{m_targetEncoder.weights()[weightIndex]};
                const std::vector<float>& contextWeights{m_contextEncoder.weights()[weightIndex]};

                for (std::size_t i = 0; i < targetWeightsRef.size(); ++i)
                    targetWeightsRef[i] = m_emaDecay * targetWeightsRef[i] + (1.0f - m_emaDecay) * contextWeights[i];

                std::vector<float>& targetBiasesRef{m_targetEncoder.biases()[weightIndex]};
                const std::vector<float>& contextBiases{m_contextEncoder.biases()[weightIndex]};
                for (std::size_t i = 0; i < targetBiasesRef.size(); ++i)
                    targetBiasesRef[i] = m_emaDecay * targetBiasesRef[i] + (1.0f - m_emaDecay) * contextBiases[i];
            }
    }

    return averageLoss;
}

std::vector<float> jepa::encode(const std::vector<float>& frame, bool useTargetEncoder) const
{
    return useTargetEncoder ? m_targetEncoder.predict(frame) : m_contextEncoder.predict(frame);
}

std::vector<float> jepa::predict(const std::vector<float>& encoding) const
{
    return m_predictor.predict(encoding);
}