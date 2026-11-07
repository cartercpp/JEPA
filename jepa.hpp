//
// Created by cartercpp on 11/6/26.
//

#pragma once

#include <initializer_list>
#include <cstddef>
#include "neural_network.hpp"

class jepa
{
public:

    // CONSTRUCTORS

    explicit jepa(
        std::initializer_list<std::size_t> encoderLayerSizes,
        std::initializer_list<std::size_t> predictorLayerSizes,
        std::size_t contextFrames,
        float learningRate,
        float emaDecay,
        float varianceWeight,
        float minStdDev,
        bool frozenTargetEncoder
    );
    /*
        pre(*predictorLayerSizes.begin() == *(encoderLayerSizes.end() - 1) * contextFrames)
        pre(*(predictorLayerSizes.end() - 1) == *(encoderLayerSizes.end() - 1))
        pre(contextFrames > 0)
        pre((learningRate > 0.0f) && (learningRate < 1.0f))
        pre((emaDecay >= 0.0f) && (emaDecay < 1.0f))
        pre(varianceWeight >= 0.0f)
        pre(encoderLayerSizes.size() >= 3)
        pre(predictorLayerSizes.size() >= 3)
    */

    // METHODS

    float train(const std::vector<std::vector<float>>& frames);
        //pre(frames.size() > m_contextFrames);

    std::vector<float> encode(const std::vector<float>& frame, bool useTargetEncoder) const;
    std::vector<float> predict(const std::vector<float>& encoding) const;
        //pre(encoding.size() == m_encodingSize * m_contextFrames);

private:

    neural_network m_contextEncoder,
                   m_targetEncoder,
                   m_predictor;
    float m_emaDecay,
          m_varianceWeight,
          m_minStdDev;
    std::size_t m_contextFrames,
                m_encodingSize;
    bool m_frozenTargetEncoder;
};