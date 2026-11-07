#include <iostream>
#include <format>
#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <limits>
#include <thread>
#include <stop_token>
#include <cstddef>
#include "jepa.hpp"

int main()
{
    // change these numbers if necessary:
    constexpr std::size_t rows = 16,
                          columns = 30,
                          encodingSize = 32,
                          contextFrames = 5;

    constexpr float minX = 0.0f,
                    maxX = 10.0f,
                    minY = 0.0f,
                    maxY = 10.0f;

    // essentials:
    std::mt19937_64 rng{std::random_device{}()};
    std::uniform_real_distribution<double> xDist(minX, maxX),
                                           yDist(minY, maxY),
                                           xVelocityDist(0.0f, (maxX - minX) / 10.0f),
                                           yVelocityDist(0.0f, (maxY - minY) / 10.0f);
    float x = xDist(rng),
          y = yDist(rng),
          xVelocity = xVelocityDist(rng),
          yVelocity = yVelocityDist(rng);

    jepa worldModel(
        {2, 128, 64, 32},
        {encodingSize * contextFrames, 128, 64, 32},
        contextFrames,
        0.001f,
        0.99f,
        0.05f,
        0.1f,
        false
    );

    auto displayFrame = []() {
        std::array<std::array<char, columns>, rows> displayFrame;

        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t column = 0; column < columns; ++column)
                displayFrame[row][column] = ' ';

        return displayFrame;
    }();

    // training:
    std::vector<std::vector<float>> frames(100, std::vector<float>(2, 0.0f));
    std::vector<std::size_t> indices(100, 0);

    std::cout << "\033[2J";
    std::cout << "\033[?25l";
    std::jthread trainingThread{[&](std::stop_token st) {
        std::size_t trainingEpoch = 0;

        while (!st.stop_requested())
        {
            // randomize position and velocity:
            x = xDist(rng);
            y = yDist(rng);

            xVelocity = xVelocityDist(rng);
            yVelocity = yVelocityDist(rng);

            if (std::bernoulli_distribution{0.5}(rng))
                xVelocity = -xVelocity;

            if (std::bernoulli_distribution{0.5}(rng))
                yVelocity = -yVelocity;

            // create frames:
            for (std::size_t frameIndex = 0; frameIndex < frames.size(); ++frameIndex)
            {
                if ((x + xVelocity < minX) || (x + xVelocity >= maxX))
                    xVelocity = -xVelocity;

                if ((y + yVelocity < minY) || (y + yVelocity >= maxY))
                    yVelocity = -yVelocity;

                x += xVelocity;
                y += yVelocity;

                const std::size_t row = rows - 1 - static_cast<std::size_t>((y - minY) / (maxY - minY) * (rows - 1)),
                                  column = static_cast<std::size_t>((x - minX) / (maxX - minX) * (columns - 1));

                indices[frameIndex] = row * columns + column;
                frames[frameIndex][0] = static_cast<float>(column) / (columns - 1);
                frames[frameIndex][1] = static_cast<float>(row) / (rows - 1);
            }

            // visualize predictions:
            if (trainingEpoch % 20 == 0)
            {
                std::vector<std::vector<float>> positionEncodings;
                positionEncodings.reserve(rows * columns);

                for (std::size_t position = 0; position < rows * columns; ++position)
                {
                    const std::size_t row = position / columns,
                                      column = position % columns;

                    const std::vector<float> frame{
                        static_cast<float>(column) / (columns - 1),
                        static_cast<float>(row) / (rows - 1)
                    };

                    positionEncodings.emplace_back(worldModel.encode(frame, true));
                }

                for (std::size_t start = 0; start + contextFrames < frames.size(); ++start)
                {
                    std::vector<float> contextEncoding;
                    contextEncoding.reserve(contextFrames * encodingSize);

                    for (std::size_t frameIndex = start; frameIndex < start + contextFrames; ++frameIndex)
                    {
                        const std::vector<float> &frame{frames[frameIndex]};
                        const std::vector<float> frameEncoding{worldModel.encode(frame, false)};
                        contextEncoding.insert(contextEncoding.end(), frameEncoding.begin(), frameEncoding.end());
                    }

                    const std::vector<float> predictedNextFrameEncoding{worldModel.predict(contextEncoding)};

                    std::size_t bestPosition = 0;
                    float bestDistance = std::numeric_limits<float>::max();
                    for (std::size_t position = 0; position < positionEncodings.size(); ++position)
                    {
                        const std::vector<float> &positionEncoding{positionEncodings[position]};
                        float distance = 0.0f;

                        for (std::size_t i = 0; i < encodingSize; ++i)
                            distance += (predictedNextFrameEncoding[i] - positionEncoding[i])
                                    * (predictedNextFrameEncoding[i] - positionEncoding[i]);

                        if (distance < bestDistance)
                        {
                            bestPosition = position;
                            bestDistance = distance;
                        }
                    }

                    //--------------------------------------------------------------------
                    const std::size_t realRow = indices[start + contextFrames] / columns,
                                      realColumn = indices[start + contextFrames] % columns;

                    displayFrame[realRow][realColumn] = 'O';

                    std::cout << "\033[0H";
                    std::cout << std::string(columns + 2, '-') << '\n';
                    for (std::size_t row = 0; row < rows; ++row)
                    {
                        std::cout << '|';

                        for (std::size_t column = 0; column < columns; ++column)
                        {
                            if (displayFrame[row][column] == 'O')
                            {
                                std::cout << "\033[38;2;0;0;255mO";
                                std::cout << "\033[38;2;255;255;255m";
                            }
                            else
                                std::cout << displayFrame[row][column];
                        }

                        std::cout << "|\n";
                    }
                    std::cout << std::string(columns + 2, '-') << "\n\n";

                    displayFrame[realRow][realColumn] = ' ';
                    //--------------------------------------------------------------------

                    //--------------------------------------------------------------------
                    const std::size_t predictedRow = bestPosition / columns,
                                      predictedColumn = bestPosition % columns;

                    displayFrame[predictedRow][predictedColumn] = 'O';

                    std::cout << std::string(columns + 2, '-') << '\n';
                    for (std::size_t row = 0; row < rows; ++row)
                    {
                        std::cout << '|';

                        for (std::size_t column = 0; column < columns; ++column)
                        {
                            if (displayFrame[row][column] == 'O')
                            {
                                std::cout << "\033[38;2;255;0;0mO";
                                std::cout << "\033[38;2;255;255;255m";
                            }
                            else
                                std::cout << displayFrame[row][column];
                        }

                        std::cout << "|\n";
                    }
                    std::cout << std::string(columns + 2, '-') << "\n\n";
                    std::cout << std::flush;

                    displayFrame[predictedRow][predictedColumn] = ' ';
                    std::this_thread::sleep_for(std::chrono::milliseconds(25));
                }
                //--------------------------------------------------------------------
            }

            // train:
            worldModel.train(frames);
            ++trainingEpoch;
        }
    }};
    std::cin.get();
}