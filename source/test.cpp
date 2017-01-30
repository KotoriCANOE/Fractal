#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "mandelbrot.hpp"

int main(int argc, char **argv)
{
    typedef std::complex<double> CT;

    const std::string winname = "Mandelbrot Set";
    int coloring = 1;
    int iter_step = 8;
    std::string input;

    // Set resolution
    int width = 1280;
    int height = 720;

    std::cout << "Set rendering resolution - default 1280x720.\n"
        "    Leave blank to use the default setting.\n";

    while (true)
    {
        std::cout << "Your option: ";
        std::getline(std::cin, input);
        if (input == "") break;
        auto splitted = split(input, 'x');

        if (splitted.size() < 2)
        {
            std::cout << "Invalid input! Try again.\n";
        }
        else
        {
            width = std::stoi(splitted[0]);
            height = std::stoi(splitted[1]);
            break;
        }
    }

    std::cout << std::endl;

    // Choose center position
    // -1.401155, 0 | -1.4012, 0
    // -0.1528, 1.0397
    // -0.745429, 0.113008 | -0.74542900002, 0.11300799998
    // -0.77568377, 0.13646737 | -0.775683770001364, 0.136467369999090
    // -0.743643887037158704752191506114774, -0.131825904205311970493132056385139 | from FFmpeg
    std::vector<CT> center_presets;
    center_presets.push_back({ -1.4012, 0 });
    center_presets.push_back({ -0.1528, 1.0397 });
    center_presets.push_back({ -0.74542900002, 0.11300799998 });
    center_presets.push_back({ -0.775683770001364, 0.136467369999090 });
    center_presets.push_back({ -0.743643887037158704752191506114774, -0.131825904205311970493132056385139 });
    CT center;

    std::cout << "Choose center position:\n"
        "    0: input your custom coordinate\n"
        "    1: " + std::to_string(center_presets[0]) +"\n"
        "    2: " + std::to_string(center_presets[1]) + "\n"
        "    3: " + std::to_string(center_presets[2]) + "\n"
        "    4: " + std::to_string(center_presets[3]) + "\n"
        "    5: " + std::to_string(center_presets[4]) + " (default)\n"
        "    Leave blank to use the default setting.\n"
        "Your option: ";
    std::getline(std::cin, input);

    int center_choice = 5;
    if (input != "") center_choice = std::stoi(input);
    if (center_choice <= 0 || center_choice > center_presets.size())
    {
        while (true)
        {
            std::cout << "Input your custom corrdinate: ";
            std::getline(std::cin, input);
            auto splitted = std::split(input);

            if (splitted.size() < 2)
            {
                std::cout << "Invalid input! Try again.\n";
            }
            else
            {
                center._Val[0] = std::stod(splitted[0]);
                center._Val[1] = std::stod(splitted[1]);
                break;
            }
        }
    }
    else
    {
        center = center_presets[center_choice - 1];
    }

    std::cout << std::endl;

    // Set start zoom
    double zoom_start = 0;

    std::cout << "Set start zoom (scale in log2) - a float in [0, +inf), default 0.\n"
        "    Leave blank to use the default setting.\n"
        "Your option: ";

    while (true)
    {
        std::getline(std::cin, input);
        if (input == "") break;
        else zoom_start = std::stod(input);
        if(zoom_start < 0) std::cout << "Invalid input! Try again.\n";
        else break;
    }

    std::cout << std::endl;

    // Set start max iterations
    int iters = 256;

    std::cout << "Set start max iterations (relative to zoom=0) - an integer in [1, +inf), default 32.\n"
        "    Leave blank to use the default setting.\n"
        "Your option: ";

    while (true)
    {
        std::getline(std::cin, input);
        if (input == "") break;
        else iters = std::stoi(input);
        if (iters < 1) std::cout << "Invalid input! Try again.\n";
        else break;
    }

    std::cout << std::endl;

    double zoom = 0;
    while (zoom < zoom_start)
    {
        zoom += 0.02 * (1 + zoom * 0.05);
        iters += 1;
    }

    // Render
    Mandelbrot<double> filter(center.real(), center.imag(), coloring);
    filter.SetIterStep(iter_step);
    cv::Mat image(cv::Size(width, height), CV_8UC1);

    while (true)
    {
        int64_t start_t = cv::getTickCount();
        filter.SetZoom(zoom);
        filter.SetIters(iters);
        filter.Render(image.data, image.rows, image.cols, image.step, uchar(255), uchar(0));
        double duration = (cv::getTickCount() - start_t) * 1000 / cv::getTickFrequency();
        std::cout << "zoom: " << zoom << ", max iterations: " << iters << ", time cost: " << duration << "ms.\n";

        cv::imshow(winname, image);
        cv::setWindowTitle(winname, winname + " - zoom: " + std::to_string(zoom)
            + ", max iterations: " + std::to_string(iters));
        cv::waitKey(1);
        zoom += 0.02 * (1 + zoom * 0.05);
        iters += 1;
    }

    cv::destroyAllWindows();

    return 0;
}
