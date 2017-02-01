#include <iostream>
#include <iomanip>
#include <functional>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "mandelbrot.hpp"

typedef std::complex<double> CT;

static void TrackbarCallbackFunc(int pos, void *userdata)
{
    (*reinterpret_cast<std::function<void()> *>(userdata))();
}

static void MouseCallbackFunc(int event, int x, int y, int flags, void *userdata)
{
    (*reinterpret_cast<std::function<void(int event, int x, int y, int flags)> *>(userdata))(event, x, y, flags);
}

static int mandelbrot()
{
    const std::string winname_preview = "Mandelbrot Set <Preview>";
    const std::string winname_control = "Mandelbrot Set <Control>";
    int coloring = 1;
    int iter_step = 8;
    double zoom_max = 50;
    std::string input;

    // Standard I/O
    std::streamsize io_precision_origin = std::cout.precision();
    std::streamsize io_precision = 16;
    std::cout.setf(std::ios::fixed, std::ios::floatfield); // floatfield set to fixed

    // Set resolution
    int width = 1280;
    int height = 720;

    std::cout <<
        "Set rendering resolution - default " + std::to_string(width) + "x" + std::to_string(height) + ".\n"
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
    int center_choice = 5;
    CT center;

    std::cout << std::setprecision(io_precision) <<
        "Choose center position - default " + std::to_string(center_choice) + ".\n"
        "    0: input your custom coordinate\n"
        "    1: " << center_presets[0].real() << " " << center_presets[0].imag() << "\n"
        "    2: " << center_presets[1].real() << " " << center_presets[1].imag() << "\n"
        "    3: " << center_presets[2].real() << " " << center_presets[2].imag() << "\n"
        "    4: " << center_presets[3].real() << " " << center_presets[3].imag() << "\n"
        "    5: " << center_presets[4].real() << " " << center_presets[4].imag() << "\n"
        "    Leave blank to use the default setting.\n"
        "Your option: "
        << std::setprecision(io_precision_origin);
    std::getline(std::cin, input);

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

    // Choose mode
    int mode = 0;

    std::cout <<
        "Set start iteration increment - default " + std::to_string(mode) + ".\n"
        "    0: animating mode, automatically zoom in\n"
        "    1: interactive mode, preview is refreshed everytime you adjust parameters\n"
        "    Leave blank to use the default setting.\n"
        "Your option: ";

    while (true)
    {
        std::getline(std::cin, input);
        if (input == "") break;
        else mode = std::stoi(input);
        if (mode < 0 || mode > 1) std::cout << "Invalid input! Try again.\n";
        else break;
    }

    std::cout << std::endl;

    // For mode=1
    if (mode == 1)
    {
        // Help
        std::cout <<
            "Interactive Mode Tips:\n"
            "    1. Use control panel to adjust zoom, iters, etc.\n"
            "        1.1. Note that the zoom here is 100x (e.g. 500 means 5.00)."
            "    2. Use mouse on preview window to adjust center, zoom, etc.\n"
            "        2.1. Left button to drag the image.\n"
            "        2.2. Right click to change the center position.\n"
            "        2.3. Scroll up to zoom-in, scroll down to zoom-out.\n"
            "    3. Press ESC to exit.\n"
            ;

        // Initializations
        int zoom100 = 0;
        int iters = 1024;

        Mandelbrot<double> filter(center.real(), center.imag(), coloring);
        filter.SetIterStep(iter_step);
        cv::Mat image(cv::Size(width, height), CV_8UC1);

        // Render
        std::function<void()> refreshPreview([&]()
        {
            const double zoom = zoom100 / 100.;
            filter.SetZoom(zoom);
            filter.SetIters(iters);
            filter.Render(image.data, image.rows, image.cols, image.step, uchar(255), uchar(0));

            cv::imshow(winname_preview, image);
            cv::setWindowTitle(winname_preview, winname_preview + " - zoom: " + std::to_string(zoom)
                + ", max iterations: " + std::to_string(iters));
            cv::waitKey(1);
        });

        refreshPreview();

        // Trackbar
        void *trackbarCallbackData = reinterpret_cast<void *>(&refreshPreview);

        cv::namedWindow(winname_control);
        cv::createTrackbar("zoom", winname_control, &zoom100, 5000, TrackbarCallbackFunc, trackbarCallbackData);
        cv::createTrackbar("iters", winname_control, &iters, 8192, TrackbarCallbackFunc, trackbarCallbackData);
        cv::setTrackbarMin("iters", winname_control, 8);

        // Mouse
        int x_last = 0;
        int y_last = 0;

        std::function<void(int event, int x, int y, int flags)> mouseAction([&](int event, int x, int y, int flags)
        {
            switch (event)
            {
            case cv::EVENT_LBUTTONDOWN: // left drag - down
            {
                x_last = x;
                y_last = y;
                break;
            }
            case cv::EVENT_MOUSEMOVE: // left drag - move
            {
                if ((flags & cv::EVENT_FLAG_LBUTTON) && (x != x_last || y != y_last))
                {
                    center += filter.Position2Coordinate(width, height, x_last, y_last) - filter.Position2Coordinate(width, height, x, y);
                    x_last = x;
                    y_last = y;
                    filter.SetCenter(center.real(), center.imag());
                    refreshPreview();
                }
                break;
            }
            case cv::EVENT_LBUTTONUP: // left drag - release
            {
                center += filter.Position2Coordinate(width, height, x_last, y_last) - filter.Position2Coordinate(width, height, x, y);
                filter.SetCenter(center.real(), center.imag());
                std::cout << std::setprecision(io_precision) << "Center position changed to "
                    << center << std::setprecision(io_precision_origin) << std::endl;
                if (x != x_last || y != y_last) refreshPreview();
                break;
            }
            case cv::EVENT_RBUTTONUP: // right click
            {
                center = filter.Position2Coordinate(width, height, x, y);
                filter.SetCenter(center.real(), center.imag());
                std::cout << std::setprecision(io_precision) << "Center position changed to "
                    << center << std::setprecision(io_precision_origin) << std::endl;
                refreshPreview();
                break;
            }
            case cv::EVENT_MOUSEWHEEL: // scrolling
            {
                zoom100 += cv::getMouseWheelDelta(flags) / 30;
                cv::setTrackbarPos("zoom", winname_control, zoom100);
                refreshPreview();
                break;
            }
            default:
                break;
            }
        });
        void *mouseCallbackData = reinterpret_cast<void *>(&mouseAction);

        cv::setMouseCallback(winname_preview, MouseCallbackFunc, mouseCallbackData);

        // Running
        while (cv::waitKey(0) != 27); // exit when pressing ESC
        cv::destroyAllWindows();
        return 0;
    }

    // Set start zoom
    double zoom_start = 0;

    std::cout <<
        "Set start zoom (scale in log2) - a float in [0, +inf), default " + std::to_string(zoom_start) + ".\n"
        "    Leave blank to use the default setting.\n"
        "Your option: ";

    while (true)
    {
        std::getline(std::cin, input);
        if (input == "") break;
        else zoom_start = std::stod(input);
        if (zoom_start < 0) std::cout << "Invalid input! Try again.\n";
        else break;
    }

    std::cout << std::endl;

    // Set start max iterations
    int iters = 256;

    std::cout <<
        "Set start max iterations (relative to zoom=0) - an integer in [8, +inf), default " + std::to_string(iters) + ".\n"
        "    Leave blank to use the default setting.\n"
        "Your option: ";

    while (true)
    {
        std::getline(std::cin, input);
        if (input == "") break;
        else iters = std::stoi(input);
        if (iters < 8) std::cout << "Invalid input! Try again.\n";
        else break;
    }

    std::cout << std::endl;

    // Set iteration increment
    int iter_inc = 1;

    std::cout <<
        "Set start iteration increment - an integer in [0, +inf), default " + std::to_string(iter_inc) + ".\n"
        "    Leave blank to use the default setting.\n"
        "Your option: ";

    while (true)
    {
        std::getline(std::cin, input);
        if (input == "") break;
        else iter_inc = std::stoi(input);
        if (iter_inc < 0) std::cout << "Invalid input! Try again.\n";
        else break;
    }

    std::cout << std::endl;

    // Initializations
    double zoom = 0;
    while (zoom < zoom_start)
    {
        zoom += 0.02 * (1 + zoom * 0.05);
        iters += iter_inc;
    }

    Mandelbrot<double> filter(center.real(), center.imag(), coloring);
    filter.SetIterStep(iter_step);
    cv::Mat image(cv::Size(width, height), CV_8UC1);

    // Render
    int64_t start_t = cv::getTickCount();
    while (zoom < zoom_max)
    {
        int64_t start_t = cv::getTickCount();
        filter.SetZoom(zoom);
        filter.SetIters(iters);
        filter.Render(image.data, image.rows, image.cols, image.step, uchar(255), uchar(0));
        double duration = (cv::getTickCount() - start_t) * 1000 / cv::getTickFrequency();
        std::cout << "zoom: " << zoom << ", max iterations: " << iters << ", time elapsed: " << duration << "ms.\n";

        cv::imshow(winname_preview, image);
        cv::setWindowTitle(winname_preview, winname_preview + " - zoom: " + std::to_string(zoom)
            + ", max iterations: " + std::to_string(iters));
        if (cv::waitKey(1) == 27) break; // stop when pressing ESC
        zoom += 0.02 * (1 + zoom * 0.05);
        iters += iter_inc;
    }
    double duration = (cv::getTickCount() - start_t) * 1000 / cv::getTickFrequency();
    std::cout << "final zoom: " << zoom << ", max iterations: " << iters << ", total time elapsed: " << duration << "ms.\n";

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}

int main(int argc, char **argv)
{
    mandelbrot();
    return 0;
}
