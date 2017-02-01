#pragma once

#include "utility.h"

class Mandelbrot
{
public:
    typedef Mandelbrot _Myt;
    typedef double FLT;
    typedef std::complex<FLT> CT;
    const double BOUNDARY[4] = { 1, 1, -2, -1 }; // outer-most boundary: right, top, left, bottom

private:
    int threads = 0;

    FLT cutoff;
    int iters = 2048;
    int iter_step = 8;

    int coloring;
    double center_real;
    double center_imag;
    double zoom = 0; // log2

public:
    explicit Mandelbrot(double center_real = -0.5, double center_imag = 0, int coloring = 1, FLT cutoff = 1 << 8);

    int GetThreads() const { return this->threads; }
    void SetThreads(const int &threads) { this->threads = threads; }

    int GetIters() const { return this->iters; }
    void SetIters(const int &iters) { assert(iters > iter_step); this->iters = iters; }

    int GetIterStep() const { return this->iter_step; }
    void SetIterStep(const int &iter_step) { assert(iter_step > 0); this->iter_step = iter_step; }

    void SetCenter(const double &center_real, const double &center_imag)
    { this->center_real = center_real; this->center_imag = center_imag; }

    double GetZoom() const { return this->zoom; }
    void SetZoom(const double &zoom) { this->zoom = zoom; }

    std::complex<double> Position2Coordinate(int width, int height, int x, int y) const;

    void Render(uint8_t *dst, int height, int width, size_t stride, uint8_t max_val, uint8_t min_val) const;
    void Render(uint16_t *dst, int height, int width, size_t stride, uint16_t max_val, uint16_t min_val) const;
    void Render(float *dst, int height, int width, size_t stride, float max_val, float min_val) const;

private:
    void coordinateHelper(double *real_start, double *imag_start, double *real_step, double *imag_step,
        int width, int height) const;

    template <typename _Ty>
    void render(_Ty *dst, int height, int width, size_t stride, _Ty max_val, _Ty min_val) const;
};
