#pragma once

#include "utility.h"

// Intrinsics
#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#elif defined(__SSE4_2__)
#include <nmmintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#elif defined(__SSSE3__)
#include <tmmintrin.h>
#elif defined(__SSE3__)
#include <pmmintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif

// OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

template <typename FLT = float>
class Mandelbrot
{
public:
    typedef Mandelbrot<FLT> _Myt;
    typedef std::complex<FLT> CT;
    const double BOUNDARY[4] = { 1, 1, -2, -1 }; // outer-most boundary: right, top, left, bottom

private:
    FLT cutoff;
    int iters = 2048;
    int iter_step = 8;

    int coloring;
    double center_real;
    double center_imag;
    double zoom = 0; // log2

public:
    explicit Mandelbrot(double center_real = -0.5, double center_imag = 0, int coloring = 1, FLT cutoff = 1 << 8);

    void SetCenter(const double &center_real, const double &center_imag)
    { this->center_real = center_real; this->center_imag = center_imag; }

    int GetIters() const { return iters; }
    void SetIters(const int &iters) { assert(iters > 1); this->iters = iters; }

    int GetIterStep() const { return iter_step; }
    void SetIterStep(const int &iter_step) { assert(iter_step > 0); this->iter_step = iter_step; }

    double GetZoom() const { return zoom; }
    void SetZoom(const double &zoom) { this->zoom = zoom; }

    std::complex<double> Position2Coordinate(int width, int height, int x, int y) const;

    template <typename _Ty>
    void Render(_Ty *dst, int height, int width, size_t stride, _Ty max_val, _Ty min_val) const;

private:
    void coordinateHelper(double *real_start, double *imag_start, double *real_step, double *imag_step,
        int width, int height) const;
};

template <typename FLT>
Mandelbrot<FLT>::Mandelbrot(double center_real, double center_imag, int coloring, FLT cutoff)
    : center_real(center_real), center_imag(center_imag), coloring(coloring), cutoff(cutoff)
{
    assert(coloring > 0 && coloring < 2);
    assert(cutoff > 0);
}

template <typename FLT>
void Mandelbrot<FLT>::coordinateHelper(double *real_start, double *imag_start, double *real_step, double *imag_step,
    int width, int height) const
{
    const double scale = pow(0.5, zoom);
    const double ratio = static_cast<double>(width) / height;
    double real_range = BOUNDARY[0] - BOUNDARY[2];
    double imag_range = BOUNDARY[1] - BOUNDARY[3];

    if (ratio > real_range / imag_range)
    {
        real_range = imag_range * ratio;
    }
    else if (ratio < real_range / imag_range)
    {
        imag_range = real_range / ratio;
    }

    real_range *= scale;
    imag_range *= scale;
    *real_start = center_real - real_range / 2;
    *imag_start = center_imag - imag_range / 2;
    *real_step = real_range / (width - 1);
    *imag_step = imag_range / (height - 1);
}

template <typename FLT>
std::complex<double> Mandelbrot<FLT>::Position2Coordinate(int width, int height, int x, int y) const
{
    double real_start, imag_start, real_step, imag_step;
    coordinateHelper(&real_start, &imag_start, &real_step, &imag_step, width, height);
    return std::complex<double>(real_start + real_step * x, imag_start + imag_step * y);
}

template <typename _Ty>
static inline _Ty _sqrabs_(const std::complex<_Ty> &c)
{
    return c.real() * c.real() + c.imag() * c.imag();
}

static inline __m128d _mm_sqrabs_pd(const __m128d r, const __m128d i)
{
    return _mm_add_pd(_mm_mul_pd(r, r), _mm_mul_pd(i, i));
}

static inline __m256d _mm256_sqrabs_pd(const __m256d r, const __m256d i)
{
    return _mm256_add_pd(_mm256_mul_pd(r, r), _mm256_mul_pd(i, i));
}

template <typename FLT>
template <typename _Ty>
void Mandelbrot<FLT>::Render(_Ty *dst, int height, int width, size_t stride, _Ty max_val, _Ty min_val) const
{
    // Constants
    const int iters = (this->iters + iter_step - 1) / iter_step * iter_step; // set iters to a multiplier of iter_step
    const FLT cutoff_sqr = cutoff * cutoff;

    // Coordinate transformation
    double real_start, imag_start, real_step, imag_step;
    coordinateHelper(&real_start, &imag_start, &real_step, &imag_step, width, height);

    // Kernel
#pragma omp parallel for
    for (int j = 0; j < height; ++j)
    {
        _Ty *dstp = ArrayIndex(dst, stride, j);
        const FLT imag = static_cast<FLT>(imag_start + imag_step * j);
        int i = 0;

#if defined(__AVX__)
        const __m256d cutoff_sqr_v = _mm256_set1_pd(cutoff_sqr);
        const __m256d c_imag = _mm256_set1_pd(imag);

        static const ptrdiff_t simd_step = 4;
        const ptrdiff_t simd_residue = width % simd_step;
        const ptrdiff_t simd_width = width - simd_residue;

        for (; i < simd_width; dstp += simd_step)
        {
            const FLT real = static_cast<FLT>(real_start + real_step * i++);
            const FLT real2 = static_cast<FLT>(real_start + real_step * i++);
            const FLT real3 = static_cast<FLT>(real_start + real_step * i++);
            const FLT real4 = static_cast<FLT>(real_start + real_step * i++);
            const __m256d c_real = _mm256_set_pd(real4, real3, real2, real);

            int n = 0;
            int64_t converge = -1;
            __m256d sign_v = _mm256_setzero_pd();
            __m256d z_real = _mm256_setzero_pd();
            __m256d z_imag = _mm256_setzero_pd();

            // Cardioid / Bulb checking
            const __m256d imag_sqr = _mm256_set1_pd(imag * imag);
            const __m256d sqrabs = _mm256_add_pd(imag_sqr, _mm256_mul_pd(c_real, c_real));
            const __m256d q = _mm256_add_pd(_mm256_set1_pd(0.0625),
                _mm256_sub_pd(sqrabs, _mm256_mul_pd(_mm256_set1_pd(0.5), c_real)));

            const __m256d cmp1 = _mm256_cmp_pd(_mm256_mul_pd(q, _mm256_add_pd(q, _mm256_sub_pd(c_real, _mm256_set1_pd(0.25)))),
                _mm256_mul_pd(_mm256_set1_pd(0.25), imag_sqr), _CMP_GE_OQ); // Cardioid
            const __m256d cmp2 = _mm256_cmp_pd(_mm256_add_pd(sqrabs, _mm256_add_pd(c_real, c_real)),
                _mm256_set1_pd(-0.9375), _CMP_GE_OQ); // Bulb
            const __m256d cmp = _mm256_hadd_pd(cmp1, cmp2);

            alignas(32) int64_t cmp_array[4];
            _mm256_store_si256(reinterpret_cast<__m256i *>(cmp_array), _mm256_castpd_si256(cmp));

            if ((cmp_array[0] | cmp_array[2]) == 0 || (cmp_array[1] | cmp_array[3]) == 0)
            { // skip iterations for points in set
                sign_v = _mm256_castsi256_pd(_mm256_set1_epi64x(iters));
            }
            else for (; n < iters && converge;)
            { // iterations
                for (int nupper = n + iter_step; n < nupper; ++n)
                { // inner loop for multiple iterations
                    const __m256d temp = _mm256_mul_pd(z_real, z_imag);
                    z_real = _mm256_add_pd(c_real, _mm256_sub_pd(_mm256_mul_pd(z_real, z_real), _mm256_mul_pd(z_imag, z_imag)));
                    z_imag = _mm256_add_pd(c_imag, _mm256_add_pd(temp, temp));
                }

                // update current iterations until diverging
                const __m256d z_sqrabs = _mm256_add_pd(_mm256_mul_pd(z_real, z_real), _mm256_mul_pd(z_imag, z_imag));
                __m256d cmp = _mm256_cmp_pd(z_sqrabs, cutoff_sqr_v, _CMP_LE_OQ);
                sign_v = _mm256_or_pd(_mm256_andnot_pd(cmp, sign_v), _mm256_and_pd(cmp, _mm256_castsi256_pd(_mm256_set1_epi64x(n))));

                // judge if all elements in the vector are diverging
                alignas(32) int64_t cmp_array[4];
                cmp = _mm256_hadd_pd(cmp, cmp);
                _mm256_store_si256(reinterpret_cast<__m256i *>(cmp_array), _mm256_castpd_si256(cmp));
                converge = cmp_array[0] | cmp_array[2];
            }

            alignas(32) int sign[simd_step * 2];
            _mm256_store_si256(reinterpret_cast<__m256i *>(sign), _mm256_castpd_si256(sign_v));

            switch (coloring)
            {
            case 1:
            default:
                dstp[0] = max_val - (max_val - min_val) * sign[0] / iters;
                dstp[1] = max_val - (max_val - min_val) * sign[2] / iters;
                dstp[2] = max_val - (max_val - min_val) * sign[4] / iters;
                dstp[3] = max_val - (max_val - min_val) * sign[6] / iters;
            }
        }
#elif defined(__SSE2__)
        const __m128d cutoff_sqr_v = _mm_set1_pd(cutoff_sqr);
        const __m128d c_imag = _mm_set1_pd(imag);

        static const ptrdiff_t simd_step = 2;
        const ptrdiff_t simd_residue = width % simd_step;
        const ptrdiff_t simd_width = width - simd_residue;

        for (; i < simd_width; dstp += simd_step)
        {
            const FLT real = static_cast<FLT>(real_start + real_step * i++);
            const FLT real2 = static_cast<FLT>(real_start + real_step * i++);
            const __m128d c_real = _mm_set_pd(real2, real);

            int n = 0;
            int64_t converge = -1;
            __m128i sign_v = _mm_setzero_si128();
            __m128d z_real = _mm_setzero_pd();
            __m128d z_imag = _mm_setzero_pd();

            // Cardioid / Bulb checking
            const __m128d imag_sqr = _mm_set1_pd(imag * imag);
            const __m128d sqrabs = _mm_add_pd(imag_sqr, _mm_mul_pd(c_real, c_real));
            const __m128d q = _mm_add_pd(_mm_set1_pd(0.0625),
                _mm_sub_pd(sqrabs, _mm_mul_pd(_mm_set1_pd(0.5), c_real)));

            const __m128d cmp1 = _mm_cmpge_pd(_mm_mul_pd(q, _mm_add_pd(q, _mm_sub_pd(c_real, _mm_set1_pd(0.25)))),
                _mm_mul_pd(_mm_set1_pd(0.25), imag_sqr)); // Cardioid
            const __m128d cmp2 = _mm_cmpge_pd(_mm_add_pd(sqrabs, _mm_add_pd(c_real, c_real)),
                _mm_set1_pd(-0.9375)); // Bulb

            alignas(32) int64_t cmp_array[4];
            _mm_store_si128(reinterpret_cast<__m128i *>(cmp_array), _mm_castpd_si128(cmp1));
            _mm_store_si128(reinterpret_cast<__m128i *>(cmp_array + 2), _mm_castpd_si128(cmp2));

            if ((cmp_array[0] | cmp_array[1]) == 0 || (cmp_array[2] | cmp_array[3]) == 0)
            { // skip iterations for points in set
                sign_v = _mm_set1_epi64x(iters);
            }
            else for (; n < iters && converge;)
            {
                for (int nupper = n + iter_step; n < nupper; ++n)
                { // inner loop for multiple iterations
                    const __m128d temp = _mm_mul_pd(z_real, z_imag);
                    z_real = _mm_add_pd(c_real, _mm_sub_pd(_mm_mul_pd(z_real, z_real), _mm_mul_pd(z_imag, z_imag)));
                    z_imag = _mm_add_pd(c_imag, _mm_add_pd(temp, temp));
                }

                // update current iterations until diverging
                const __m128d z_sqrabs = _mm_add_pd(_mm_mul_pd(z_real, z_real), _mm_mul_pd(z_imag, z_imag));
                const __m128i cmp = _mm_castpd_si128(_mm_cmple_pd(z_sqrabs, cutoff_sqr_v));
                sign_v = _mm_or_si128(_mm_andnot_si128(cmp, sign_v), _mm_and_si128(cmp, _mm_set1_epi64x(n)));

                // judge if all elements in the vector are diverging
                alignas(32) int64_t cmp_array[2];
                _mm_store_si128(reinterpret_cast<__m128i *>(cmp_array), cmp);
                converge = cmp_array[0] | cmp_array[1];
            }

            alignas(32) int sign[simd_step * 2];
            _mm_store_si128(reinterpret_cast<__m128i *>(sign), sign_v);

            switch (coloring)
            {
            case 1:
            default:
                dstp[0] = max_val - (max_val - min_val) * sign[0] / iters;
                dstp[1] = max_val - (max_val - min_val) * sign[2] / iters;
            }
        }
#endif

        for (; i < width; ++i, ++dstp)
        {
            const FLT real = static_cast<FLT>(real_start + real_step * i);
            const CT c(real, imag);

            int n = 0;
            CT z = 0;

            // Cardioid / Bulb checking
            const FLT imag_sqr = c.imag() * c.imag();
            const FLT sqrabs = c.real() * c.real() + imag_sqr;
            const FLT q = sqrabs - 0.5 * c.real() + 0.0625;

            if (q * (q + (c.real() - 0.25)) < 0.25 * imag_sqr // Cardioid
                || sqrabs + 2 * c.real() < -0.9375) // Bulb
            { // skip iterations for points in set
                n = iters;
            }
            else for (; n < iters && _sqrabs_(z) <= cutoff_sqr;)
            { // iterations
                for (int nupper = n + iter_step; n < nupper; ++n)
                { // inner loop for multiple iterations
                    z = z * z + c;
                }
            }

            switch (coloring)
            {
            case 1:
            default:
                *dstp = max_val - (max_val - min_val) * n / iters;
                //*dstp = n < iters ? max_val : min_val;
            }
        }
    }
}
