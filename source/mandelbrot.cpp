#include "mandelbrot.h"
#include <algorithm>

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

// Constructor
Mandelbrot::Mandelbrot(double center_real, double center_imag, int coloring, FLT cutoff)
    : center_real(center_real), center_imag(center_imag), coloring(coloring), cutoff(cutoff)
{
    assert(coloring > 0 && coloring < 2);
    assert(cutoff > 0);
}

// Methods
void Mandelbrot::coordinateHelper(double *real_start, double *imag_start, double *real_step, double *imag_step,
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

std::complex<double> Mandelbrot::Position2Coordinate(int width, int height, int x, int y) const
{
    double real_start, imag_start, real_step, imag_step;
    coordinateHelper(&real_start, &imag_start, &real_step, &imag_step, width, height);
    return std::complex<double>(real_start + real_step * x, imag_start + imag_step * y);
}

void Mandelbrot::Render(uint8_t *dst, int height, int width, size_t stride, uint8_t val_max, uint8_t val_min) const
{
    render(std::vector<uint8_t *>{dst}, height, width, stride, val_max, val_min);
}

void Mandelbrot::Render(uint16_t *dst, int height, int width, size_t stride, uint16_t val_max, uint16_t val_min) const
{
    render(std::vector<uint16_t *>{dst}, height, width, stride, val_max, val_min);
}

void Mandelbrot::Render(float *dst, int height, int width, size_t stride, float val_max, float val_min) const
{
    render(std::vector<float *>{dst}, height, width, stride, val_max, val_min);
}

// Implementation
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

template <typename _Ty>
void Mandelbrot::render(std::vector<_Ty *> dst, const int height, const int width, const size_t stride,
    const _Ty val_max, const _Ty val_min) const
{
    // Constants
    const int iters = (this->iters + iter_step - 1) / iter_step * iter_step; // set iters to a multiplier of iter_step
    const FLT cutoff_sqr = cutoff * cutoff;
    const _Ty val_range = val_max - val_min;

    // Coordinate transformation
    double real_start, imag_start, real_step, imag_step;
    coordinateHelper(&real_start, &imag_start, &real_step, &imag_step, width, height);

    // Kernel
#ifdef _OPENMP
    const int threads_origin = omp_get_max_threads();
    const int threads_new = this->threads > 0 ? this->threads : std::max(1, omp_get_num_procs() - this->threads);
    omp_set_num_threads(threads_new);
#endif

#pragma omp parallel for
    for (int j = 0; j < height; ++j)
    {
        std::vector<_Ty *> dstp;
        for (auto &e : dst) { dstp.push_back(ArrayIndex(e, stride, j)); }
        const FLT imag = static_cast<FLT>(imag_start + imag_step * j);
        int i = 0;

#if defined(__AVX__)
        const __m256d cutoff_sqr_v = _mm256_set1_pd(cutoff_sqr);
        const __m256d c_imag = _mm256_set1_pd(imag);

        static const ptrdiff_t simd_step = 4;
        const ptrdiff_t simd_residue = width % simd_step;
        const ptrdiff_t simd_width = width - simd_residue;

        for (; i < simd_width; ([&](){for (auto &e : dstp) e += simd_step; })())
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
            const __m256d cmp3 = _mm256_hadd_pd(cmp1, cmp2);
            const __m128d cmp = _mm_or_pd(_mm256_castpd256_pd128(cmp3), _mm256_extractf128_pd(cmp3, 1));

            alignas(32) int64_t cmp_array[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(cmp_array), _mm_castpd_si128(cmp));

            if (cmp_array[0] == 0 || cmp_array[1] == 0)
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
                const __m256d cmp = _mm256_cmp_pd(z_sqrabs, cutoff_sqr_v, _CMP_LE_OQ);
                sign_v = _mm256_or_pd(_mm256_andnot_pd(cmp, sign_v), _mm256_and_pd(cmp, _mm256_castsi256_pd(_mm256_set1_epi64x(n))));

                // judge if all elements in the vector are diverging
                const __m128i cmp2 = _mm_castpd_si128(_mm_or_pd(_mm256_castpd256_pd128(cmp), _mm256_extractf128_pd(cmp, 1)));
                converge = _mm_cvtsi128_si64(_mm_or_si128(cmp2, _mm_srli_si128(cmp2, 8)));
            }

            // Coloring
            alignas(32) int sign[simd_step * 2];
            _mm256_store_si256(reinterpret_cast<__m256i *>(sign), _mm256_castpd_si256(sign_v));

            switch (coloring)
            {
            case 2:
            {
                break;
            }
            case 1:
            default:
            {
                const __m128i sign_v2 = _mm_set_epi32(sign[6], sign[4], sign[2], sign[0]);
                const __m128 val = _mm_sub_ps(_mm_set1_ps(static_cast<float>(val_max)), _mm_mul_ps(
                    _mm_cvtepi32_ps(sign_v2), _mm_set1_ps(static_cast<float>(val_range) / iters)));

                if (std::is_same<_Ty, uint8_t>())
                {
                    __m128i temp = _mm_cvtps_epi32(val);
                    temp = _mm_packus_epi32(temp, temp);
                    temp = _mm_packus_epi16(temp, temp);
                    int res = _mm_cvtsi128_si32(temp);
                    for (auto &e : dstp) *reinterpret_cast<int *>(e) = res;
                }
                else if (std::is_same<_Ty, uint16_t>())
                {
                    __m128i temp = _mm_cvtps_epi32(val);
                    temp = _mm_packus_epi32(temp, temp);
                    __int64 res = _mm_cvtsi128_si64(temp);
                    for (auto &e : dstp) *reinterpret_cast<__int64 *>(e) = res;
                }
                else
                {
                    for (auto &e : dstp) _mm_storeu_ps(reinterpret_cast<float *>(e), val);
                }

                break;
            }
            }
        }
#elif defined(__SSE2__)
        const __m128d cutoff_sqr_v = _mm_set1_pd(cutoff_sqr);
        const __m128d c_imag = _mm_set1_pd(imag);

        static const ptrdiff_t simd_step = 2;
        const ptrdiff_t simd_residue = width % simd_step;
        const ptrdiff_t simd_width = width - simd_residue;

        for (; i < simd_width; ([&]() {for (auto &e : dstp) e += simd_step; })())
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
            const __m128d cmp = _mm_or_pd(_mm_unpacklo_pd(cmp1, cmp2), _mm_unpackhi_pd(cmp1, cmp2));

            alignas(32) int64_t cmp_array[2];
            _mm_store_si128(reinterpret_cast<__m128i *>(cmp_array), _mm_castpd_si128(cmp));

            if (cmp_array[0] == 0 || cmp_array[1] == 0)
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
                converge = _mm_cvtsi128_si64(_mm_or_si128(cmp, _mm_srli_si128(cmp, 8)));
            }

            alignas(32) int sign[simd_step * 2];
            _mm_store_si128(reinterpret_cast<__m128i *>(sign), sign_v);

            switch (coloring)
            {
            case 1:
            default:
                for (auto &e : dstp)
                {
                    e[0] = val_max - val_range * sign[0] / iters;
                    e[1] = val_max - val_range * sign[2] / iters;
                }
            }
        }
#endif
        for (; i < width; ++i, ([&]() {for (auto &e : dstp) ++e; })())
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
                for (auto &e : dstp)
                {
                    *e = val_max - val_range * n / iters;
                }
            }
        }
    }

#ifdef _OPENMP
    omp_set_num_threads(threads_origin);
#endif
}
