#pragma once

#include <cassert>
#include <cstdint>
#include <cfloat>
#include <complex>
#include <string>
#include <vector>

typedef long double ldbl;

template <typename _Ty, typename _Ty2, typename _Ty3>
_Ty *ArrayIndex(_Ty *start, _Ty2 stride, _Ty3 index)
{
    return reinterpret_cast<_Ty *>(reinterpret_cast<uint8_t *>(start) + index * stride);
}

namespace std {
    template <typename _Ty>
    inline string to_string(const complex<_Ty> &_Val)
    {
        return to_string(_Val.real()) + " " + to_string(_Val.imag());
    }

    inline vector<string> split(const string &_Str, char _Key = ' ')
    {
        vector<string> splitted;

        size_t last = 0;
        for (size_t i = 0;;)
        {
            i = _Str.find_first_of(_Key, i);

            if (i == std::string::npos)
            {
                if (_Str.length() - last > 0) splitted.push_back(_Str.substr(last));
                break;
            }
            else
            {
                size_t count = i - last;
                if (count > 0) splitted.push_back(_Str.substr(last, count));
                last = ++i;
            }
        }

        return splitted;
    }
}
