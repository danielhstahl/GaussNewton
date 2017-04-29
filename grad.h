#include <complex>
#include <vector>
/**base case*/
template<typename FNC, typename T>
auto gradientHelper(FNC&& fnc, std::vector<T>&& val){
    return std::move(val);
}

/**recursive case*/
template<typename FNC, typename T,  typename...Ts>
auto gradientHelper(const FNC& fnc, std::vector<T>&& val, const T& current, const Ts&... others){
    val.emplace_back(fnc(std::complex<T>(current, 1.0), others...).imag());
    return gradientHelper(
        [&](const Ts&... others){
        return fnc(current, others...);
    }, std::move(val), others...);
}

/**calling case*/
template<typename FNC, typename T, typename...Ts>
auto gradient(FNC&& fnc, const T& param, const Ts&... params){
    return gradientHelper(fnc, std::vector<T>(), param, params...);
}