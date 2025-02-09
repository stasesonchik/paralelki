#include <iostream>
#include <cmath>
#include <vector>

#define _USE_MATH_DEFINES 

#ifndef TYPE
#define TYPE double  // Значение по умолчанию
#endif

int main() {
    int period = pow(10, 7);
    TYPE sum = 0;
    std::vector<TYPE> array(period);
    double Pi = M_PI;

    for (int i = 0; i < period; i++) {
        double value = sin(i * 2 * Pi / period);
        array[i] = static_cast<TYPE>(value);
        sum += array[i];
    }

    std::cout << (std::is_same<TYPE, double>::value ? "double" : "float") 
              << ": " << sum << std::endl;

    return 0;
}
