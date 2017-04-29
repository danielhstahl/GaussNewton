#include "grad.h"
#include <vector>
int main(){
    auto myTestFunc=[](const auto& x, const auto& y){
        return x*y;//gradient should be [y, x]
    };
    double testX=2;
    double testY=3;
    std::vector<double> answer({testY, testX});
    gradient(myTestFunc,  testX, testY);
}
    