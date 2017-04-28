#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "Newton.h"


TEST_CASE("Test Straight Deriv", "[GaussNewton]"){
    auto squareTestV=[](auto& val){
        return val*val-2.0;
    };
    auto deriv=[](auto& val){
        return 2*val;
    };
    auto guess=2.0;
    REQUIRE(newton::zeros(squareTestV, deriv, guess, .00001, 20)==Approx(sqrt(2.0)));
}  
  
TEST_CASE("Test autodiff Deriv", "[GaussNewton]"){
    auto squareTestV=[](auto& val){
        return val*val-2.0;
    }; 
    auto guess=2.0;
    REQUIRE(newton::zeros(squareTestV,  guess, .00001, 20)==Approx(sqrt(2.0)));
}

TEST_CASE("Test gradient two args", "[GaussNewton]"){
    auto myTestFunc=[](const auto& x, const auto& y){
        return x*y;//gradient should be [y, x]
    };
    double testX=2;
    double testY=3;
    std::vector<double> answer={testY, testX};
    REQUIRE(newton::gradient(myTestFunc,  testX, testY)==Approx(answer));
}