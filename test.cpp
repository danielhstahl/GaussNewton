#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "Newton.h"


TEST_CASE("Test Straight Deriv", "[Functional]"){
    auto squareTestV=[](auto& val){
        return val*val-2.0;
    };
    auto deriv=[](auto& val){
        return 2*val;
    };
    auto guess=2.0;
    REQUIRE(newton::zeros(squareTestV, deriv, guess, .00001, 20)==Approx(sqrt(2.0)));
}  
  
TEST_CASE("Test autodiff Deriv", "[Functional]"){
    auto squareTestV=[](auto& val){
        return val*val-2.0;
    }; 
    auto guess=2.0;
    REQUIRE(newton::zeros(squareTestV,  guess, .00001, 20)==Approx(sqrt(2.0)));
}