#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include <iostream>
#include "Newton.h"


TEST_CASE("Test Straight Deriv", "[GaussNewton]"){
    auto squareTestV=[](auto& val){
        return val*val-2.0;
    };
    auto deriv=[](auto& val){
        return 2*val;
    };
    auto guess=2.0;
    REQUIRE(newton::zeros(squareTestV, deriv, guess, .00001, .00001, 20)==Approx(sqrt(2.0)));
}  
  
TEST_CASE("Test autodiff Deriv", "[GaussNewton]"){
    auto squareTestV=[](auto& val){
        return val*val-2.0;
    }; 
    auto guess=2.0;
    REQUIRE(newton::zeros(squareTestV,  guess, .00001, .00001, 20)==Approx(sqrt(2.0)));
}

TEST_CASE("Test gradient two args", "[GaussNewton]"){
    auto myTestFunc=[](const auto& x, const auto& y){
        return x*y;//gradient should be [y, x]
    };
    double testX=2;
    double testY=3;
    std::vector<double> answer({testY, testX});
    REQUIRE(newton::gradient(myTestFunc,  testX, testY)==answer);
}

TEST_CASE("Test gradient one args", "[GaussNewton]"){
    auto myTestFunc=[](const auto& x){
        return x;//gradient should be [1]
    };
    double testX=2;
    std::vector<double> answer({1});
    REQUIRE(newton::gradient(myTestFunc,  testX)==answer);
}

TEST_CASE("Test gradient three args", "[GaussNewton]"){
    auto myTestFunc=[](const auto& x, const auto& y, const auto& z){
        return x*y*z;//gradient should be [yz, xz, yx]
    };
    double testX=2;
    double testY=3;
    double testZ=4;
    std::vector<double> answer({testY*testZ, testX*testZ, testY*testX});
    REQUIRE(newton::gradient(myTestFunc,  testX, testY, testZ)==answer);
}

TEST_CASE("Test gradientTuple three args", "[GaussNewton]"){
    auto myTestFunc=[](const auto& x, const auto& y, const auto& z){
        return x*y*z;//gradient should be [yz, xz, yx]
    };
    double testX=2;
    double testY=3;
    double testZ=4;
    auto answer=std::make_tuple(testY*testZ, testX*testZ, testY*testX);
    REQUIRE(newton::gradientTuple(myTestFunc,  testX, testY, testZ)==answer);
}


TEST_CASE("Test gradient descent one args", "[GaussNewton]"){
    auto myTestFunc=[](const auto& x){
        return (x-5)*(x-5); //minimum is at 5
    };
    double testX=2;
    
    //auto myResult=newton::gradientDescent(myTestFunc,  50, .00001, .5, testX);
    auto answer=5.0;
    REQUIRE(std::get<0>(newton::gradientDescent(myTestFunc, 50, .00001, .5, testX))==answer);
}


TEST_CASE("Test gradient descent two args", "[GaussNewton]"){
    auto myTestFunc=[](const auto& x, const auto& y){
        return (x-5.0)*(x-5.0)+(y-5.0)*(y-5.0); //minimum is at 5, 5
    };
    double testX=2;
    double testY=2;
    
    //auto myResult=newton::gradientDescent(myTestFunc,  50, .00001, .5, testX);
    auto answer=std::make_tuple(5.0, 5.0);

    REQUIRE(newton::gradientDescent(myTestFunc, 50, .00001, .5, testX, testY)==answer);
}

TEST_CASE("Test gradient descent two args and appox", "[GaussNewton]"){
    auto myTestFunc=[](const auto& x, const auto& y){
        return (x-5.0)*(x-5.0)+(y-5.0)*(y-5.0); //minimum is at 5, 5
    };
    double testX=2;
    double testY=2;
    
    //auto myResult=newton::gradientDescent(myTestFunc,  50, .00001, .5, testX);
    auto answer=std::make_tuple(5.0, 5.0);
    int maxNum=50;
    double prec=.00001;
    double peterb=.0001;
    double step=.5;
    REQUIRE(newton::gradientDescentApprox(myTestFunc, maxNum, prec, peterb, step, testX, testY)==answer);
}
/*
TEST_CASE("Test gradient descent two args and complex", "[GaussNewton]"){
    auto myTestFunc=[](const auto& x, const auto& y){
        return std::norm(x*std::complex<double>(1.0, 0.0)+y*std::complex<double>(0.0, 1.0));
    };
    auto testX=std::complex<double>(2, 0);
    auto testY=std::complex<double>(2, 0);
    
    //auto myResult=newton::gradientDescent(myTestFunc,  50, .00001, .5, testX);
    auto answer=std::make_tuple(std::complex<double>(0, 0), std::complex<double>(0, 0));

    REQUIRE(newton::gradientDescent(myTestFunc, 50, .00001, .5, testX, testY)==answer);
}*/


TEST_CASE("Test bisect", "[GaussNewton]"){
    auto myTestFunc=[](const auto& x){
        return x*x-4.0;
    };

    REQUIRE(newton::bisect(myTestFunc, 0, 5, .0001, .00001)==Approx(2.0));
}
