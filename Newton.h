#ifndef __NEWTON_H_INCLUDED__
#define __NEWTON_H_INCLUDED__

#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <complex>
#include "TupleUtilities.h"
#include "AutoDiff.h"
#include "FunctionalUtilities.h"




namespace newton{
  template<typename Val, typename Obj, typename Deriv>
  auto iterateStep(Val&& val, Obj&& obj, Deriv&& deriv){
    return val-obj/deriv;
  }
  template<typename OBJFUNC, typename DERIV, typename Guess, typename Index> //one dimension
  auto zeros(OBJFUNC&& objective, DERIV&& derivative, const Guess& guess, const Guess& precision2, const Index& maxNum){ 
    return futilities::recurse(maxNum, guess, [&](const auto& val, const auto& index){
      return iterateStep(val, objective(val), derivative(val));
    }, [&](const auto& val, const auto& evalAtArg){
      return std::abs(evalAtArg)>precision2;//keep going criteria
    });
  }
  template<typename OBJFUNC, typename Guess, typename Index> //one dimension
  auto zeros(OBJFUNC&& objective, const Guess& guess, const Guess& precision2, const Index& maxNum){ 
    AutoDiff<Guess> aGuess=AutoDiff<Guess>(guess, 1.0);
    auto getNewtonCoef=[](const auto& val, auto&& objResult){
      objResult.setStandard(iterateStep(val.getStandard(), objResult.getStandard(), objResult.getDual()));
      objResult.setDual(1.0);
      return std::move(objResult);
    };
    return futilities::recurse(maxNum, aGuess, [&](const auto& val, const auto& index){
      return getNewtonCoef(val, objective(val));
    }, [&](const auto& arg, const auto& evalAtArg){
      return std::abs(evalAtArg.getStandard())>precision2;//keep going criteria
    }).getStandard();
  }


 /**base case*/
  template<typename FNC, typename T>
  auto gradientHelper(FNC&& fnc, std::vector<T>&& val){
    return std::move(val);
  }

  /**recursive case*/
  template<typename FNC, typename T,  typename...Ts>
  auto gradientHelper(FNC&& fnc, std::vector<T>&& val, const T& current, const Ts&... others){
    val.emplace_back(fnc(AutoDiff<T>(current, 1.0), others...).getDual());
    return gradientHelper(
      [&](const auto&... remaining){
        return fnc(current, remaining...);
      }, std::move(val), others...);
  }
  /**calling case*/
  template<typename FNC, typename T,  typename...Ts>
  auto gradient(const FNC& fnc, const T& current, const Ts&... others){
    return gradientHelper(fnc, std::vector<T>(), current, others...);
  }
  





 /**base case*/
  template<typename FNC, typename...Tr>
  auto gradientHelperTuple(FNC&& fnc, std::tuple<Tr...>&& myResult){
    return std::move(myResult);
  }
  /**recursive case*/
  template<typename FNC, typename T,  typename...Ts, typename...Tr>
  auto gradientHelperTuple(FNC&& fnc, std::tuple<Tr...>&& myResult, const T& current, const Ts&... myparms){
    return gradientHelperTuple(
      [&](const auto&... remaining){
        return fnc(current, remaining...);
      }, std::tuple_cat(myResult, std::make_tuple(fnc(AutoDiff<T>(current, 1.0), myparms...).getDual())), myparms...);
  }
  template<typename FNC,  typename...Ts>
  auto gradientTuple(FNC&& fnc, const Ts&... myparms){
    return gradientHelperTuple(fnc, std::make_tuple(), myparms...);
  }


  template<typename FNC, typename T>
  auto gradientIterate(FNC&& fnc, const T& tuple){
    return tutilities::for_each(tuple, [&](const auto& val, const auto& index, auto&& priorTuple, auto&& nextTuple){
      return std::make_tuple(
        fnc(
            std::tuple_cat(
            priorTuple, 
            std::make_tuple(AutoDiff<std::remove_const_t<std::remove_reference_t<decltype(val)> > >(val, 1.0)),   
            nextTuple
          )
        ).getDual(), 
        val
      );
    });
  }



  /**base case*/
  template<typename T>
  auto pack_params(std::vector<T>&& val){
    return std::move(val);
  }
  /**recursive case*/
  template<typename T,  typename...Ts>
  auto pack_params(std::vector<T>&& val, const T& current, const Ts&... others){
    return pack_params(std::move(val), others...);
  }

  template<typename Theta, typename Alpha, typename Gradient>
  auto gradientDescentObjective(const Theta& theta, const Alpha& alpha, const Gradient& grad){
    return theta-alpha*grad;
  }


  template<typename T>
  auto square(const T& val){
    return val*val;
  }

  template<typename FNC, typename Index, typename Precision,typename T, typename...Params>
  auto gradientDescent(const FNC& fnc, const Index& maxNum, const Precision& precision, const T& alpha, const Params&... params){
    auto tupleFnc=[&](const auto& tuple){ //this converts the incoming function into one that takes tuples
      return tutilities::apply_tuple(fnc, tuple);
    };
    double tol=5;//this is bad practice! not "functional"!
    return futilities::recurse_move(
      maxNum, 
      std::make_tuple(params...), ///inital guess
      [&](const auto& updatedTheta, const auto& numberOfAttempts){
        tol=0;
        return tutilities::for_each(
          gradientIterate(tupleFnc, updatedTheta), //gradient at updatedTheta
          [&](const auto& grad, const auto& index, auto&& priorTuple, auto&& nextTuple){
            tol+=square(std::get<0>(grad));
            return gradientDescentObjective(std::get<1>(grad), alpha, std::get<0>(grad));
          }
        );
      }, 
      [&](const auto& updatedTheta){
        return tol>precision;
      }
    );
  }
  template<typename T, typename S>
  auto isSameSign(const T& x1, const S& x2){
    return x1*x2>0;
  }
  template<typename T, typename S>
  auto isEndBiggerThanBeginning(const T& x1, const S& x2){
    return x2>x1;
  }

  constexpr int arraySize=3;
  constexpr int beginIndex=0;
  constexpr int endIndex=1;
  constexpr int priorResultIndex=2;
  template< typename OBJFUNC> //one dimension
  auto bisect(OBJFUNC&& objective, double begin, double end, double precision1, double precision2){
      double beginResult=objective(begin);
      double endResult=objective(end);
      double prec=2;
      auto maxNum=10000;//will get there befre 10000
      return isSameSign(beginResult, endResult)&&isEndBiggerThanBeginning(begin, end)?begin:futilities::recurse_move(maxNum, std::array<double, arraySize>({begin, end, beginResult}), [&](const auto& value, const auto& index){
        auto c=(value[beginIndex]+value[endIndex])*.5;
        auto result=objective(c);
        return isSameSign(result, value[priorResultIndex])?std::array<double, arraySize>({c, value[endIndex], result}):std::array<double, arraySize>({c, value[beginIndex], result});
      }, [&](const auto& current){
        return fabs(current[priorResultIndex])>precision1&&fabs(current[endIndex]-current[beginIndex])*.5>precision2;
      })[0];
  }
}

#endif
