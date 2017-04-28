#ifndef __NEWTON_H_INCLUDED__
#define __NEWTON_H_INCLUDED__
#include <vector>
#include <cmath>
#include <iostream>
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
  auto gradient(FNC&& fnc, std::vector<T>&& val){
    return std::move(val);
  }

  /**recursive case*/
  template<typename FNC, typename T,  typename...Ts>
  auto gradient(FNC&& fnc, std::vector<T>&& val, const T& current, const Ts&... others){
    val.emplace_back(fnc(AutoDiff<T>(current, 1.0), others...).getDual());
    return gradient(
      [&](const Ts&... others){
        return fnc(current, others...);
      }, std::move(val), others...);
  }

  /**calling case*/
  template<typename FNC, typename T,  typename...Ts>
  auto gradient(FNC&& fnc, const T& current, const Ts&... others){
    return gradient(fnc, std::vector<T>(), current, others...);
  }



}

#endif
