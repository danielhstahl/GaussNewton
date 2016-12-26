#ifndef __NEWTON_H_INCLUDED__
#define __NEWTON_H_INCLUDED__
#include <vector>
#include <cmath>
#include <iostream>
#include "AutoDiff.h"
#include "FunctionalUtilities.h"
namespace newton{

  template<typename OBJFUNC, typename DERIV, typename Guess, typename Index> //one dimension
  auto zeros(OBJFUNC&& objective, DERIV&& derivative, const Guess& guess, const Guess& precision2, const Index& maxNum){ //guess is modified and used as the "result"
    return futilities::recurse(maxNum, guess, [&](const auto& val, const auto& index){
      return val-objective(val)/derivative(val);
    }, [&](const auto& val, const auto& evalAtArg){
      return std::abs(evalAtArg)>precision2;//keep going criteria
    });
  }
  template<typename OBJFUNC, typename Guess, typename Index> //one dimension
  auto zeros(OBJFUNC&& objective, const Guess& guess, const Guess& precision2, const Index& maxNum){ //guess is modified and used as the "result"
    AutoDiff<Guess> aGuess=AutoDiff<Guess>(guess, 1.0);
    auto getNewtonCoef=[](const auto& val, auto&& objResult){
      objResult.setStandard(val.getStandard()-objResult.getStandard()/objResult.getDual());
      objResult.setDual(1.0);
      return std::move(objResult);
    };
    return futilities::recurse(maxNum, aGuess, [&](const auto& val, const auto& index){
      return getNewtonCoef(val, objective(val));
    }, [&](const auto& arg, const auto& evalAtArg){
      return std::abs(evalAtArg.getStandard())>precision2;//keep going criteria
    }).getStandard();
  }
}

#endif
