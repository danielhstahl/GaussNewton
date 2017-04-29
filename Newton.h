#ifndef __NEWTON_H_INCLUDED__
#define __NEWTON_H_INCLUDED__

#include <vector>
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


  template<typename FNC, typename Index, typename Precision,typename T, typename...Params>
  auto gradientDescent(const FNC& fnc, const Index& maxNum, const Precision& precision, const T& alpha, const Params&... params){
    auto tupleFnc=[&](const auto& tuple){ //this converts the incoming function into one that takes tuples
      return tutilities::expand_tuple(fnc, tuple);
    };
    return futilities::recurse_move(
      maxNum, 
      std::make_tuple(params...), ///inital guess
      [&](const auto& updatedTheta, const auto& numberOfAttempts){
        return tutilities::for_each(
          gradientIterate(tupleFnc, updatedTheta), //gradient at updatedTheta
          [&](const auto& grad, const auto& index, auto&& priorTuple, auto&& nextTuple){
            return gradientDescentObjective(std::get<1>(grad), alpha, std::get<0>(grad));
          }
        );
      }, 
      [&](const auto& updatedTheta){
        return true;//std::abs(evalAtArg)>precision;//keep going criteria
      }
    );
  }

}

#endif
