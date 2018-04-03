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


  constexpr int totalOptimizationSize=3;
  template<typename T>
  using OptimizationObject = std::array<T, totalOptimizationSize>;
  constexpr int functionOutputIndex=0;
  constexpr int previousInputIndex=1;
  constexpr int currentInputIndex=2;

  template<typename OptObj, typename T>
  auto setOptimizationObject(OptObj&& optObj, const T& functionOutput, const T& previousInput, const T& currentInput){
    optObj[functionOutputIndex]=functionOutput;
    optObj[previousInputIndex]=previousInput;
    optObj[currentInputIndex]=currentInput;
    return std::move(optObj);
  }

  template<typename T>
  auto createOptimizationObject(const T& functionOutput, const T& previousInput, const T& currentInput){
    OptimizationObject<T> optObj;
    optObj[functionOutputIndex]=functionOutput;
    optObj[previousInputIndex]=previousInput;
    optObj[currentInputIndex]=currentInput;
    return optObj;
  }




  template<typename OptObj, typename T>
  auto checkPrecision(const OptObj& optObj, const T& precision1, const T& precision2){
    
    return fabs(optObj[functionOutputIndex])>precision1&&fabs(optObj[currentInputIndex]-optObj[previousInputIndex])*.5>precision2;
  }

  template<typename OptObj>
  auto printResults(const OptObj& optObj){
    std::cout<<"Function evaluation: "<<optObj[functionOutputIndex]<<", Previous input: "<<optObj[previousInputIndex]<<", Current Input: "<<optObj[currentInputIndex]<<std::endl;
  }
  template<typename Index>
  auto printIteration(const Index& index){
    std::cout<<"Iteration: "<<index<<", ";
  }

  template<typename OptObj>
  auto getCurrent(const OptObj& optObj){
    return optObj[currentInputIndex];
  }
  template<typename OptObj>
  auto getPrevious(const OptObj& optObj){
    return optObj[previousInputIndex];
  }
  template<typename OptObj>
  auto getOutput(const OptObj& optObj){
    return optObj[functionOutputIndex];
  }




  template<typename OBJFUNC, typename DERIV, typename Guess, typename Index> //one dimension
  auto zeros(OBJFUNC&& objective, DERIV&& derivative, const Guess& guess, const Guess& precision1, const Guess& precision2, const Index& maxNum){ //({2.0, guess+1.0, guess})
    return getCurrent(futilities::recurse_move(maxNum, createOptimizationObject(2.0, guess+1.0, guess), [&](auto&& value, const auto& index){
      const auto guess=getCurrent(value);
      const auto result=objective(guess);
      #ifdef VERBOSE_FLAG
        printIteration(index);
      #endif
      return setOptimizationObject(value, result, guess, iterateStep(guess, result, derivative(guess)));
    }, [&](const auto& value){
      #ifdef VERBOSE_FLAG
        printResults(value);
      #endif
      return checkPrecision(value, precision1, precision2);
    }));
  }
  template<typename OBJFUNC, typename Guess, typename Index> //one dimension
  auto zeros(OBJFUNC&& objective, const Guess& guess, const Guess& precision1, const Guess& precision2, const Index& maxNum){ 
    AutoDiff<Guess> aGuess=AutoDiff<Guess>(guess, 1.0);
    auto getNewtonCoef=[](const auto& val, auto&& objResult){ 
      return AutoDiff<Guess>(iterateStep(val.getStandard(), objResult.getStandard(), objResult.getDual()), 1.0);
    };
    
    return getCurrent(futilities::recurse_move(maxNum, createOptimizationObject(AutoDiff<Guess>(2.0, 0.0), aGuess+1.0, aGuess), [&](auto&& value, const auto& index){
      const auto guess=getCurrent(value);
      auto result=objective(guess);
      #ifdef VERBOSE_FLAG
        printIteration(index);
      #endif
      return setOptimizationObject(value, result, guess, getNewtonCoef(guess, result));  
    }, [&](const auto& value){
      #ifdef VERBOSE_FLAG
        printResults(value);
      #endif
      return checkPrecision(value, precision1, precision2);
    })).getStandard();
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
  /**NOTE that the gradientTuple is not used, but is kept here because I find the implementation interesting*/
  template<typename FNC,  typename...Ts>
  auto gradientTuple(FNC&& fnc, const Ts&... myparms){
    return gradientHelperTuple(fnc, std::make_tuple(), myparms...);
  }



  /**returns tuple of tuples*/
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


  /**returns tuple of tuples*/
  template<typename FNC, typename T>
  auto gradientIterateApprox(FNC&& fnc, const T& tuple, double perterb){
    return tutilities::for_each(tuple, [&](const auto& val, const auto& index, auto&& priorTuple, auto&& nextTuple){
      return std::make_tuple(
        (fnc(
          std::tuple_cat(
            priorTuple, 
            std::make_tuple(val+perterb), //perterb   
            nextTuple
          )
        )-fnc(
          std::tuple_cat(
            priorTuple, 
            std::make_tuple(val-perterb), //perterb   
            nextTuple
          )
        ))/(2.0*perterb), 
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


  /**This is for gradDescentPack*/
  constexpr int GRAD=0;
  constexpr int PARAMVAL=1;
  /**This is for updatedTheta*/
  constexpr int PARAMPACK=0;
  constexpr int ERRORVAL=1;
  constexpr int OBJVAL=2;
 

  template<typename FNC, typename GFN, typename Index, typename Precision,typename T, typename...Params>
  auto gradientDescentGeneric(const GFN& gradientIter, const FNC& fnc, const Index& maxNum, const Precision& precision, const T& alpha,  const Params&... params){
    auto tupleFnc=[&](const auto& tuple){ //this converts the incoming function into one that takes tuples
      return tutilities::apply_tuple(fnc, tuple);
    };
    return std::get<PARAMPACK>(futilities::recurse_move(
      maxNum, 
      std::make_tuple(std::make_tuple(params...), precision+1.0, 5.0), ///inital guess, initial "error", initial value
      [&](auto&& updatedTheta, const auto& numberOfAttempts){
        #ifdef VERBOSE_FLAG
          printIteration(numberOfAttempts);
        #endif
        double error=0;
        return std::make_tuple(tutilities::for_each(
          gradientIter(tupleFnc, std::get<PARAMPACK>(updatedTheta)), //gradient at updatedTheta
          [&](const auto& gradDescentPack, const auto& index, auto&& priorTuple, auto&& nextTuple){
            error+=futilities::const_power(std::get<GRAD>(gradDescentPack), 2);
            return gradientDescentObjective(std::get<PARAMVAL>(gradDescentPack), alpha, std::get<GRAD>(gradDescentPack));
          }
        ), error, tupleFnc(std::get<PARAMPACK>(updatedTheta)));
      }, 
      [&](const auto& updatedTheta){
        auto error=std::get<ERRORVAL>(updatedTheta);
        #ifdef VERBOSE_FLAG
          auto objFn=std::get<OBJVAL>(updatedTheta);
          std::cout<<"Error Val: "<<error;
          std::cout<<", Obj Val: "<<objFn<<std::endl;
        #endif
        return error>precision;
      }
    ));
  }
  template<typename FNC, typename Index, typename Precision,typename T, typename...Params>
  auto gradientDescent(const FNC& fnc, const Index& maxNum, const Precision& precision, const T& alpha, const Params&... params){
    return gradientDescentGeneric(
      [](const auto& tupleFnc, const auto& updatedGradient){
        return gradientIterate(tupleFnc, updatedGradient);
      },
      fnc,
      maxNum,
      precision,
      alpha, 
      params...
    );
  }
  
  template<typename FNC, typename Index, typename Precision,typename T, typename...Params>
  auto gradientDescentApprox(const FNC& fnc, const Index& maxNum, const Precision& precision, const Precision& peterbation, const T& alpha, const Params&... params){
    return gradientDescentGeneric(
      [&](const auto& tupleFnc, const auto& updatedGradient){
        return gradientIterateApprox(tupleFnc, updatedGradient, peterbation);  
      },fnc,
      maxNum,
      precision,
      alpha, 
      params...
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




  template< typename OBJFUNC> //one dimension
  auto bisect(OBJFUNC&& objective, double begin, double end, double precision1, double precision2){
      double beginResult=objective(begin);
      double endResult=objective(end);
      double prec=2;
      auto maxNum=10000;//will get there befre 10000
      return isSameSign(beginResult, endResult)&&isEndBiggerThanBeginning(begin, end)?begin:getCurrent(futilities::recurse_move(maxNum, createOptimizationObject(beginResult, begin, end), [&](auto&& value, const auto& index){
        auto c=(getPrevious(value)+getCurrent(value))*.5;
        auto result=objective(c);
        #ifdef VERBOSE_FLAG
          printIteration(index);
        #endif
        if(isSameSign(result, getOutput(value))){
          return setOptimizationObject(value, result, c, getCurrent(value));
        }
        else{
          return setOptimizationObject(value, result, c, getPrevious(value));
        }
      }, [&](const auto& value){
        #ifdef VERBOSE_FLAG
          printResults(value);
        #endif
        return checkPrecision(value, precision1, precision2);
      }));
  }
}

#endif
