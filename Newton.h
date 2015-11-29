#ifndef __NEWTON_H_INCLUDED__
#define __NEWTON_H_INCLUDED__
#include <vector>
#include <cmath>
#include "Eigen/Dense"
#include <iostream>
#include <type_traits>
#include "AutoDiff.h"
//typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
class Newton{
public:

  /*This function is UNTESTED! */
  template<typename OBJFUNC> //one dimension
  void optimize(OBJFUNC&& objective, double &guess){ //guess is modified and used as the "result"
    double prec2=1;
    double d1=1;
    double d2=0;
    double p=0;
    int j=0;
    while(abs(d1)>precision1 && abs(prec2)>precision2 && j<maxNum){
      d1=objective(guess+dx);
      d2=objective(guess);
      p=objective(guess-dx);
      d2=(d1+p-2*d2)/(dx*dx);
      d1=(d1-p)/(2*dx);
      prec2=guess;
      guess=guess-d1/d2;
      prec2=guess-prec2;
      j++;
    }
    //return guess;
  }

  template<typename OBJFUNC, typename DERIV> //one dimension
  void zeros(OBJFUNC&& objective, DERIV&& derivative, double &guess){ //guess is modified and used as the "result"
    double prec2=1;
    int j=0;
    while(abs(prec2)>precision2 && j<maxNum){

      prec2=guess;
      guess=guess-objective(guess)/derivative(guess);
      prec2=guess-prec2;
      j++;
    }
    //return guess;
  }
  template<typename OBJFUNC, typename gs> //one dimension
  void zeros(OBJFUNC&& objective, gs &guesses){ //guess is modified and used as the "result"
    AutoDiff prec2(1, 0);
    AutoDiff guess;
    if(std::is_fundamental<gs>::value){
      guess.setStandard(guesses);
      guess.setDual(1);
    }
    else{
      guess.setStandard(guesses.getStandard());
      guess.setDual(1);
    }
    int j=0;
    while(abs(prec2.getStandard())>precision2 && j<maxNum){
      prec2=guess;
      guess=guess-objective(guess).getStandard()/objective(guess).getDual();
      prec2=guess-prec2;
      j++;
    }
    //return guess;
  }

/*  template<typename OBJFUNC> //one dimension
  void zeros(OBJFUNC&& objective, double &guess){
    double prec2=1;
    double d1=1;
    double d2=0;
    double p=0;
    int j=0;
    while(abs(d1)>precision1 && abs(prec2)>precision2 && j<maxNum){
      AutoDiff result=objective(AutoDiff(guess, 1));
      prec2=guess;
      guess=guess-result.getStandard()/result.getDual();
      prec2=guess-prec2;
      j++;
    }
  }*/
  /*This function is UNTESTED! */
  template<typename OBJFUNC>
  void optimize(OBJFUNC&& objective, std::vector<double> &guess){ //multidimension...
    int n=guess.size();
    Eigen::MatrixXd Hessian(n, n);
    Eigen::VectorXd Gradient(n);
    double prec2=1;
    double d1=1;
    double d2=0;
    double p=0;
    int k=0;
    //std::vector<double> guessP(n);

    while(prec2>precision2 && k<maxNum){
      for(int i=0; i<n; i++){
        //guessP[i]=guess[i]+dx;
        guess[i]=guess[i]-dx;
        //guess[i]+=dx;
        d1=objective(guess);
        guess[i]=guess[i]+2*dx;
        d2=objective(guess);
        Gradient(i)=(d2-d1)/(2*dx);
        //guess[i]=guess[i]+dx;
        for(int j=0; j<n; j++){
          d1=objective(guess);//one up
          guess[j]=guess[j]+dx;
          d2=objective(guess);//both up
          guess[i]=guess[i]-dx;
          p=objective(guess); //other one up
          guess[j]=guess[j]-dx; //all back to normal
          Hessian(i, j)=(d2-d1-p-objective(guess))/(dx*dx);
        }
      }
      Eigen::VectorXd result=Hessian.householderQr().solve(Gradient);
      prec2=0;
      for(int i=0; i<n; i++){
        d1=guess[i]-result(i);
        guess[i]=d1;
        prec2+=d1*d1;
      }

      k++;
    }
    std::cout<<"number of iterations: "<<k<<std::endl;
  }


  template<typename OBJFUNC>
  void optimize(std::vector<OBJFUNC>& objective, std::vector<double> &data, std::vector<double> &guess){ /*least squares: Gauss Newton*/
    int n=guess.size(); //number of parameters
    int m=objective.size(); //number of data to optimize over
    Eigen::MatrixXd Jacobian(m, n);
    Eigen::VectorXd Function(m);
    Eigen::VectorXd Parameters(n);
    double fnc=0;
    double prec2=1;
    k=0;
    double d1=0;
    double d2=0;
    std::vector<AutoDiff> augmentedGuess;
    for(int i=0; i<n; i++){
        augmentedGuess.push_back(AutoDiff(guess[i], 1));
    }
    while(prec2>precision2 && k<maxNum){
      for(int j=0; j<m; j++){
        AutoDiff func;
        for(int i=0; i<n; i++){
          augmentedGuess[i].setDual(0);
          func=objective[j](augmentedGuess);
          augmentedGuess[i].setDual(1);
          Jacobian(j, i)=func.getDual();
        }
        Function(j)=data[j]-func.getStandard();
      }
      Parameters=Jacobian.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Function); //not entirely sure what "ComputeThinU" is for...
      //std::cout<<Parameters<<std::endl;
      //std::cout<<Jacobian<<std::endl;
      //std::cout<<Function<<std::endl;
      prec2=0;
      for(int i=0; i<n; i++){
        d1=Parameters(i);
        //std::cout<<d1<<", ";
        guess[i]=guess[i]+d1;
        prec2+=d1*d1;
      }
      //std::cout<<""<<std::endl;
      k++;
    }
    //std::cout<<"number of iterations: "<<k<<std::endl;
  }
  template<typename OBJFUNC>
  void optimize(std::vector<OBJFUNC>& objective, std::vector<double> &data, std::vector<double> &guess, std::vector<std::vector<double> > &arguments){ /*least squares: Gauss Newton*/
    int n=guess.size(); //number of parameters
    int m=objective.size(); //number of data to optimize over
    Eigen::MatrixXd Jacobian(m, n);
    Eigen::VectorXd Function(m);
    Eigen::VectorXd Parameters(n);
    double fnc=0;
    double prec2=1;
    k=0;
    double d1=0;
    double d2=0;

    std::vector<AutoDiff> augmentedGuess;
    for(int i=0; i<n; i++){
        augmentedGuess.push_back(AutoDiff(guess[i], 0));
    }
    while(prec2>precision2 && k<maxNum){
      for(int j=0; j<m; j++){
        AutoDiff func;
        for(int i=0; i<n; i++){
          augmentedGuess[i].setDual(1);
          func=objective[j](augmentedGuess, arguments[j]);
          augmentedGuess[i].setDual(0);
          Jacobian(j, i)=func.getDual();
        }
        Function(j)=data[j]-func.getStandard();
      }
      Parameters=Jacobian.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Function); //not entirely sure what "ComputeThinU" is for...
      prec2=0;
      for(int i=0; i<n; i++){
        d1=Parameters(i);
        augmentedGuess[i].setStandard(augmentedGuess[i].getStandard()+d1);
        prec2+=d1*d1;
      }
      k++;
    }
    for(int i=0; i<n; i++){
        guess[i]=augmentedGuess[i].getStandard();
    }
    //std::cout<<"number of iterations: "<<k<<std::endl;
  }
  int getIterations();
  Newton();
  Newton(double, double, double);
private:
  int k;
  double precision1=.0000001;
  double precision2=.000001;
  double dx=.0001;
  int maxNum=500;
};

#endif
