#ifndef __NEWTON_H_INCLUDED__
#define __NEWTON_H_INCLUDED__
//#include <vector>
#include <cmath>
#include <Eigen/Dense>
//#include <type_traits>
#include <iostream>
//typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
class Newton{
public:

  /*This function is UNTESTED! */
  template<typename OBJFUNC> //one dimension
  void optimize(OBJFUNC&& objective, double &guess){ //slightly inefficient to pass a primitive by reference
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

  /*This function is UNTESTED! */
  template<typename OBJFUNC>
  void optimize(OBJFUNC&& objective, std::vector<double> &guess){ //multidimension...only compile if OBJFUNC doesnt have an iterator (eg, isn't a vector)
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
    int k=0;
    double d1=0;
    double d2=0;
    while(prec2>precision2 && k<maxNum){
      for(int j=0; j<m; j++){
        for(int i=0; i<n; i++){
          guess[i]=guess[i]-dx;
          d1=objective[j](guess);
          guess[i]=guess[i]+2*dx;
          d2=objective[j](guess);
          Jacobian(j, i)=(d2-d1)/(2*dx);
          //std::cout<<(d2-d1)/(2*dx)<<std::endl;
          guess[i]=guess[i]-dx;
        }
        Function(j)=data[j]-objective[j](guess);
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
    std::cout<<"number of iterations: "<<k<<std::endl;
  }
  template<typename OBJFUNC>
  void optimize(std::vector<OBJFUNC>& objective, std::vector<std::vector<double> > &additionalParameters, std::vector<double> &data, std::vector<double> &guess){ /*least squares: Gauss Newton*/
    int n=guess.size(); //number of parameters
    int m=objective.size(); //number of data to optimize over
    Eigen::MatrixXd Jacobian(m, n);
    Eigen::VectorXd Function(m);
    Eigen::VectorXd Parameters(n);
    double fnc=0;
    double prec2=1;
    int k=0;
    double d1=0;
    double d2=0;
    while(prec2>precision2 && k<maxNum){
      for(int j=0; j<m; j++){
        for(int i=0; i<n; i++){
          guess[i]=guess[i]-dx;
          d1=objective[j](guess, additionalParameters[j]);
          guess[i]=guess[i]+2*dx;
          d2=objective[j](guess, additionalParameters[j]);
          Jacobian(j, i)=(d2-d1)/(2*dx);
          //std::cout<<(d2-d1)/(2*dx)<<std::endl;
          guess[i]=guess[i]-dx;
        }
        Function(j)=data[j]-objective[j](guess, additionalParameters[j]);
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
    std::cout<<"number of iterations: "<<k<<std::endl;
  }
  Newton();
  Newton(double, double, double);
private:
  double precision1=.0000001;
  double precision2=.000001;
  double dx=.0001;
  int maxNum=500;
};

#endif
