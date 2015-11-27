#include "Newton.h"
Newton::Newton(){

}
Newton::Newton(double prec1, double prec2, double dx_){
  precision1=prec1;
  precision2=prec2;
  dx=dx_;
}
int Newton::getIterations(){
    return k;
}