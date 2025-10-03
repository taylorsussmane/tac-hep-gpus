#include <iostream>

void swapValues(int &a, int &b){
  int c = a;
  a = b;
  b = c;
}

int main(){
  int A[10] = {2,4,6,8,10,12,14,16,18,20};
  int B[10] = {1,3,5,7,9,11,13,15,17,19};

  std::cout<<"Before swapping values:"<<std::endl;
  for(int i = 0; i < 10; i++){
    std::cout<<"A["<<i<<"]: "<<A[i]<<std::endl;
    std::cout<<"B["<<i<<"]: "<<B[i]<<std::endl;
  }

  std::cout<<"After swapping values:"<<std::endl;
  for(int i = 0; i < 10; i++){
    swapValues(A[i], B[i]);
    std::cout<<"A["<<i<<"]: "<<A[i]<<std::endl;
    std::cout<<"B["<<i<<"]: "<<B[i]<<std::endl;
  }
}
