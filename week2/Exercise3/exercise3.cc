#include <iostream>
#include <iostream>
using namespace std;

void rpsResult(char p1, char p2){
  if(p1 == 's' && p2 == 'p')
   cout<<"Player one wins!"<<endl;
  if(p1 == 'r' && p2 == 's')
   cout<<"Player one wins!"<<endl;
  if(p1 == 'p' && p2 == 'r')
   cout<<"Player one wins!"<<endl;
  if(p1 == 'p' && p2 == 's')
   cout<<"Player two wins!"<<endl;
  if(p1 == 's' && p2 == 'r')
   cout<<"Player two wins!"<<endl;
  if(p1 == 'r' && p2 == 'p')
   cout<<"Player two wins!"<<endl;
  if(p1==p2)
    cout<<"It's a tie!"<<endl;
}

int main(){
  char p1input;
  char p2input;

  cout<<"Let's play rock, paper, scissors!"<<endl;
  cout<<"Player one, please choose rock, paper, or scissors:"<<endl
    <<"Enter 'r' for rock."<<endl
    <<"Enter 'p' for paper."<<endl
    <<"Enter 's' for scissors."<<endl;

  while(!(cin>>p1input) || (p1input != 'r' && p1input != 'p' && p1input != 's')){
    cout<<"Invalid input."<<endl<<"Enter 'r' for rock."<<endl
      <<"Enter 'p' for paper."<<endl
      <<"Enter 's' for scissors."<<endl;
  }

  cout<<"Player two, please choose rock, paper, or scissors:"<<endl
    <<"Enter 'r' for rock."<<endl
    <<"Enter 'p' for paper."<<endl
    <<"Enter 's' for scissors."<<endl;

  while(!(cin>>p2input) || (p2input != 'r' && p2input != 'p' && p2input != 's')){
    cout<<"Invalid input."<<endl<<"Enter 'r' for rock."<<endl
      <<"Enter 'p' for paper."<<endl
      <<"Enter 's' for scissors."<<endl;
  }

  rpsResult(p1input,p2input);
}
