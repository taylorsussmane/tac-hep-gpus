#include <string>
#include<iostream>

struct TACHEPstudent{
  std::string name;
  std::string email;
  std::string experiment;
  std::string university;
  unsigned char PhDyear;
};


void print(const TACHEPstudent &person){
  std::cout<<"-----STUDENT INFORMATION-----"<<std::endl;
  std::cout<<"Name: "<<person.name<<std::endl;
  std::cout<<"Email: "<<person.email<<std::endl;
  std::cout<<"University: "<<person.university<<std::endl;
  std::cout<<"Experiment: "<<person.experiment<<std::endl;
  std::cout<<"Year: "<<person.PhDyear<<std::endl;
}

int main(){
  TACHEPstudent Taylor;
  Taylor.name = "Taylor Sussmane";
  Taylor.email = "tsussmane@wisc.edu";
  Taylor.experiment = "CMS";
  Taylor.university = "UW Madison";
  Taylor.PhDyear = '2';
  print(Taylor);

  TACHEPstudent Kyla;
  Kyla.name = "Kyla Martinez";
  Kyla.experiment = "CMS";
  Kyla.PhDyear = '2';
  Kyla.university = "UW Madison";
  print(Kyla);

  TACHEPstudent Kayleigh;
  Kayleigh.name = "Kayleigh Excell";
  Kayleigh.experiment = "Reuben";
  Kayleigh.PhDyear = '2';
  Kayleigh.university = "UW Madison";
  print(Kayleigh);

  TACHEPstudent Julianne;
  Julianne.name = "Julianne Starzee";
  Julianne.email = "jstarzee@umass.edu";
  Julianne.university = "UMass Amherst";
  print(Julianne);
}

