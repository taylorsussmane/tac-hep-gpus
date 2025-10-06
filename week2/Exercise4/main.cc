#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "hh/t1.h"

#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TLorentzVector.h>



//------------------------------------------------------------------------------
// Particle Class
//
class Particle{

        public:
        Particle();
        Particle(double, double, double, double);
        double   pt, eta, phi, E, m, p[4];
        void     p4(double, double, double, double);
        void     print();
        void     setMass(double);
        double   sintheta();
};

class Lepton : public Particle{
        public:
        Lepton();
        int charge;
        void setCharge(int);
        void     print();
};

class Jet : public Particle{
        public:
        Jet();
        int hadronFlavor;
        void setHadronFlavor(int);
        void     print();
};

//------------------------------------------------------------------------------

//*****************************************************************************
//                                                                             *
//    MEMBERS functions of the Particle Class                                  *
//                                                                             *
//*****************************************************************************

//
//*** Default constructor ------------------------------------------------------
//
Particle::Particle(){
        pt = eta = phi = E = m = 0.0;
        p[0] = p[1] = p[2] = p[3] = 0.0;
}

//*** Additional constructor ------------------------------------------------------
Particle::Particle(double p0, double p1, double p2, double p3){
        pt = eta = phi = E = m = 0.0;
        p[0]=p0;
        p[1]=p1;
        p[2]=p2;
        p[3]=p3;
}

Lepton::Lepton(){
        pt = eta = phi = E = m = 0.0;
        p[0] = p[1] = p[2] = p[3] = 0.0;
        charge = 0;
}

Jet::Jet(){
        pt = eta = phi = E = m = 0.0;
        p[0] = p[1] = p[2] = p[3] = 0.0;
        hadronFlavor = 0;
}

//
//*** Members  ------------------------------------------------------
//
double Particle::sintheta(){
        double theta = 2*std::atan(std::exp(-eta));
        return std::sin(theta);
}

void Particle::p4(double pT, double eta, double phi, double energy){
        p[0] = energy;
        p[1] = pT*std::cos(phi);
        p[2] = pT*std::sin(phi);
        p[3] = pT*std::sinh(eta);
}

void Particle::setMass(double mass){
        m = mass;
}

void Lepton::setCharge(int chrg){
        charge = chrg;
}

void Jet::setHadronFlavor(int flvr){
        hadronFlavor = flvr;
}
//
//*** Prints 4-vector ----------------------------------------------------------
//
void Particle::print(){
        std::cout << std::endl;
        std::cout << "(" << p[0] <<",\t" << p[1]
                <<",\t"<< p[2] <<",\t"<< p[3] << ")"
                << "  " <<  sintheta() << std::endl;
}

void Lepton::print(){
        std::cout<< "Charge = " << charge << std::endl;
        std::cout << std::endl;
        std::cout << "(" << p[0] <<",\t" << p[1]
                <<",\t"<< p[2] <<",\t"<< p[3] << ")"
                << "  " <<  sintheta() << std::endl;

}

void Jet::print(){
        std::cout << "Hadron flavor = " << hadronFlavor << std::endl;
        std::cout << std::endl;
        std::cout << "(" << p[0] <<",\t" << p[1]
                <<",\t"<< p[2] <<",\t"<< p[3] << ")"
                << "  " <<  sintheta() << std::endl;

}

int main() {

        /* ************* */
        /* Input Tree   */
        /* ************* */

        TFile *f      = new TFile("input.root","READ");
        TTree *t1 = (TTree*)(f->Get("t1"));

        // Read the variables from the ROOT tree branches
        t1->SetBranchAddress("nleps", &nleps);
        t1->SetBranchAddress("lepPt",&lepPt);
        t1->SetBranchAddress("lepEta",&lepEta);
        t1->SetBranchAddress("lepPhi",&lepPhi);
        t1->SetBranchAddress("lepE",&lepE);
        t1->SetBranchAddress("lepQ",&lepQ);

        t1->SetBranchAddress("njets",&njets);
        t1->SetBranchAddress("jetPt",&jetPt);
        t1->SetBranchAddress("jetEta",&jetEta);
        t1->SetBranchAddress("jetPhi",&jetPhi);
        t1->SetBranchAddress("jetE", &jetE);
        t1->SetBranchAddress("jetHadronFlavour",&jetHadronFlavour);

        // Total number of events in ROOT tree
        Long64_t nentries = t1->GetEntries();

        for (Long64_t jentry=0; jentry<100;jentry++)
        {
                t1->GetEntry(jentry);
                std::cout<<"-------------------------------"<<
                        " Event " << jentry <<
                        " -------------------------------"<<
                        std::endl;


                for(int ilep = 0; ilep < nleps; ++ilep){
                        Lepton lep = Lepton();
                        lep.p4(lepPt[ilep], lepEta[ilep], lepPhi[ilep], lepE[ilep]);
                        lep.setCharge(lepQ[ilep]);
                        lep.print();
                }

                for(int ijet = 0; ijet < njets; ++ijet){
                        Jet jet = Jet();
                        jet.p4(jetPt[ijet], jetEta[ijet], jetPhi[ijet], jetE[ijet]);
                        jet.setHadronFlavor(jetHadronFlavour[ijet]);
                        jet.print();
                }


        } // Loop over all events

        return 0;
}
