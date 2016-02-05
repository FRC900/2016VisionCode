#include "kalman/ekfilter.hpp"

class PositionKF : public Kalman::EKFilter<float,1> { //"1" is the starting index of matricies

public:

PositionKF();

protected:

void makeA();
void makeH();
void makeV();
void makeR();
void makeW();
void makeQ();
void makeProcess();
void makeMeasure();

};
