#include "PositionKF.hpp"

using namespace std;



PositionKF::PositionKF() {
	//sets the number of states,
	// the number of inputs,
	// the number of process noise random variables,
	// the number of measures and
	// the number of measurement noise random variables (respectively)
	setDim(4, 1, 2, 2, 2);

}

PositionKF::makeA() {

	A(1,1) = 

}
