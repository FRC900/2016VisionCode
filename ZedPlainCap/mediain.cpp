#include "mediain.hpp"
#include <iostream>

using namespace std;

MediaIn::MediaIn()
{
}

int MediaIn::frameCount(void)
{
   return -1;
}

int MediaIn::frameCounter(void)
{
   return -1;
}

void MediaIn::frameCounter(int frameCount)
{
}

double MediaIn::getDepth(int x, int y)
{
   return -1000.;
}
