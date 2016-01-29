#include "mediain.hpp"

MediaIn::MediaIn()
{
}

int MediaIn::frameCount(void) const
{
   return -1;
}

int MediaIn::frameCounter(void) const
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
