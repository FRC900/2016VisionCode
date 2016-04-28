#include <tinyxml2.h>
#include "ZvSettings.hpp"

using namespace std;
using namespace tinyxml2;

ZvSettings::ZvSettings(const std::string &filename)
{
  if (xmlDoc_.LoadFile(filename.c_str()) == XML_SUCCESS) {
  }
  else {
    cerr << "Failed to open settings file" << endl;
  }
}

bool
ZvSettings::getInt(const std::string &sectionName,
                   const std::string &name,
                   int &value)
{
  int tmpInt;
  XMLElement *elem = getElement(sectionName, name);
  if (elem && elem->QueryIntText(&tmpInt) == XML_SUCCESS) {
    value = tmpInt;
    return true;
  }
  else {
    return false;
  }
}

bool
ZvSettings::getDouble(const std::string &sectionName,
                      const std::string &name,
                      double &value)
{
  double tmpDouble;
  XMLElement *elem = getElement(sectionName, name);
  if (elem && elem->QueryDoubleText(&tmpDouble) == XML_SUCCESS) {
    value = tmpDouble;
    return true;
  }
  else {
    return false;
  }
}

XMLElement*
ZvSettings::getElement(const std::string &sectionName,
                       const std::string &name)
{
  XMLElement *ret = NULL;
  XMLElement *topElem = xmlDoc_.FirstChildElement("ZebraVision");
  if (topElem) {
    XMLElement *secElem = topElem->FirstChildElement(sectionName.c_str());
    if (secElem) {
      ret = secElem->FirstChildElement(name.c_str());
    }
  }
  return ret;
}
