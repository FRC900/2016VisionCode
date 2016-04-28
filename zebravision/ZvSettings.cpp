#include <tinyxml2.h>
#include "ZvSettings.hpp"

using namespace std;
using namespace tinyxml2;

static const char * const TOPLEVEL_NAME = "ZebraVision";

ZvSettings::ZvSettings(const std::string &filename) :
  filename_(filename)
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

/*
bool
ZvSettings::setInt(const std::string &sectionName,
                   const std::string &name,
                   int &value)
{
  XMLElement *elem = getElement(sectionName, name);
  if (elem && elem->QueryDoubleText(&tmpDouble) == XML_SUCCESS) {
    value = tmpDouble;
    return true;
  }
  else {
    return false;
  }
}

bool
ZvSettings::setDouble(const std::string &sectionName,
                      const std::string &name,
                      double &value)
{
  XMLElement *elem = getElement(sectionName, name);
  if (elem && elem->QueryDoubleText(&tmpDouble) == XML_SUCCESS) {
    value = tmpDouble;
    return true;
  }
  else {
    return false;
  }
}
*/

template <class T>
void
ZvSettings::set(const std::string &sectionName,
         const std::string &name,
         const T value)
{
  XMLElement *elem = getElement(sectionName, name);
  if (elem && elem->SetText(value) != XML_SUCCESS) {
    cerr << "Failed to set value, name=" << name << " value=" << value << endl;
  }
}

XMLElement*
ZvSettings::getElement(const std::string &sectionName,
                       const std::string &name)
{
  XMLElement *ret = NULL;
  XMLElement *topElem = xmlDoc_.FirstChildElement(TOPLEVEL_NAME);
  if (!topElem) {
    topElem = xmlDoc_.InsertFirstChild(xmlDoc_.NewElement(TOPLEVEL_NAME))->ToElement();
  }
  if (topElem) {
    XMLElement *secElem = topElem->FirstChildElement(sectionName.c_str());
    if (!secElem) {
      secElem = topElem->InsertFirstChild(xmlDoc_.NewElement(sectionName.c_str()))->ToElement();
    }
    if (secElem) {
      ret = secElem->FirstChildElement(name.c_str());
      if (!ret) {
        ret = secElem->InsertFirstChild(xmlDoc_.NewElement(name.c_str()))->ToElement();
      }
    }
  }
  if (xmlDoc_.SaveFile(filename_.c_str()) != XML_SUCCESS)
    cerr << "Failed to save settings file: " << filename_ << endl;
  return ret;
}
