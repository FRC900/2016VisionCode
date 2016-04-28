#ifndef ZVSETTINGS_HPP__
#define ZVSETTINGS_HPP__

#include <iostream>
#include <tinyxml2.h>

class ZvSettings
{
public:
  ZvSettings(const std::string &filename);

  bool getInt(const std::string &sectionName,
              const std::string &name,
              int &value);

  bool getDouble(const std::string &sectionName,
                 const std::string &name,
                 double &value);

private:
  tinyxml2::XMLElement *getElement(const std::string &sectionName,
                                   const std::string &name);
  tinyxml2::XMLDocument xmlDoc_;
};

#endif
