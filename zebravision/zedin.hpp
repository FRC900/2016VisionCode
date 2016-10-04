// Base class for ZED-derived inputs (cameras and svo files)
// All this really does is provide a common getCameraParams call
// to all of them - the real work is done in those derived classes
#pragma once

//opencv include
#include <opencv2/core/core.hpp>
#include "mediain.hpp"

#ifdef ZED_SUPPORT
//zed include
#include <zed/Camera.hpp>
#endif

class ZvSettings;

class ZedIn : public MediaIn
{
	public:
		ZedIn(ZvSettings *settings = NULL);
		~ZedIn(void);

#ifdef ZED_SUPPORT
		CameraParams getCameraParams(bool left) const;

	protected:
		sl::zed::Camera* zed_;
#endif
};
