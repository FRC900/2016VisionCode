#ifndef DETECT_STATE_HPP__
#define DETECT_STATE_HPP__

#include "classifierio.hpp"
#include "objdetect.hpp"

// A class to manage the currently loaded detector plus the state loaded
// into that detector.
class DetectState
{
	public:
		DetectState(const ClassifierIO &d12IO, const ClassifierIO &d24IO, const ClassifierIO &c12IO, const ClassifierIO &c24IO, float hfov, bool gie = false);
		~DetectState()
		{
			if (detector_)
				delete detector_;
		}
		bool update(void);
		void toggleGPU(void);
		void changeD12Model(bool increment);
		void changeD12SubModel(bool increment);
		void changeD24Model(bool increment);
		void changeD24SubModel(bool increment);
		void changeC12Model(bool increment);
		void changeC12SubModel(bool increment);
		void changeC24Model(bool increment);
		void changeC24SubModel(bool increment);
		std::string print(void) const;
		ObjDetect *detector(void)
		{
			return detector_;
		}
	private:
		ObjDetect    *detector_;
		ClassifierIO  d12IO_;
		ClassifierIO  d24IO_;
		ClassifierIO  c12IO_;
		ClassifierIO  c24IO_;
		Classifier   *d12_;
		Classifier   *d24_;
		Classifier   *c12_;
		Classifier   *c24_;
		float         hfov_;
		bool          gie_;
		bool          reload_;
};

#endif
