/*
  ==============================================================================

    EQKnobsSection.h
    Created: 6 Sep 2019 11:53:35pm
    Author:  Trenton

  ==============================================================================
*/

#pragma once

#include "../JuceLibraryCode/JuceHeader.h"
#include "UISection.h"

#include <vector>

class EQBandKnobsSection : public Component
{
public:
    EQBandKnobsSection(float, float, float, float, float, float);
	void resized() override;
private:
    Slider cutoffKnob;
    Slider gainKnob;
    Slider bandwidthKnob;
};
