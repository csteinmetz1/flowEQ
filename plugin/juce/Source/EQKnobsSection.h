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

class EQKnobsSection : public Component
{
public:
	EQKnobsSection();
	void resized() override;
private:
	OwnedArray<Slider> eqKnobs;
};
