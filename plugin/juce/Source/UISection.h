/*
  ==============================================================================

    UISection.h
    Created: 7 Sep 2019 12:24:45am
    Author:  Trenton

  ==============================================================================
*/

#pragma once
#include "../JuceLibraryCode/JuceHeader.h"

// if making a section of UI elements, inherit from this class first and it will
// automatically put updated slider values into a data structure in PluginProcessor.
class UISection : public Component,
				  public Slider::Listener
{
	void sliderValueChanged(Slider* slider) override;
};
