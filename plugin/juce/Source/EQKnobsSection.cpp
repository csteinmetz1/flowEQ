/*
  ==============================================================================

    EQKnobsSection.cpp
    Created: 6 Sep 2019 11:53:35pm
    Author:  Trenton

  ==============================================================================
*/

#include "EQKnobsSection.h"

EQKnobsSection::EQKnobsSection()
{
	for (int i = 0; i < 5; i++)
	{
		eqKnobs.add(new Slider());
		addAndMakeVisible(eqKnobs[i]);
		eqKnobs[i]->setRange(20, 20000);
		eqKnobs[i]->setValue(440);
		eqKnobs[i]->setSliderStyle(Slider::SliderStyle::Rotary);
		eqKnobs[i]->setTextBoxStyle(Slider::TextEntryBoxPosition::NoTextBox, true, 0, 0);
	}
}

void EQKnobsSection::resized()
{
	for (int i = 0; i < eqKnobs.size(); i++)
	{
		eqKnobs[i]->setBounds(i * (getWidth() / eqKnobs.size()) + 25, getHeight() / 2 - 50, 100, 100);
	}
}
