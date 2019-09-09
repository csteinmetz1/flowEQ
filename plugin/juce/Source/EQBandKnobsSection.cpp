/*
  ==============================================================================

    EQKnobsSection.cpp
    Created: 6 Sep 2019 11:53:35pm
    Author:  Trenton

  ==============================================================================
*/

#include "EQBandKnobsSection.h"

EQBandKnobsSection::EQBandKnobsSection(float gainMin, float gainMax, float cutoffMin, float cutoffMax, float bandwidthMin, float bandwidthMax)
{

	addAndMakeVisible(gainKnob);
	gainKnob.setRange(gainMin, gainMax);
	gainKnob.setValue(440);
	gainKnob.setSliderStyle(Slider::SliderStyle::Rotary);
	gainKnob.setTextBoxStyle(Slider::TextEntryBoxPosition::TextBoxLeft, true, 75, 25);
	gainKnob.setNumDecimalPlacesToDisplay(1);
    gainKnob.setTextValueSuffix(" dB");

	addAndMakeVisible(cutoffKnob);
	cutoffKnob.setRange(cutoffMin, cutoffMax);
	cutoffKnob.setValue(440);
	cutoffKnob.setSliderStyle(Slider::SliderStyle::Rotary);
	cutoffKnob.setTextBoxStyle(Slider::TextEntryBoxPosition::TextBoxLeft, true, 75, 25);
	cutoffKnob.setNumDecimalPlacesToDisplay(0);
    cutoffKnob.setTextValueSuffix(" Hz");

	addAndMakeVisible(bandwidthKnob);
	bandwidthKnob.setRange(bandwidthMin, bandwidthMax);
	bandwidthKnob.setValue(440);
	bandwidthKnob.setSliderStyle(Slider::SliderStyle::Rotary);
	bandwidthKnob.setTextBoxStyle(Slider::TextEntryBoxPosition::TextBoxLeft, true, 75, 25);
	bandwidthKnob.setNumDecimalPlacesToDisplay(2);
    //bandwidthKnob.setTextValueSuffix (" ");

}

void EQBandKnobsSection::resized()
{

	auto gainStart = 50;
	gainKnob.setBounds(gainStart, gainStart, 200, 100);
    cutoffKnob.setBounds(gainKnob.getX(), gainKnob.getBottom() + 5, 200, 100);
    bandwidthKnob.setBounds(cutoffKnob.getX(), cutoffKnob.getBottom() + 5, 200, 100);
    

}
