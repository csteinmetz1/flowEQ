/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include "../JuceLibraryCode/JuceHeader.h"
#include "PluginProcessor.h"
#include "EQBandKnobsSection.h"

#include <vector>

//==============================================================================
/**
*/
class FlowEqAudioProcessorEditor  : public AudioProcessorEditor
{
public:
    FlowEqAudioProcessorEditor (FlowEqAudioProcessor&);
    ~FlowEqAudioProcessorEditor();

    //==============================================================================
    void paint (Graphics&) override;
    void resized() override;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    FlowEqAudioProcessor& processor;

	// UI sections
	EQBandKnobsSection lowpass;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (FlowEqAudioProcessorEditor)
};
