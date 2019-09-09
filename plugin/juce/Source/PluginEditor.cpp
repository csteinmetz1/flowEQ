/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
FlowEqAudioProcessorEditor::FlowEqAudioProcessorEditor (FlowEqAudioProcessor& p)
    : AudioProcessorEditor (&p), processor (p), lowpass(-12.0, 12.0, 22, 1000, 0.1, 10.0)
{
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    setSize (800, 400);
	
	addAndMakeVisible(lowpass);
}

FlowEqAudioProcessorEditor::~FlowEqAudioProcessorEditor()
{
}

//==============================================================================
void FlowEqAudioProcessorEditor::paint (Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (ResizableWindow::backgroundColourId));
}

void FlowEqAudioProcessorEditor::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..

	//eqKnobsSection.setBounds(0, getHeight() / 2, getWidth(), getHeight() / 2);
	lowpass.setBounds(getLocalBounds());
}
