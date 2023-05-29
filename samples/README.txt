Open the "demo_page.html" file in a browser to listen to the samples for results in the originial manuscript (we recommend using Google Chrome).
The page contains uncurated (not cherry-picked) audio samples for results in Tables 1,2,3 of the manuscript.
For Table3, we include the exact same audios that were used for the MOS test.

If the HTML demo page is sufficient for you, you may ignore the rest of this README file.
For those of you who prefer listening directly to the files, read the following instructions.


Each folder (Table1, Table2, Table3) contains uncurated (not cherry-picked) audio samples for results in Tables 1,2,3 of the manuscript.

- Table1 contains results for each row of Table1 in the manuscript.

- Table2 contains results for each column of Table2 in the manuscript.
	The title of the folders (0, +0.1, -0.1) represent the difference between the used epsilon value, and the epsilon* value obtained with the HiFi-GAN.

- For Table3, we include the exact same audios that were used for the MOS test.
	They are first separated by the dataset used for test (LibriTTS or VCTK).


General Rule:

- For folders that contain VC results, there will be triplet pairs of audio. 
	These are each source audio, target audio, and converted audio.
	(i_source.wav: source audio, i_target.wav: target audio, i_voice_synthesized.wav: converted audio)

- For folders that contain TTS results, there will be triplet pairs of audio. 
	These are each ground truth speech of the text, reference audio, and synthesized audio.
	(i_gt.wav: ground truth speech of text, i_target.wav: reference audio, i_voice_synthesized.wav: synthesized audio)
