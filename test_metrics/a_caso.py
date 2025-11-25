import cdpam
loss_fn = cdpam.CDPAM()
wav_ref = cdpam.load_audio('/home/cerovaz/repos/ICML/Eulero_BackBone/runs/inference/cplx_dataset_outputs_24epoch/382.wav')
wav_out = cdpam.load_audio('/home/cerovaz/repos/data/jamendo_full/test_trimmed/382.mp3')

dist = loss_fn.forward(wav_ref,wav_out)
print(f"CDPAM distance: {dist}")