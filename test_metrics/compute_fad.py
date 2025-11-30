## WARNING: THIS IS NOT THE OFFICIAL FAD IMPEMENTATION. FOR THE OFFICIAL VERSION USE ANOTHER ENV

from frechet_audio_distance import FrechetAudioDistance

frechet = FrechetAudioDistance(
    model_name="clap",   # o "pann", "clap", "encodec"
    sample_rate=48000,
    submodel_name= "music_audioset"
)

fad_score = frechet.score(
    "/home/cerovaz/repos/data/jamendo_full/test_trimmed",  # dataset di riferimento
    "/home/cerovaz/repos/ICML/Eulero_BackBone/runs/inference/111_tris_cplx_24epoch",        # dataset generato/da valutare
    dtype="float32"
)

print("FAD:", fad_score)