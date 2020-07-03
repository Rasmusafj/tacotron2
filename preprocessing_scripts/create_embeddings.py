from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import os

data_dir = "/media/user/data/complete_synth/"
encoder = VoiceEncoder()
counter = 0
end_this_shit = False
for path, subdirs, files in os.walk(data_dir):
    for name in files:
        extension = name[-3:]

        if extension == "wav":
            full_file_path = path + "/" + name
            wav = preprocess_wav(Path(full_file_path))
            should_reproces = False
            while len(wav) / 16000 < 3.00:
                wav = np.concatenate((wav, wav))
                should_reproces = True
            if should_reproces:
                embed = encoder.embed_utterance(wav)
                counter += 1
                np.save(path + "/" + name[:-4] + "_embed.npy", embed)

    print("Processed: " + str(counter))

#fpath = Path("/home/user/data/complete_synth/nst_synth/3468.wav")
#wav = preprocess_wav(fpath)

#encoder = VoiceEncoder()
#embed = encoder.embed_utterad med studiet så? Hvornår slutter semesteret? ance(wav)
#np.set_printoptions(precision=3, suppress=True)
#print(embed)