!pip install opencv-python

import os
import numpy as np
from google.colab import files

np.random.seed(42)

print("Upload dataset.zip")

uploaded = files.upload()

if not uploaded:
    print("GRESKA: Niste uploadovali fajl.")
else:
    for fn in uploaded.keys():
        if fn.endswith('.zip'):
            !unzip -q -o {fn} -d .
            if os.path.exists('dataset'):
                print("USPEH: Dataset raspakovan.")
                !ls dataset
            else:
                print("GRESKA: Folder 'dataset' nije pronadjen.")
