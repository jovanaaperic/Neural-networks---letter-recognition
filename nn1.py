import cv2
import os

CLASSES = ['A','B','Б','Г','Д','Ђ','Е','Ж','З','И']

DATASET_DIR = './dataset/'

# preprocesiranje slika
def preprocess_image(path):

    img = cv2.imread( path, cv2.IMREAD_GRAYSCALE) # ucitavanje slike (crno belo)

    if img is None:
        return None

    _, img = cv2.threshold( img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # binarizacija (pravljenje cistog crno-belog kontrasta)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea) # uzmi konturu sa max povrsinom
        x,y,w,h = cv2.boundingRect(c)
        img = img[y:y+h, x:x+w] # cuva samo deo slike gde je slovo

    max_dim = max(img.shape[0],img.shape[1]) # smart skaliranje (trazi se veca strana: sirina/visina)

    scale = 20 / max_dim # faktor za skaliranje tkd da max dimenzija bude 20 piksela

    new_w = int(img.shape[1]*scale) # nove dimenzije
    new_h = int(img.shape[0]*scale)

    img = cv2.resize(img,(new_w,new_h)) # skaliranje

    canvas = np.zeros((28,28), dtype=np.uint8) # prazno crno platno (sve nule)

    x_off = (28-new_w)//2
    y_off = (28-new_h)//2


# lepimo skalirano slovo na centar canvasa
    canvas[
        y_off:y_off+new_h,
        x_off:x_off+new_w
    ] = img

    return (canvas.astype(np.float32) / 255.0) # 2d matrica -> 1d niz


def load_data():

    X=[]
    y=[]

    for idx, letter in enumerate(CLASSES):

        folder = os.path.join(DATASET_DIR,letter)

        for file in os.listdir(folder):

            path = os.path.join(folder,file )

            img = preprocess_image(path)

            if img is not None:
                X.append(
                    img.flatten()
                )
                y.append(idx)

    return np.array(X), np.array(y) # python liste -> numpy nizove


X, y = load_data()

indices = np.arange(len(X))
np.random.shuffle(indices) # nasumicno mesanje niz indeksa

X = X[indices]
y = y[indices]

# podela podataka 80% je ucenje (trening,) 20 je test
split = int(len(X)*0.8)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

print("Train:", len(X_train))
print("Test :", len(X_test))
