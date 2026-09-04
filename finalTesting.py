from google.colab import files
import matplotlib.pyplot as plt
import pickle

print("Upload test slike")

# ucitavanje pr naucene tezine

uploaded = files.upload()

with open(
    "cyrillic_model.pkl",
    "rb"
) as f:
    model = pickle.load(f)  # otvaranje sacuvanog modela

# vracanje naucenih tezina u mrezu
dense1.weights = model['w1']  # ucitava tezine za 1. dense sloj
dense1.biases = model['b1']

dense2.weights = model['w2']
dense2.biases = model['b2']

dense3.weights = model['w3']
dense3.biases = model['b3']

for fn in uploaded.keys():

    img = preprocess_image(fn)

    plt.imshow(
        img,
        cmap='gray'
    )
    plt.axis('off')
    plt.show() # prikazuje sliku

    x = img.reshape(1,-1) # pretvaranje slike u vektor (u 1 red)

# forward prolazak kroz mrezu
    dense1.forward(x)
    relu1.forward(dense1.output)

    dense2.forward(relu1.output)
    relu2.forward(dense2.output) # pretvara neg val -> 0

    dense3.forward(relu2.output)

    softmax = Activation_Softmax()
    softmax.forward(dense3.output) # pretvara output u procente, vrv

    probs = softmax.output[0] # uzima vrv od softmaxa
    pred = np.argmax(probs) # resenje

    print("PREDIKCIJA:",CLASSES[pred])

    for i,c in enumerate(CLASSES):
        print(
            c,
            f"{probs[i]*100:.2f}%"
        )
