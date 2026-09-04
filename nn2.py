import pickle

dense1 = Layer_Dense(784,128)
relu1 = Activation_ReLU()

dense2 = Layer_Dense(128,64)
relu2 = Activation_ReLU()

dense3 = Layer_Dense(64,10)

loss_activation = (Activation_Softmax_Loss_CategoricalCrossentropy())

optimizer = Optimizer_Adam()

EPOCHS = 500
#BATCH_SIZE = 32

print("TRENING...")

for epoch in range(EPOCHS):

    dense1.forward(X_train)
    # ulaze pixeli
    # full batch metoda: saljem ceo X_train po epohi, bez batch size

    relu1.forward(dense1.output) # filtriranje neg vred -> 0

    dense2.forward(relu1.output) # 2. sloj
    relu2.forward(dense2.output)

    dense3.forward(relu2.output) # izlaz iz mreze

    loss = loss_activation.forward(dense3.output,y_train)
    predictions = np.argmax(loss_activation.output,axis=1)
    accuracy = np.mean(predictions == y_train)


# backward propagation  (izmena tezina radi manje greske)
    loss_activation.backward(loss_activation.output,y_train)

    dense3.backward(loss_activation.dinputs) # obrnuti redosled

    relu2.backward(dense3.dinputs)
    dense2.backward(relu2.dinputs)

    relu1.backward(dense2.dinputs)
    dense1.backward(relu1.dinputs)

    optimizer.pre_update_params()

    optimizer.update_params(dense1) # menjanje tezina za sva 3 sloja
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)

    optimizer.post_update_params()

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch} | "
            f"Acc: {accuracy:.3f} | "
            f"Loss: {loss:.4f}"
        )

print("TRENING GOTOV")


# TEST: evaluacija
dense1.forward(X_test)
relu1.forward(dense1.output)

dense2.forward(relu1.output)
relu2.forward(dense2.output)

dense3.forward(relu2.output)

loss = loss_activation.forward(dense3.output, y_test)
predictions = np.argmax(loss_activation.output,axis=1)
test_acc = np.mean(predictions == y_test)

print()
print("TEST ACCURACY:", test_acc*100,"%")

# CONFUSION MATRIX (matrica zabune)
conf = np.zeros((10,10), dtype=int)

# na preseku reda (stv slovo) i kol (predikcija) -> brojac ++
for true, pred in zip(y_test, predictions):
    conf[true][pred] += 1

print()
print("CONFUSION MATRIX:")
print(conf)

# save model
model = {
    'w1': dense1.weights,
    'b1': dense1.biases,
    'w2': dense2.weights,
    'b2': dense2.biases,
    'w3': dense3.weights,
    'b3': dense3.biases
}

with open(
    "cyrillic_model.pkl",
    "wb"
) as f:
    pickle.dump(model,f)

print("MODEL SACUVAN")
