import numpy as np


# Dense layer - racuna izlaz nerona
# neuronski sloj klasa: z = xw + b
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):

        # random postavljanje tezina sa skaliranjem (He - inicijalizacija)
        self.weights = np.random.randn(
            n_inputs, n_neurons
        ) * np.sqrt(2.0 / n_inputs)

        self.biases = np.zeros((1, n_neurons)) # bias - dodajemo svakom neuronu (pocinje od nule)
        # neuron racuna output = input * weights + bias, bias daje vecu flexibilnost
        # bias pomera svaki neuron

    def forward(self, inputs):  # cuvam inputs za backpropagation
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases # racunam output sloja: input * weights + bias

    def backward(self, dvalues):
      # racuna gradijent za tezine, koliko treba promeniti svaku tezinu

        self.dweights = np.dot(self.inputs.T, dvalues) # T je transponovani input
        # T input * greska (dvalues)

        # i gradijent za bias
        self.dbiases = np.sum(   # ZBIR greske duz svih uzoraka
            dvalues,
            axis=0,
            keepdims=True
        )
# CUVA gradijent koji se prenosi u mrezi. prosledjuje gresku pr sloju
        self.dinputs = np.dot(
            dvalues,
            self.weights.T  # T je transponovana tezina, koristim kako bih imala info za pr sloj
            # trans tezina  * greska (dvalues)
        )


# ReLU - uklanja neg val -> 0
# uvodi nelinearnost, ReLU ne propagira grešku unazad tamo gde je ulaz bio negativan
# gradijent se prenosi SAMO tamo gde ima smisla
class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs  # cuvanje ulaza za backward propagaciju
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()  # kopiranje gradijenta
        self.dinputs[self.inputs <= 0] = 0 # tamo gde je ulaz bio <=0, gradijent postaje 0


# SOFTMAX - pravi procente i vrv
class Activation_Softmax:
# oduzimamo max val radi numericke stabilnosti
    def forward(self, inputs):
        exp_values = np.exp(
            inputs - np.max(  # izbegava prevelike eksponencijale
                inputs,
                axis=1,
                keepdims=True
            )
        )
# delimo eksponencijalne val sa sumom -> procenti
        probabilities = exp_values / np.sum(
            exp_values,
            axis=1,
            keepdims=True
        )

        self.output = probabilities


# LOSS - meri gresku
class Loss_CategoricalCrossentropy:

    def forward(self, y_pred, y_true): # meri koliko je model pogresio

        samples = len(y_pred) # br uzoraka
        # ogranicavanje val da ne budu 0 ili 1 da ih ne bi imali u log
        y_pred_clipped = np.clip(
            y_pred,
            1e-7,
            1 - 1e-7
        )
        # ako su labeli obicni br, uzmi vrv tacne klase npr: y_true = [0, 1, 1] niz brojeva
        # prvi uzorak  -> klasa 0
        # drugi uzorak -> klasa 1
        # treći uzorak -> klasa 1
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[  # uzmi vrv za svaku klasu po tom indexu
                range(samples),
                y_true
            ]
        else: # one-hot matrica
        #  y_true = [
        #    [1,0,0],
        #    [0,1,0],
        #    [0,1,0] ]  isto sto i [0,1,1]
        # vrv se racuna kao suma po tacnoj klasi za svaki uzorak
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        return -np.log(correct_confidences) # -log(vrv koja se odnosi na tacno predvidjenu klasu)
        # -log() predstavlja gresku koji zelimo da minimizujemo

    def calculate(self, output, y):  # avg gubitak
        sample_losses = self.forward(output, y) # forward je iznad
        return np.mean(sample_losses)


# spajanje softmax (pretvara izlaz u vrv) i merenje greske -> stabilniji i brzi gradijent
class Activation_Softmax_Loss_CategoricalCrossentropy:

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(   # vrati avg gresku
            self.output,
            y_true
        )

    def backward(self, dvalues, y_true):  # (self, greska iz pr sloja, prava klasa)
# backward - racuna kako da se koriguje greska
        samples = len(dvalues)  # uzorci

# formatu one-hot: svaki red je uzorak a kol je jedna klasa
        if len(y_true.shape) == 2:
            y_true = np.argmax( # linija matrice -> index klase
                y_true,
                axis=1
            )

        self.dinputs = dvalues.copy() # kopiramo greske



# gradient = predikcija − stvarna vrednost
        self.dinputs[range(samples), y_true] -= 1  # povecaj vrv pogresnih klasa
        # smanji vrv tacne klase
        # kako bi model bio bolji u predvidjanju

        self.dinputs /= samples # normalizacija greske ( /br slika)

# ADAM - menja tezine da bi model bio bolji
class Optimizer_Adam:

    def __init__(
        self,
        learning_rate=0.001,
        decay=1e-5,  # postepeno smanjuje learning rate
        epsilon=1e-7, # jako mali br, izbegava /0
        beta_1=0.9,  # faktori pamcenja
        beta_2=0.999  # faktori pamcenja
    ):

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0 # brojac koraka
        self.epsilon = epsilon
        self.beta_1 = beta_1 # pamti pr gradijente (momentum)
        self.beta_2 = beta_2 # pamti kvadratne gradijente (cache)

    def pre_update_params(self):
        # pre svakog update-a smanjuj learning rate
        # lr = lr / (1 + decay * iteration)
        if self.decay:
            self.current_learning_rate = (  # tehnika decay, usporavaj brzinu ucenja
                self.learning_rate *
                (1. / ( 1. + self.decay * self.iterations ))
            )

    def update_params(self, layer):

        if not hasattr(layer, 'weight_cache'):  # ako sloj nema faktore pamcenja

            layer.weight_momentums = np.zeros_like(layer.weights) # stavi nule
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

# komb starih i novih gradijenata:
        layer.weight_momentums = (self.beta_1 * layer.weight_momentums +(1-self.beta_1) * layer.dweights)  # stari
        layer.bias_momentums = (self.beta_1 *layer.bias_momentums +(1-self.beta_1) * layer.dbiases)
        weight_m_corrected = (layer.weight_momentums /(1 - self.beta_1 ** (self.iterations + 1)) ) # ispravljeni, koriste se za azuriranje tezina
        bias_m_corrected = (layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1)) )

        layer.weight_cache = (self.beta_2 *  layer.weight_cache + (1-self.beta_2) * layer.dweights**2) # stari cache
        layer.bias_cache = (self.beta_2 * layer.bias_cache + (1-self.beta_2) * layer.dbiases**2) # stari bias
        weight_c_corrected = (layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))) # i novi
        bias_c_corrected = (layer.bias_cache /  (1 - self.beta_2 **  (self.iterations + 1)))

# konacno azuriranje tezina:
        layer.weights += -(self.current_learning_rate *  weight_m_corrected / (np.sqrt(weight_c_corrected) + self.epsilon))
        layer.biases += -(self.current_learning_rate * bias_m_corrected /  (np.sqrt(bias_c_corrected) + self.epsilon))

    def post_update_params(self):
        self.iterations += 1
