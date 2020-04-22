import numpy as np
import matplotlib.pyplot as plt
import h5py
import time as t
import scipy
from PIL import Image
from scipy import ndimage


print("\n\nPROCESSAMENTO DEI DATA............\n")

def load_dataset():
    train_dataset = h5py.File("train_catvnoncat.h5", "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # Train set di feature
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # Train set di etichette

    test_dataset = h5py.File("test_catvnoncat.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # Test set di features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # Test set di etichette

    classes = np.array(test_dataset["list_classes"][:])  # Lista delle Classi

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# Carichiamo i dataset di immagini
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]


print("\n\nNumero esempi di apprendimento: m_train = " + str(m_train))
print("Numero esempi di testing: m_test = " + str(m_test))
print("Altezza/Larghezza di ogni immagine: num_px = " + str(num_px))
print("Ogni immagine ha dimensione: (" + str(num_px) + ", " + str(num_px) + ", 3)")

print("Dimensione del train_set_x: " + str(train_set_x_orig.shape))
print("Dimensione del train_set_y: " + str(train_set_y.shape))

print("Dimensione test_set_x: " + str(test_set_x_orig.shape))
print("Dimensione test_set_y: " + str(test_set_y.shape))

print("\n\nPROCESSAMENTO DEI DATASETS COMPLETATO\n")


train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T


train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


def sigmoid(z):
    """
    Calcoliamo la Sigmoid di Z

    Argomenti della funzione:
    z -- Array numpy di qualsiasi dimensione
    """

    s = 1 / (1 + np.exp(-z))

    return s


def initialize_with_zeros(dim):
    """
    Creiamo un vettore di zeri di dimensione (dim, 1) per w e inizializiamo b a zero

    Argomenti della funzione:
    dim -- dimensione del vettore w che vogliamo (in questo caso il numero dei parametri)

    return:
    w,b -- Parametri
    """

    w = np.zeros(shape=(dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, X, Y):
    """
    Implementiamo la funzione di costo e il suo gradiente

    Argomenti della funzione:
    w -- Pesi, un array numpy di dimensione (num_px * num_px * 3, 1)
    b -- Uno scalare
    X -- Dati di dimensione (num_px * num_px * 3, numero di esempi)
    Y -- Vettore delle etichette corrette (contenente 0 se non-gatto, 1 se gatto) di dimensione (1, numero di esempi)

    Return:
    cost -- Funzione di Costo per la Regressione Logistica
    dw -- Gradiente della perdita rispetto a w
    db -- Gradiente della perdita rispetto a b
    """

    m = X.shape[1]

    # Propagazione in avanti
    A = sigmoid(np.dot(w.T, X) + b)                                        # Calcolo della funzione di attivazione
    cost = (-1 / m) * np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1 - A)))  # Calcolo della funzione di costo

    # Propagazione all'indietro
    dw = 1 / m * (np.dot(X, (A - Y).T))
    db = 1 / m * (np.sum(A - Y))

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=True):
    """
    Funzione per ottimizzare i parametri w e b con l'algoritmo del gradiente discendente

    Argomenti della funzione:
    w -- Pesi, un array numpy di dimensione (num_px * num_px * 3, 1)
    b -- Uno scalare
    X -- Dati di dimensione (num_px * num_px * 3, numero di esempi)
    Y -- Vettore delle etichette corrette (contenente 0 se non-gatto, 1 se gatto) di dimensione (1, numero di esempi)
    num_iterations -- Numero di iterazioni per il ciclo di ottimizzazione
    learning rate -- fattore di apprendimento del gradiente discendente

    Return:
    params -- Dizionario contenente i parametri w e b
    grads -- Dizionario contenente i gradienti di w e b rispetto alla funzione di costo
    costs -- Lista di tutte le funzioni di costo calcolate durante l'ottimizzazione, la useremo per graficare la curva di apprendimento
    """

    costs = []

    for i in range(num_iterations):

        # Calcolo della funzione di costo e del gradiente
        grads, cost = propagate(w, b, X, Y)

        # Recuperiamo le derivate dai gradienti
        dw = grads["dw"]
        db = grads["db"]

        # Aggiorniamo i parametri
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("\nFunz. Costo alla iterazione %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    """
    Predice "Non-Oggetto" oppure "Oggetto" tramite i parametri appresi (w,b)

    Argomenti della funzione:
    w -- Pesi, un array numpy di dimensione (num_px * num_px * 3, 1)
    b -- Uno scalare
    X -- Dati di dimensione (num_px * num_px * 3, numero di esempi)

    Return:
    Y_prediction -- Un array numpy (vettore) contenente tutte le predizioni (0/1) per gli esempi in X
    """

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Calcola il vettore A predicendo le probabilità che un gatto sia nella foto
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        # Questa riga converte semplicemente la probabilità in predizione effettiva
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):
    """
    Funzione principale dove chiamiamo le altre funzioni ausiliarie

    Arguments:
    X_train -- Training set rappresentato da un array numpy di dimensione (num_px * num_px * 3, m_train)
    Y_train -- Etichette di apprendimento rappresentate da un array numpy di dimensione (vector) of shape (1, m_train)
    X_test -- Test set rappresentato da un array numpy di dimensione (num_px * num_px * 3, m_test)
    Y_test -- Etichette di test rappresentate da un array numpy (vettore) di dimensione (1, m_test)
    num_iterations -- Iperparametro che rappresenta il numero di iterazioni che effettua la funzione optimize()
    learning_rate -- Iperparametro che rappresenta il fattore di apprendimento

    Return:
    d -- Dizionario contenente le informazioni riguardo il modello.
    """

    # Inizializiamo i parametri

    tot_time = 0
    start = t.time()

    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradiente Discendente
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=True)

    # Ricaviamo i parametri w e b dal dizionario parameters
    w = parameters["w"]
    b = parameters["b"]

    # Predice i set di esempi
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Stampiamo l'accuratezza di riconoscimento
    print("\n\nTUTTE LE PROCEDURE COMPLETATE CORRETTAMENTE!")
    print("\nAccuratezza apprendimento: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("Accuratezza predizione: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    stop = t.time()
    tot_time = stop-start

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w, "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations,
         "tot_time": tot_time}

    return d


print("\nSTO IMPARANDO............")
print("CALCOLO DEI PARAMETRI E FUNZIONI............\n")

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2001, learning_rate=0.005)

print("Tempo totale di esecuzione:", d["tot_time"], "secondi \n\n")


while 1:

    fname = input("\nFile name:  ")

    if fname == "plot":
        costs = np.squeeze(d['costs'])
        plt.plot(costs)
        plt.ylabel('Funzione di Costo')
        plt.xlabel('Iterazioni (x100)')
        plt.title("Fattore di Apprendimento = " + str(d["learning_rate"]))
        plt.show()

    elif fname == "exit":
        break

    else:
        try:
            # Preprocessiamo l'immagine per adattarla al nostro algoritmo
            ImmagineTemp = np.array(ndimage.imread(fname, flatten=False))
            Immagine = scipy.misc.imresize(ImmagineTemp, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
            predizione = predict(d["w"], d["b"], Immagine)

            if predizione == 1.0:
                print("\nY=1.0, Nell'immagine è presente un gatto!")
            else:
                print("Y=0, Nell'immagine NON è presente un gatto!")
        except:
            print("Nome file non valido...\n")
