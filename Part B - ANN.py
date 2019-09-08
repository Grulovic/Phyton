##############################################
# Stefan Grulović (20150280) - Project part B
# 10/6/2019
# Part B was to build a Deterministic Artificial Neural Network
# consisting of binary state neurons, analogue state weights and a variable topology
# in order to classify a unary encoded number
# whether it is even or odd and whether is larger or equal to 4.
##############################################

# Libraries used for the program
import time
import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np


################################################################################################################################################################################
# PART B
################################################################################################################################################################################

hidden_layer_size = -1

# asks fro hidden layer size and set the global value for it
def hidden_size():
    global hidden_layer_size

    hidden_layer_size = -1

    while (hidden_layer_size <= 0):
        size = input("Enter the size of the hidden layer ( > 0 and Integer ): ")

        if size is '':
            size = int(3)

        try:
            hidden_layer_size = int(size)

            if (int(size) <= 0):
                print("Size smaller or equal to 0!")

        except ValueError:
            print("Not a integer!")

    print("Hidden layer size set to: " + str(size))

# asks for the number of epochs the program should run for and sets it as a global value
def epoch_size():
    epoch_num = 0
    while (epoch_num <= 0):
        epochs = input("Enter the number of epochs: ")

        if epochs is '':
            epochs = int(100)

        try:
            epoch_num = int(epochs)

            if (int(epochs) <= 0):
                print("Size smaller or equal to 0!")
        except ValueError:
            print("Not a integer!")

    print("Number of epochs set to: " + str(epoch_num))
    return epoch_num


# Three different activation functions Sigmoid, Tanh and Relu (only Sigmoid used)
def sigmoid(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))

def tanh(x, derivative=False):
    return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))


def relu(x, derivative=False):
    return np.log(1 + np.exp(x))


# Performs the ANN training by the user settings which were chosen and outputs the learnings

epochs = []
errors = []
training_progress = []
learning_step = 0.0000001

def training():
    global learning_step
    global epochs
    global errors
    global hidden_layer_size
    global training_progress

    epochs.clear()
    errors.clear()
    training_progress.clear()

    if (hidden_layer_size <= 0):
        hidden_size()

    training_data_filename = input("Enter training data set file name? ")

    if training_data_filename is '':
        training_data_filename = "training_data.txt"

    print("Training data set file name set to \"", training_data_filename, "\"")

    training_data_input = []
    training_data_output = []
    for line in open(training_data_filename):
        temp = line.split(",")
        training_data_input.append(list(map(int, list(temp[0]))))
        training_data_output.append([int(temp[1]), int(temp[2])])

    # print(training_data_input)
    # print(training_data_output)

    np.random.seed(int(time.time()))
    synMatrix0 = 2 * np.random.random((10, hidden_layer_size)) - 1
    synMatrix1 = 2 * np.random.random((hidden_layer_size, 2)) - 1

    epoch_num = epoch_size()
    measure = round(epoch_num / 100)

    X = np.array(training_data_input)
    y = np.array(training_data_output)

    activation_choice = activation_menu()

    if activation_choice is 2:
        set_learning_step()

    for epoch in range(epoch_num):

        # Feed forward through layers 0, 1, and 2
        l0 = X
        l1 = sigmoid(np.dot(l0, synMatrix0))
        l2 = sigmoid(np.dot(l1, synMatrix1))

        # how much did we miss the target value?
        l2_error = y - l2

        if (epoch % measure) == 0:
            errors.append(np.mean(np.abs(l2_error)))
            epochs.append(epoch)
            training_progress.append([str(epoch), ",", str(np.mean(np.abs(l2_error)))])

        if activation_choice is 1:

            l2_delta = l2_error * sigmoid(l2, derivative=True)

            #error backpropagation.
            l1_error = l2_delta.dot(synMatrix1.T)

            # this is the weight change training rule for this MLP.
            l1_delta = l1_error * sigmoid(l1, derivative=True)
        else:
            l2_delta = l2_error * learning_step

            # error backpropagation.
            l1_error = l2_delta.dot(synMatrix1.T)

            # this is the weight change training rule for this MLP.
            l1_delta = l1_error * learning_step


        # update weights
        synMatrix1 += l1.T.dot(l2_delta)
        synMatrix0 += l0.T.dot(l1_delta)

    np.savetxt("synapse_matrix0.txt", synMatrix0, fmt="%s")
    np.savetxt("synapse_matrix1.txt", synMatrix1, fmt="%s")
    # np.savetxt("trained.txt", l1, fmt="%s")
    np.savetxt("training_progress.txt", training_progress, fmt="%s")
    print("Files outputted \" synapse_matrix0.txt, synapse_matrix1.txt, trained.txt and training_progress.txt \"")
    print("Training finished and synapses for input and hidden layer saved!")


# Uses the data generated in the ttraining and performs clasification on the dest data and outputs the results on screen and as file
def classify_data():
    test_data_filename = input("Enter the test data set file name? ")

    if test_data_filename is '':
        test_data_filename = "test_data.txt"

    print("Test data set file name set to \"", test_data_filename, "\"")

    test_data = []
    for line in open(test_data_filename):
        test_data.append(list(map(int, line.split())))

    synMatrix0 = []
    for line in open("synapse_matrix0.txt"):
        synMatrix0.append(list(map(float, line.split())))
    synMatrix1 = []
    for line in open("synapse_matrix1.txt"):
        synMatrix1.append(list(map(float, line.split())))

    print("________________________________________")
    print("         TEST DATA CLASSIFICATION")
    print("----------------------------------------")
    classification_print = []
    for test_input in test_data:
        l0 = test_input
        l1 = sigmoid(np.dot(l0, synMatrix0))
        l2 = sigmoid(np.dot(l1, synMatrix1))


        print(test_input, int(round(l2[0])), int(round(l2[1])))


        test_input.append(",")
        test_input.append(int(round(l2[0])))
        test_input.append(",")
        test_input.append(int(round(l2[1])))
        classification_print.append(test_input)
    print("________________________________________")

    np.savetxt("test_data_output.txt", classification_print, fmt="%s")
    print("File outputted \" test_data_output.txt \"")


# displays the training progress in a line chart
def display_training_progress():
    global epochs
    global errors

    plt.plot(epochs, errors, color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Error(Mean)')
    plt.title('EPOCH / ERROR')
    plt.show()

# asks the user for which of the activation functions they want to use in the training process
def activation_menu():
    print("________________________________________")
    print("          ACTIVATION FUNCTION")
    print("----------------------------------------")
    print("[1] Sigmoid")
    print("[2] Manually choose learning step")
    print("________________________________________")

    option_size = 2
    column = 0
    activation_function_choice = 0

    while (int(column) <= 0) or (int(column) > option_size):
        column = input("Chose an activation function: ")

        if column is '':
            column = int(1)

        try:
            activation_function_choice = int(column)

            if (int(column) <= 0):
                print("Choice doesnt exist!")
        except ValueError:
            print("Not a integer!")

    if activation_function_choice is 1:
        print("Sigmoid chosen as the activation function!")
    else:
        print("Manually choosing the learning step chosen as the activation function!")

    return activation_function_choice


# sets the global learning step in case the user wants to manually set up the learning step
def set_learning_step():
    global learning_step

    learning_step = -1


    step = input("Enter the learning step: ")
    while True:
        if step is '':
            step = float(0.0000001)

        try:
            learning_step = float(step)

            if (float(step) <= 0.0):
                print("Size smaller or equal to 0!")
            break

        except ValueError:
            print("Not a float!")


    print("Learning step set to: " + str(step))
    return learning_step


################################################################################################################################################################################
# MENU
################################################################################################################################################################################

def quit():
    print("__________________________________")
    print("               QUIT")
    print("----------------------------------")
    print("Program will exit, bye!")
    print("__________________________________")
    raise SystemExit


def error_invalid():
    print("__________________________________")
    print("               ERROR")
    print("----------------------------------")
    print("Invalid Choice, Please try again!")
    print("__________________________________")


def menu():
    menu = {
        "1": ("Enter size of the hidden layer.", hidden_size),
        "2": ("Initiate training pass.", training),
        "3": ("Classify test data and display results.", classify_data),
        "4": ("Display training results graphics", display_training_progress),
        "5": ("QUIT", quit),
    }

    print("__________________________________")
    print("               MENU")
    print("----------------------------------")
    for key in sorted(menu.keys()):
        print("[" + key + "]:  " + menu[key][0])
    print("__________________________________")

    choice = input("Make a Choice: ")
    menu.get(choice, [None, error_invalid])[1]()


########################################################################################
# MAIN PROGRAM
########################################################################################
while (True):
    menu()
