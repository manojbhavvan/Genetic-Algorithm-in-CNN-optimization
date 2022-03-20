from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from numpy.random import randint
from keras import layers
from keras import models
from random import choice
from random import uniform


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train.shape, y_train.shape
x_test.shape, y_test.shape

train_images =x_train.astype('float32')/255
test_images=x_test.astype('float32')/255

train_labels=to_categorical(y_train)
test_labels=to_categorical(y_test)

val_images=train_images[:10000]
partial_images=train_images[10000:]

val_labels=train_labels[:10000]
partial_labels=train_labels[10000:]

def CNN_model( f1, f2, f3, k, a1, a2, d1, d2, op, ep):
  model = models.Sequential()
  model.add(layers.Conv2D(filters = f1, kernel_size = (k, k), activation = a1, input_shape = (32,32,3)))
  model.add(layers.Conv2D(filters = f1, kernel_size = (k, k), activation = a1))
  model.add(layers.MaxPooling2D(2,2))
  model.add(layers.Conv2D(filters = f2, kernel_size = (k, k), activation = a2))
  model.add(layers.Conv2D(filters = f2, kernel_size = (k, k), activation = a2))
  model.add(layers.MaxPooling2D(2,2))
  model.add(layers.Flatten())
  model.add(layers.Dropout(rate = d1))
  model.add(layers.Dense(units = f3, activation = a2))
  model.add(layers.Dropout(rate = d2))
  model.add(layers.Dense(10, activation= "softmax"))

  model.compile(loss = "categorical_crossentropy", optimizer = op, metrics = ["accuracy"])
  es = EarlyStopping(monitor="val_accuracy", patience = 7)
  model.fit(partial_images, partial_labels, validation_data=(val_images,val_labels), epochs=ep, batch_size = 100, callbacks = [es], verbose=0)

  return model

def initialization():
  parameters = {}
  f1 = choice([32, 64])
  parameters["f1"] = f1
  f2 = choice([64, 128])
  parameters["f2"] = f2
  f3 = choice([128, 256, 512])
  parameters["f3"] = f3
  k = choice([3,5])
  parameters["k"] = k
  a1 = choice(["relu", "selu", "elu"])
  parameters["a1"] = a1
  a2 = choice(["relu", "selu", "elu"])
  parameters["a2"] = a2
  d1 = round(uniform(0.1, 0.5), 1)
  parameters["d1"] = d1
  d2 = round(uniform(0.1, 0.5), 1)
  parameters["d2"] = d2
  op = choice(["adamax", "adadelta", "adam", "adagrad"])
  parameters["op"] = op
  ep = randint(50, 100)
  parameters["ep"] = ep
  return parameters

def generate_population(n):
  population = []
  for i in range(n):
    chromosome = initialization()
    population.append(chromosome)
  return population


def fitness_evaluation(model):
    metrics = model.evaluate(test_images, test_labels)
    return metrics[1]


def selection(population_fitness):
    total = sum(population_fitness)
    percentage = [round((x / total) * 100) for x in population_fitness]
    selection_wheel = []
    for pop_index, num in enumerate(percentage):
        selection_wheel.extend([pop_index] * num)
    parent1_ind = choice(selection_wheel)
    parent2_ind = choice(selection_wheel)
    return [parent1_ind, parent2_ind]


def crossover(parent1, parent2):
    child1 = {}
    child2 = {}

    child1["f1"] = choice([parent1["f1"], parent2["f1"]])
    child1["f2"] = choice([parent1["f2"], parent2["f2"]])
    child1["f3"] = choice([parent1["f3"], parent2["f3"]])

    child2["f1"] = choice([parent1["f1"], parent2["f1"]])
    child2["f2"] = choice([parent1["f2"], parent2["f2"]])
    child2["f3"] = choice([parent1["f3"], parent2["f3"]])

    child1["k"] = choice([parent1["k"], parent2["k"]])
    child2["k"] = choice([parent1["k"], parent2["k"]])

    child1["a1"] = parent1["a2"]
    child2["a1"] = parent2["a2"]

    child1["a2"] = parent2["a1"]
    child2["a2"] = parent1["a1"]

    child1["d1"] = parent1["d1"]
    child2["d1"] = parent2["d1"]

    child1["d2"] = parent2["d2"]
    child2["d2"] = parent1["d2"]

    child1["op"] = parent2["op"]
    child2["op"] = parent1["op"]

    child1["ep"] = parent1["ep"]
    child2["ep"] = parent2["ep"]
    return [child1, child2]


def mutation(chromosome):
    flag = randint(0, 40)
    if flag <= 20:
        chromosome["ep"] += randint(0, 10)
    return chromosome


generations = 3
threshold = 90
num_pop = 10

population = generate_population(num_pop)

for generation in range(generations):

    population_fitness = []
    for chromosome in population:
        f1 = chromosome["f1"]
        f2 = chromosome["f2"]
        f3 = chromosome["f3"]
        k = chromosome["k"]
        a1 = chromosome["a1"]
        a2 = chromosome["a2"]
        d1 = chromosome["d1"]
        d2 = chromosome["d2"]
        op = chromosome["op"]
        ep = chromosome["ep"]

        try:
            model = CNN_model(f1, f2, f3, k, a1, a2, d1, d2, op, ep)
            acc = fitness_evaluation(model)
            print("Parameters: ", chromosome)
            print("Accuracy: ", round(acc, 3))
        except:
            acc = 0
            print("Parameters: ", chromosome)
            print("Invalid parameters - Build fail")

        population_fitness.append(acc)

    parents_ind = selection(population_fitness)
    parent1 = population[parents_ind[0]]
    parent2 = population[parents_ind[1]]

    children = crossover(parent1, parent2)
    child1 = mutation(children[0])
    child2 = mutation(children[1])

    population.append(child1)
    population.append(child2)

    print("Generation ", generation + 1, " Outcome: ")
    if max(population_fitness) >= threshold:
        print("Obtained desired accuracy: ", max(population_fitness))
        break
    else:
        print("Maximum accuracy in generation {} : {}".format(generation + 1, max(population_fitness)))

    first_min = min(population_fitness)
    first_min_ind = population_fitness.index(first_min)
    population.remove(population[first_min_ind])
    second_min = min(population_fitness)
    second_min_ind = population_fitness.index(second_min)
    population.remove(population[second_min_ind])
