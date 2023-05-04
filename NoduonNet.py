import pandas as pd
import random as rng
# testing only #
import matplotlib.pyplot as plt

'''
A sumuonNetwork is comprised of three different sumuon structures

sumuon, which are the baseline input sumuons
Innersumuon, which are calculator sumuons
Outputsumuon, which is an Innersumuon with a title

The sumuon API includes the following

All sumuons:
weightedSum -> Sums the weighted sum of each of the sumuons in pulls from. Inp neurons return their value.

Input sumuon:
self.title -> The title of this input, such as: distanceToWall or timeSinceJump
self.value -> The value of the input, distanceToWall could have a self.value of 3.5
updateValue -> Changes the value of this sumuon's input to the desired quantity

Operator sumuons (inner and output):
self.sumuons -> A list containing all of the sumuons that are part of this sumuon's weighted sum
self.weights -> A list equal in length to self.sumuons, saving the weight for each one
determine -> Runs weighted sum and outputs true if positive and false if negative
changeWeights -> Accepts a weight change vector, adding each value to the proper weight
setWeights -> Changes weights to the input vector
getWeights -> Returns list of weights

Output sumuon:
self.title -> The title of the output usually an action or predictor, such as: jump, accelerate, expected y value

##########
sumuon Network
##########
Constructor (self, inputs, inners, outputs, outputNames, randWeights)
### Parameters ###
inputs -> A list of titles to form into input sumuons
inners -> A list with a number of entries equal to the number of inner sumuon columns/ranks. Each element contains a list
of length equal to the number of sumuons in that rank. Each element of the list represents a sumuon, which has a length
equal to the number of previous inner ranks + 1 for the input sumuons. Each element of the list is the index of a sumuon that
to link to.
outputs -> Has the same format as a single rank of inners
outputNames -> A list of names equal to the length of outputs, setting a name for each output.
randWeights -> A boolean determining whether or not to randomize the weights of operator sumuons
### Variables ###
self.buildingBlocks = saves the input lists in order to reconstruct self.
self.inputs -> A dictionary of input sumuons
self.inner -> A list of ranks/columns of sumuons, each containing a list of inner sumuons
self.outputs -> A dictionary of output sumuons
self.innerRank -> Number of inner ranks/columns
self.mutationFactor
### Methods ###
# Gets #
getOutput -> Takes the name of an input as an argument, returns that sumuon's weightedSum
getOutputs -> Returns a dictionary of each output sumuon's title:weightedSum
getWeights -> Takes a rank and index as input and returns a list equal to the weights of the selected sumuon
getInputValue -> Takes an input key and returns its current value
getInputTitles -> Return a list of input titles/keys
getOutputTitles -> Return a list of output/keys
# Sets #
setInputValue -> Takes a value and sets in to the given input key
setInputValues -> Takes a dictionary of input keys and new values, setting them to the new values
changesumuonWeight -> Takes a rank and index of a operator sumuon, changes it by the input weight vector
setsumuonWeight -> Takes a rank and index of an operator sumuon, sets it to the input weight vector
# Copy and Mutations #
copyNet -> Returns a copy of the sumuonNet. Does not copy current input values
copyStructure -> Returns a copy of the sumuonNet structure
mutate -> Alters a random based on mutationRate

##### TO ADD #####
New inner sumuon types:
Constant (bias) sumuons
Multiplier sumuons (multiply instead of add)

New learning functions:
Crossovers
Mutate structure like NEAT

Other:
Read/write NN to csv


'''

# Create the sumuon class
class Inpsumuon :

    # Constructor method
    def __init__(self, title, val):
        self.value = val
        self.title = title

    ### Methods ###

    def weightedSum(self):
        return self.value
    
    def updateValue(self, value):
        self.value = value


class Innersumuon:

    # Constructor method
    # Inputs will be a list of sumons or inner neurons
    def __init__(self, inpsumuons, rand):
        self.sumuons = inpsumuons
        if rand:
            self.weights = [rng.random() * rng.choice([-1,1]) for selfRank in range(len(self.sumuons))]
        else:
            self.weights = [0 for selfRank in range(len(self.sumuons))]
    
    ### Methods ###

    # Inp: N/A. Out: Weighted sum of inputs and weights
    def weightedSum(self):
        total = 0
        for sumuon, weight in zip(self.sumuons, self.weights):
            total += sumuon.weightedSum() * weight
        return total
    
    # determine will perform a weighted sum, returning true on a positive result and false on negative
    def determine(self):
        summed = self.weightedSum()
        if summed > 0:
            return True
        return False

    def changeWeights(self, weightVector):
        for selfRank in range(len(self.sumuons)):
            self.weights[selfRank] += weightVector[selfRank]

    def getWeights(self):
        return self.weights
    
    def setWeights(self, newWeights):
        if len(newWeights) == len(self.weights):
            for weight in range(len(self.weights)):
                self.weights[weight] = newWeights[weight]


class outputsumuon:

    # Constructor method
    def __init__(self, title, inps, rand):
        self.title = title
        self.sumuons = inps
        if rand:
            self.weights = [rng.random() * rng.choice([-1, 1]) for selfRank in range(len(self.sumuons))]
        else:
            self.weights = [0 for selfRank in range(len(self.sumuons))]

    ### Methods ###

    def weightedSum(self):
        summedTotal = 0
        for sumuon, weight in zip(self.sumuons, self.weights):
            summedTotal += sumuon.weightedSum() * weight
        return summedTotal

    def determine(self, parameters):
        summed = self.weightedSum(parameters)
        if summed > 0:
            return True
        return False
        
    def getWeights(self):
        return self.weights
    
    def changeWeights(self, weightVector):
        for selfRank in range(len(self.sumuons)):
            self.weights[selfRank] += weightVector[selfRank]

    def setWeights(self, newWeights):
        if len(newWeights) == len(self.weights):
            for weight in range(len(self.weights)):
                self.weights[weight] = newWeights[weight]
    

class sumuonNetwork:

    """
    sumuonNetwork is a combination of the different sumuons
    It saves them in sets of inputs, inner, and outputs.
    Inputs and and outputs can be referenced through their dictionary titles.
    Inner are saved as a list of lists for each rank.
    """

    # Constructor Method
    def __init__(self, inputs, inners, outputs, outNames, randWeights, mutationRate=0):
        self.buildingBlocks = [inputs,inners,outputs,outNames,mutationRate]
        self.inputs = {title:Inpsumuon(title, 0) for title in inputs}
        self.inner = []
        self.outputs = {}
        self.mutationRate = mutationRate
        for selfRank in range(len(inners)):
            self.inner.append([])

            for numsumuon in range(len(inners[selfRank])):
                sumInps = []
                for linkRank in range(len(inners[selfRank][numsumuon])):
                    for sum in inners[selfRank][numsumuon][linkRank]:
                        if linkRank == 0:
                            sumInps.append(self.inputs[sum])
                        else:
                            sumInps.append(self.inner[linkRank-1][sum])
                self.inner[selfRank].append(Innersumuon(sumInps, randWeights))

        for numsumuon in range(len(outputs)):
            sumInps = []
            for linkRank in range(len(outputs[numsumuon])):
                for sum in outputs[numsumuon][linkRank]:
                    if linkRank == 0:
                        sumInps.append(self.inputs[sum])
                    else:
                        sumInps.append(self.inner[linkRank-1][sum])
            self.outputs[outNames[numsumuon]] = outputsumuon(outNames[numsumuon], sumInps, randWeights)

        self.innerRank = len(self.inner)

    ### get Methods ###

    def getOutput(self, key):
        if key in self.outputs.keys():
            return self.outputs[key].weightedSum()
        else:
            print("No such output sumuon")
        
    def getWeights(self, rank, num):
        if rank == len(self.inner):
            return self.outputs[num].getWeights()
        elif rank in list(range(len(self.inner))):
            return self.inner[rank][num].getWeights()
        else:
            print("Invalid weight call, no such rank")

    def getOutputs(self):
        outs = {}
        for title in list(self.outputs.keys()):
            outs[title] = self.outputs[title].weightedSum()
        return outs
    
    def getInputValue(self, key):
        if key in self.inputs.keys():
            return self.inputs[key].weightedSum()
        else:
            print("No such input sumuon")

    def getInputTitles(self):
        return list(self.inputs.keys())

    def getOutputTitles(self):
        return list(self.outputs.keys())

    ### set/change Methods ###

    def setInputValue(self, key, value):
        if key in self.inputs.keys():
            self.inputs[key].updateValue(value)
        else:
            print("No such input sumuon")

    def setInputValues(self, valuesDict):
        for key in list(valuesDict.keys()):
            if not key in self.inputs:
                print("Invalid key detected, cancelling input changes")
                return 0
        for key in list(valuesDict.keys()):
            self.inputs[key].updateValue(valuesDict[key])

    def changesumuonWeight(self, rank, num, changeVector):
        if rank == len(self.inner):
            self.outputs[num].changeWeights(changeVector)
        elif rank in list(range(len(self.inner))):
            self.inner[rank][num].changeWeights(changeVector)
        else:
            print("Invalid weight change call, no such rank")

    def setsumuonWeight(self, rank, num, changeVector):
        if rank == len(self.inner):
            self.outputs[num].setWeights(changeVector)
        elif rank in list(range(len(self.inner))):
            self.inner[rank][num].setWeights(changeVector)
        else:
            print("Invalid weight set call, no such rank")

    ### Copying and Mutation methods ###

    def copyNet(self, mutate=False):
        copy = sumuonNetwork(self.buildingBlocks[0], self.buildingBlocks[1], self.buildingBlocks[2], self.buildingBlocks[3], False, self.buildingBlocks[4])
        for rank in range(len(copy.inner)):
            for sumuon in range(len(copy.inner[rank])):
                copy.setsumuonWeight(rank, sumuon, self.getWeights(rank, sumuon))
        for output in list(copy.outputs.keys()):
            copy.setsumuonWeight(len(copy.inner), output, self.getWeights(len(self.inner), output))
        if mutate: copy.mutate()
        return copy
    
    def copyStructure(self, randWeights = True):
        copy = sumuonNetwork(self.buildingBlocks[0], self.buildingBlocks[1], self.buildingBlocks[2], self.buildingBlocks[3], randWeights, self.buildingBlocks[4])
        return copy

    def mutate(self):
        for mutation in range(int(self.mutationRate)):
            targetRank = rng.randint(0, self.innerRank)
            if targetRank == self.innerRank:
                targetsumuon = rng.choice(self.getOutputTitles())
            else:
                targetsumuon = rng.randint(0, len(self.inner[targetRank]) - 1)
            mutate = [0 for i in range(len(self.getWeights(targetRank, targetsumuon)))]
            mutate[rng.randint(0,len(mutate)-1)] = rng.choice([-1,1]) * rng.random() / 10
            self.changesumuonWeight(targetRank, targetsumuon, mutate)
        # self.mutationRate += rng.choice([-1, 1]) * rng.random()
        if self.mutationRate < 0: self.mutationRate = 0

    def crossover(self, otherNN):
        pass


def main():

    # Test methods

    net = sumuonNetwork(["num1", "num2", "num3"],
                         [[[["num1", "num2", "num3"]],[["num1", "num2", "num3"]],[["num1", "num2", "num3"]]]], 
                         [[["num1", "num2", "num3"],[0,1,2]]], 
                          ["sum"],
                            True, mutationRate=2)
    out = "sum"

    # First Generation #
    bestInGen = []
    print("Generation 0")
    generation = [net.copyStructure(randWeights=True) for i in range(1000)]
    results = []
    for agent in generation:
        agent.setInputValues({"num1":rng.random()*rng.choice([-1,1]) * 5,"num3":rng.random()*rng.choice([-1,1]) * 5, "num2":rng.random()*rng.choice([-1,1]) * 5})
        summed = agent.getInputValue("num1") + agent.getInputValue("num3") + agent.getInputValue("num2")
        results.append(abs(agent.getOutput(out) - summed))
    print(min(results))
    bestInGen.append(min(results))
    print(generation[results.index(min(results))].getOutput(out))
    print([generation[results.index(min(results))].getInputValue(val) for val in net.getInputTitles()])
    nextGen = []
    for i in range(len(generation) // 25):
        target = results.index(min(results))
        nextGen.append(generation[target].copyNet(mutate=False))
        if i == 0:
            for b in range(5):
                nextGen.append(generation[target].copyNet(mutate=False))
        for j in range(len(generation) // 25 - i):
            nextGen.append(generation[target].copyNet(mutate=True))
        results[target] = max(results) + 1
    for i in range(len(generation) - len(nextGen)):
        nextGen.append(net.copyStructure(True))
    generation = nextGen

    for i in range(1000):
        results = []
        for agent in generation:
            agent.setInputValues({"num1":rng.random()*rng.choice([-1,1]) * 5,"num3":rng.random()*rng.choice([-1,1]) * 5, "num2":rng.random()*rng.choice([-1,1]) * 5})
            summed = agent.getInputValue("num1") + agent.getInputValue("num3") + agent.getInputValue("num2")
            results.append(abs(agent.getOutput(out) - summed))
        bestInGen.append(min(results))

        if (i+1) % 100 == 0: print(f"Generation {i+1}")
        nextGen = []
        for i in range(len(generation) // 25):
            target = results.index(min(results))
            nextGen.append(generation[target].copyNet(mutate=False))
            for j in range(len(generation) // 25 - i):
                nextGen.append(generation[target].copyNet(mutate=True))
            results[target] = max(results) + 1
        for i in range(len(generation) - len(nextGen)):
            nextGen.append(net.copyStructure(True))
        generation = nextGen
    plt.plot(bestInGen)
    plt.yscale("log")
    plt.show()
    


if __name__ == "__main__":
    main()