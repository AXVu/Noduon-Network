import pandas as pd
import random as rng
# testing only #
import matplotlib.pyplot as plt

'''
A NoduonNetwork is comprised of three different noduon structures

Noduon, which are the baseline input Noduons
InnerNoduon, which are calculator noduons
OutputNoduon, which is an InnerNoduon with a title

The noduon API includes the following

All Noduons:
weightedSum -> Sums the weighted sum of each of the Noduons in pulls from. Inp neurons return their value.

Input noduon:
self.title -> The title of this input, such as: distanceToWall or timeSinceJump
self.value -> The value of the input, distanceToWall could have a self.value of 3.5
updateValue -> Changes the value of this noduon's input to the desired quantity

Operator noduons (inner and output):
self.noduons -> A list containing all of the noduons that are part of this noduon's weighted sum
self.weights -> A list equal in length to self.noduons, saving the weight for each one
determine -> Runs weighted sum and outputs true if positive and false if negative
changeWeights -> Accepts a weight change vector, adding each value to the proper weight
setWeights -> Changes weights to the input vector
getWeights -> Returns list of weights

Output noduon:
self.title -> The title of the output usually an action or predictor, such as: jump, accelerate, expected y value

##########
Noduon Network
##########
Constructor (self, inputs, inners, outputs, outputNames, randWeights)
### Parameters ###
inputs -> A list of titles to form into input Noduons
inners -> A list with a number of entries equal to the number of inner Noduon columns/ranks. Each element contains a list
of length equal to the number of Noduons in that rank. Each element of the list represents a noduon, which has a length
equal to the number of previous inner ranks + 1 for the input noduons. Each element of the list is the index of a Noduon that
to link to.
outputs -> Has the same format as a single rank of inners
outputNames -> A list of names equal to the length of outputs, setting a name for each output.
randWeights -> A boolean determining whether or not to randomize the weights of operator Noduons
### Variables ###
self.buildingBlocks = saves the input lists in order to reconstruct self.
self.inputs -> A dictionary of input noduons
self.inner -> A list of ranks/columns of noduons, each containing a list of inner noduons
self.outputs -> A dictionary of output noduons
self.innerRank -> Number of inner ranks/columns
self.mutationFactor
### Methods ###
# Gets #
getOutput -> Takes the name of an input as an argument, returns that Noduon's weightedSum
getOutputs -> Returns a dictionary of each output Noduon's title:weightedSum
getWeights -> Takes a rank and index as input and returns a list equal to the weights of the selected noduon
getInputValue -> Takes an input key and returns its current value
getInputTitles -> Return a list of input titles/keys
getOutputTitles -> Return a list of output/keys
# Sets #
setInputValue -> Takes a value and sets in to the given input key
setInputValues -> Takes a dictionary of input keys and new values, setting them to the new values
changeNoduonWeight -> Takes a rank and index of a operator Noduon, changes it by the input weight vector
setNoduonWeight -> Takes a rank and index of an operator Noduon, sets it to the input weight vector
# Copy and Mutations #
copyNet -> Returns a copy of the NoduonNet. Does not copy current input values
copyStructure -> Returns a copy of the NoduonNet structure
mutate -> Alters a random based on mutationRate

'''

# Create the Noduon class
class InpNoduon :

    # Constructor method
    def __init__(self, title, val):
        self.value = val
        self.title = title

    ### Methods ###

    def weightedSum(self):
        return self.value
    
    def updateValue(self, value):
        self.value = value


class InnerNoduon:

    # Constructor method
    # Inputs will be a list of nodons or inner neurons
    def __init__(self, inpNoduons, rand):
        self.noduons = inpNoduons
        if rand:
            self.weights = [rng.random() * rng.choice([-1,1]) for selfRank in range(len(self.noduons))]
        else:
            self.weights = [0 for selfRank in range(len(self.noduons))]
    
    ### Methods ###

    # Inp: N/A. Out: Weighted sum of inputs and weights
    def weightedSum(self):
        total = 0
        for noduon, weight in zip(self.noduons, self.weights):
            total += noduon.weightedSum() * weight
        return total
    
    # determine will perform a weighted sum, returning true on a positive result and false on negative
    def determine(self):
        summed = self.weightedSum()
        if summed > 0:
            return True
        return False

    def changeWeights(self, weightVector):
        for selfRank in range(len(self.noduons)):
            self.weights[selfRank] += weightVector[selfRank]

    def getWeights(self):
        return self.weights
    
    def setWeights(self, newWeights):
        if len(newWeights) == len(self.weights):
            for weight in range(len(self.weights)):
                self.weights[weight] = newWeights[weight]


class outputNoduon:

    # Constructor method
    def __init__(self, title, inps, rand):
        self.title = title
        self.noduons = inps
        if rand:
            self.weights = [rng.random() * rng.choice([-1, 1]) for selfRank in range(len(self.noduons))]
        else:
            self.weights = [0 for selfRank in range(len(self.noduons))]

    ### Methods ###

    def weightedSum(self):
        summedTotal = 0
        for noduon, weight in zip(self.noduons, self.weights):
            summedTotal += noduon.weightedSum() * weight
        return summedTotal

    def determine(self, parameters):
        summed = self.weightedSum(parameters)
        if summed > 0:
            return True
        return False
        
    def getWeights(self):
        return self.weights
    
    def changeWeights(self, weightVector):
        for selfRank in range(len(self.noduons)):
            self.weights[selfRank] += weightVector[selfRank]

    def setWeights(self, newWeights):
        if len(newWeights) == len(self.weights):
            for weight in range(len(self.weights)):
                self.weights[weight] = newWeights[weight]
    

class NoduonNetwork:

    """
    NoduonNetwork is a combination of the different noduons
    It saves them in sets of inputs, inner, and outputs.
    Inputs and and outputs can be referenced through their dictionary titles.
    Inner are saved as a list of lists for each rank.
    """

    # Constructor Method
    def __init__(self, inputs, inners, outputs, outNames, randWeights, mutationRate=0):
        self.buildingBlocks = [inputs,inners,outputs,outNames,mutationRate]
        self.inputs = {title:InpNoduon(title, 0) for title in inputs}
        self.inner = []
        self.outputs = {}
        self.mutationRate = mutationRate
        for selfRank in range(len(inners)):
            self.inner.append([])

            for numNoduon in range(len(inners[selfRank])):
                nodInps = []
                for linkRank in range(len(inners[selfRank][numNoduon])):
                    for nod in inners[selfRank][numNoduon][linkRank]:
                        if linkRank == 0:
                            nodInps.append(self.inputs[nod])
                        else:
                            nodInps.append(self.inner[linkRank-1][nod])
                self.inner[selfRank].append(InnerNoduon(nodInps, randWeights))

        for numNoduon in range(len(outputs)):
            nodInps = []
            for linkRank in range(len(outputs[numNoduon])):
                for nod in outputs[numNoduon][linkRank]:
                    if linkRank == 0:
                        nodInps.append(self.inputs[nod])
                    else:
                        nodInps.append(self.inner[linkRank-1][nod])
            self.outputs[outNames[numNoduon]] = outputNoduon(outNames[numNoduon], nodInps, randWeights)

        self.innerRank = len(self.inner)

    ### get Methods ###

    def getOutput(self, key):
        if key in self.outputs.keys():
            return self.outputs[key].weightedSum()
        else:
            print("No such output noduon")
        
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
            print("No such input noduon")

    def getInputTitles(self):
        return list(self.inputs.keys())

    def getOutputTitles(self):
        return list(self.outputs.keys())

    ### set/change Methods ###

    def setInputValue(self, key, value):
        if key in self.inputs.keys():
            self.inputs[key].updateValue(value)
        else:
            print("No such input noduon")

    def setInputValues(self, valuesDict):
        for key in list(valuesDict.keys()):
            if not key in self.inputs:
                print("Invalid key detected, cancelling input changes")
                return 0
        for key in list(valuesDict.keys()):
            self.inputs[key].updateValue(valuesDict[key])

    def changeNoduonWeight(self, rank, num, changeVector):
        if rank == len(self.inner):
            self.outputs[num].changeWeights(changeVector)
        elif rank in list(range(len(self.inner))):
            self.inner[rank][num].changeWeights(changeVector)
        else:
            print("Invalid weight change call, no such rank")

    def setNoduonWeight(self, rank, num, changeVector):
        if rank == len(self.inner):
            self.outputs[num].setWeights(changeVector)
        elif rank in list(range(len(self.inner))):
            self.inner[rank][num].setWeights(changeVector)
        else:
            print("Invalid weight set call, no such rank")

    ### Copying and Mutation methods ###

    def copyNet(self, mutate=False):
        copy = NoduonNetwork(self.buildingBlocks[0], self.buildingBlocks[1], self.buildingBlocks[2], self.buildingBlocks[3], False, self.buildingBlocks[4])
        for rank in range(len(copy.inner)):
            for noduon in range(len(copy.inner[rank])):
                copy.setNoduonWeight(rank, noduon, self.getWeights(rank, noduon))
        for output in list(copy.outputs.keys()):
            copy.setNoduonWeight(len(copy.inner), output, self.getWeights(len(self.inner), output))
        if mutate: copy.mutate()
        return copy
    
    def copyStructure(self, randWeights = True):
        copy = NoduonNetwork(self.buildingBlocks[0], self.buildingBlocks[1], self.buildingBlocks[2], self.buildingBlocks[3], randWeights, self.buildingBlocks[4])
        return copy

    def mutate(self):
        for mutation in range(int(self.mutationRate)):
            targetRank = rng.randint(0, self.innerRank)
            if targetRank == self.innerRank:
                targetNoduon = rng.choice(self.getOutputTitles())
            else:
                targetNoduon = rng.randint(0, len(self.inner[targetRank]) - 1)
            mutate = [0 for i in range(len(self.getWeights(targetRank, targetNoduon)))]
            mutate[rng.randint(0,len(mutate)-1)] = rng.choice([-1,1]) * rng.random() / 10
            self.changeNoduonWeight(targetRank, targetNoduon, mutate)
        self.mutationRate += rng.choice([-1, 1]) * rng.random()
        if self.mutationRate < 0: self.mutationRate = 0


def main():

    # Test methods

    net = NoduonNetwork(["Emily blinks", "6", "happy"],
                         [[[["Emily blinks", "6"]],[["6", "happy"]]],
                          [[[],[0,1]]]], [[[],[0,1],[0]]], 
                          ["nod"],
                            True, mutationRate=4)
    out = "nod"
    print(net.getInputTitles())
    print(net.getOutputTitles)
    print(net.getWeights(2,"nod"))
    print(f"Weighted sum: {net.getOutput(out)}")
    net.setInputValue("happy", 5)
    net.setInputValue("Emily blinks", -2)
    net.setInputValue("6", 6)
    print(f"Weighted sum: {net.getOutput(out)}")

    # First Generation #
    bestInGen = []
    avgInGen = []
    print("Generation 0")
    generation = [net.copyStructure(randWeights=True) for i in range(500)]
    results = []
    for agent in generation:
        agent.setInputValues({"Emily blinks":rng.random()*rng.choice([-1,1]) * 5,"happy":rng.random()*rng.choice([-1,1]) * 5, "6":rng.random()*rng.choice([-1,1]) * 5})
        summed = agent.getInputValue("Emily blinks") + agent.getInputValue("happy") + agent.getInputValue("6")
        results.append(abs(agent.getOutput(out) - summed))
    print(min(results))
    bestInGen.append(min(results))
    avgInGen.append(sum(results) / len(results))
    print(generation[results.index(min(results))].getOutput(out))
    print([generation[results.index(min(results))].getInputValue(val) for val in net.getInputTitles()])
    nextGen = []
    for i in range(len(generation) // 20):
        target = results.index(min(results))
        nextGen.append(generation[target].copyNet(mutate=False))
        for j in range(len(generation) // 20 - i):
            nextGen.append(generation[target].copyNet(mutate=True))
        results[target] = max(results) + 1
    for i in range(len(generation) - len(nextGen)):
        nextGen.append(net.copyStructure(True))
    generation = nextGen

    for i in range(214309):
        results = []
        for agent in generation:
            agent.setInputValues({"Emily blinks":rng.random()*rng.choice([-1,1]) * 5,"happy":rng.random()*rng.choice([-1,1]) * 5, "6":rng.random()*rng.choice([-1,1]) * 5})
            summed = agent.getInputValue("Emily blinks") + agent.getInputValue("happy") + agent.getInputValue("6")
            results.append(abs(agent.getOutput(out) - summed))
        bestInGen.append(min(results))
        avgInGen.append(sum(results) / len(results))

        if (i+1) % 100 == 0: print(f"Generation {i+1}")
        nextGen = []
        for i in range(len(generation) // 20):
            target = results.index(min(results))
            nextGen.append(generation[target].copyNet(mutate=False))
            for j in range(len(generation) // 20 - i):
                nextGen.append(generation[target].copyNet(mutate=True))
            results[target] = max(results) + 1
        for i in range(len(generation) - len(nextGen)):
            nextGen.append(net.copyStructure(True))
        generation = nextGen
    plt.plot(bestInGen)
    plt.yscale("log")
    plt.show()
    plt.plot(avgInGen)
    plt.yscale("log")
    plt.show()
    


if __name__ == "__main__":
    main()