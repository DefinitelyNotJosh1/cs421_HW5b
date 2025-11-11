# Neural Network AI, HW 5b, CS-421
# Authors:
# - Trenton Pham
# - Joshua Krasnogorov

# The goal of this assignment is to build a neural network that copies the Utility function we have now


import random
import sys
import numpy as np
import pandas as pd
import os

sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *


# Define it as a class - will make the implementation in ReANTICS way easier if we do it this way
# Josh:Took inspiration from my own implementation of an ANN in ML class, but simplified it for this assignment
class ANN:
    def __init__(self, input_size, hidden_size, output_size, alpha, batch_size, stop_threshold, weights_and_biases_file):
        # Initialize weights and biases
        self.weights_and_biases_file = weights_and_biases_file
        if self.weights_and_biases_file:
            # if the file exists, great, load it up
            if os.path.exists(self.weights_and_biases_file):
                with open(weights_and_biases_file, "rb") as f:
                    self.w1 = np.load(f)["w1"]
                    self.b1 = np.load(f)["b1"]
                    self.w2 = np.load(f)["w2"]
                    self.b2 = np.load(f)["b2"]
            else:
                # if it doesn't exist, create new weights and biases and a new file
                print(f"Weights and biases file {self.weights_and_biases_file} does not exist, creating new weights and biases and a new file")
                self.w1 = np.random.rand(input_size, hidden_size) * 2 - 1 # -1 to 1
                self.b1 = np.random.rand(1, hidden_size) * 2 - 1
                self.w2 = np.random.rand(hidden_size, output_size) * 2 - 1 
                self.b2 = np.random.rand(1, output_size) * 2 - 1 
                # save weights to new file
                with open(self.weights_and_biases_file, "wb") as f:
                    np.savez(f, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
        else:
            # if no file is specified, create new weights and biases
            self.w1 = np.random.rand(input_size, hidden_size) * 2 - 1 # -1 to 1
            self.b1 = np.random.rand(1, hidden_size) * 2 - 1
            self.w2 = np.random.rand(hidden_size, output_size) * 2 - 1 
            self.b2 = np.random.rand(1, output_size) * 2 - 1 

        # other parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size 
        self.alpha = alpha
        self.batch_size = batch_size
        self.accuracy_per_epoch = []
        self.error_per_epoch = [1]
        self.stop_threshold = stop_threshold

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Sigmoid derivative
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    ##
    # forward
    #
    # Description: Propogates the input forward through the network.
    #
    # Parameters:
    #   input - the input to the network
    # 
    # Return: The output of the network after passing through the network.
    ##
    def forward(self, input):
        # Hidden layer
        self.z1 = np.dot(input, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Output layer
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    ##
    # backward
    #
    # Description: Propogates the error backward through the network.
    #
    # Parameters:
    #   x_input - the input to the network
    #   y_output - the desired output of the network
    #
    # Return: Nothing, updates weights and biases in itself.
    ##
    def backward(self, x_input, y_output):
        # I divide deltas by n to get the average gradient across the batch to make learning rate independent of batch size
        n = x_input.shape[0]
        
        # Output layer error
        # Note: 
        #   for this assignment, omitting the sigmoid derivative makes convergence ~10x faster.
        #   I'll keep it so that it's consistent with what we did in class. May remove for 5b.
        error_output = (y_output - self.a2) * self.sigmoid_derivative(self.a2)

        # Calculate weights and biases for output layer
        dw2 = np.dot(self.a1.T, error_output) / n
        db2 = np.sum(error_output, axis=0, keepdims=True) / n

        # Hidden layer error
        error_hidden = np.dot(error_output, self.w2.T) * self.sigmoid_derivative(self.a1)

        # Calculate weights and biases for hidden layer
        dw1 = np.dot(x_input.T, error_hidden) / n
        db1 = np.sum(error_hidden, axis=0, keepdims=True) / n

        # Update weights and biases
        self.w1 += self.alpha * dw1
        self.b1 += self.alpha * db1
        self.w2 += self.alpha * dw2
        self.b2 += self.alpha * db2

    # Train the model
    def train(self, x_input, y_output):
        epoch = 1
        # Ensure targets are a (N, 1) column vector to avoid unintended broadcasting
        if y_output.ndim == 1:
            y_output = y_output.reshape(-1, 1)
        while (self.error_per_epoch[-1] > self.stop_threshold):
            epoch += 1
            # Shuffle data
            perm = np.random.permutation(len(x_input))
            x_input_shuffled = x_input[perm]
            y_output_shuffled = y_output[perm]

            # Batch training
            batch_count = len(x_input_shuffled) // self.batch_size
            for i in range(batch_count):
                batch_input = x_input_shuffled[i*self.batch_size:(i+1)*self.batch_size]
                batch_output = y_output_shuffled[i*self.batch_size:(i+1)*self.batch_size]
                if batch_output.ndim == 1:
                    batch_output = batch_output.reshape(-1, 1)

                self.forward(batch_input)
                self.backward(batch_input, batch_output)
            
            output_pred = self.forward(x_input)

            # Calculate accuracy over the full dataset
            accuracy = np.mean(output_pred == y_output)
            self.accuracy_per_epoch.append(accuracy)
            # print(f"Epoch {epoch+1}, Accuracy: {accuracy:.4f}")

            # Calculate average error over dataset for this epoch
            error = np.mean(np.abs(output_pred - y_output))
            self.error_per_epoch.append(error)

            # print every 100 epochs; wayyyy too many prints if every epoch
            if epoch % 100 == 0: 
                print(f"Epoch {epoch}, Error: {error:.4f}, Accuracy: {accuracy:.4f}")
                # save weights and biases every 100 epochs
                if self.weights_and_biases_file:
                    with open(self.weights_and_biases_file, "wb") as f:
                        np.savez(f, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
                
                # trim accuracy and error lists to the last 100 epochs
                self.accuracy_per_epoch = self.accuracy_per_epoch[-100:]
                self.error_per_epoch = self.error_per_epoch[-100:]

                # lower learning rate as we get closer to the stop threshold
                if epoch % 1000 == 0:
                    self.alpha *= 0.95

            # if average errror is less than stop threshold, stop training
            if error < self.stop_threshold:
                print(f"Training stopped at epoch {epoch+1} at an error of {error:.4f} because average error is less than stop threshold of {self.stop_threshold:.4f}")


##
# NODE
# Description: A node in the search tree; contains a game state, a move, the parent state,
# and the utility of the state.
##
class Node:
    # Use slots for memory optimization and fast attribute access -
    # HOWEVER  - we can't add new attributes dynamically now. This shouldn't be a problem tho
    __slots__ = ['parent', 'move', 'gameState', 'depth', 'evaluation']

    ## __init__
    #
    # Description: Creates a new node
    #
    # Parameters:
    #   parent - the parent node
    #   move - the move that led to this state
    #   gameState - the game state
    #   depth - how many steps to reach from the agent's actual state
    #   evalution - state depth + utility
    ##
    def __init__(self, parent, move, gameState, depth, evaluation):
        self.parent = parent
        self.move = move
        self.gameState = gameState
        self.depth = depth
        self.evaluation = evaluation

##
# mappingFunction
#
# Description: Maps a game state to a vector of features
#
# Parameters:
#   gameState - a game state
#
# Return: A vector of features
##
def mappingFunction(gameState):
    # Helper utilities - these are here to avoid divide by zero errors
    def safe_ratio(numerator, denominator):
        try:
            return float(numerator) / float(denominator) if denominator else 0.0
        except Exception:
            return 0.0

    # Normalize a count to a value between 0 and 1
    def cap_norm(count, cap):
        if cap <= 0:
            return 0.0
        count = max(0, min(int(count), int(cap)))
        return float(count) / float(cap)

    # Normalize a distance to a value between 0 and 1
    def closeness(distance):
        max_dist = 10.0
        # 0 distance -> 1.0; distance >= max_dist -> 0.0
        try:
            return max(0.0, min(1.0, 1.0 - (float(distance) / float(max_dist))))
        except Exception:
            return 0.0

    # Average a list of values, or return 0 if the list is empty
    def avg_or_zero(values):
        return (sum(values) / float(len(values))) if values else 0.0

    # Entities and lists
    me = gameState.whoseTurn
    enemy = 1 - me
    myInv = getCurrPlayerInventory(gameState)
    enemyInv = getEnemyInv(enemy, gameState)

    myAnts = getAntList(gameState, me, (WORKER, DRONE, SOLDIER, R_SOLDIER, QUEEN))
    enemyAnts = getAntList(gameState, enemy, (WORKER, DRONE, SOLDIER, R_SOLDIER, QUEEN))
    myWorkers = getAntList(gameState, me, (WORKER,))
    myAttackers = getAntList(gameState, me, (DRONE, SOLDIER, R_SOLDIER))
    enemyAttackers = getAntList(gameState, enemy, (DRONE, SOLDIER, R_SOLDIER))

    foods = getConstrList(gameState, None, (FOOD,))
    myHill = myInv.getAnthill()
    enemyHill = enemyInv.getAnthill()
    myQueen = myInv.getQueen()
    enemyQueen = enemyInv.getQueen()

    dropSites = []
    if myHill is not None:
        dropSites.append(myHill.coords)
    tunnels = myInv.getTunnels()
    if tunnels:
        dropSites.extend([t.coords for t in tunnels])

    # Precompute worker-food and carrier-drop closeness values
    nonCarryingCloseness = []
    for w in myWorkers:
        if not getattr(w, "carrying", False):
            if foods:
                minDist = min(approxDist(w.coords, f.coords) for f in foods)
                nonCarryingCloseness.append(closeness(minDist))
            else:
                nonCarryingCloseness.append(0.0)

    carryingCloseness = []
    for w in myWorkers:
        if w.carrying:
            if dropSites:
                minDrop = min(approxDist(w.coords, d) for d in dropSites)
                carryingCloseness.append(closeness(minDrop))
            else:
                carryingCloseness.append(0.0)

    # Threats/defense geometry
    def on_my_side(coords):
        return coords[1] <= 4

    def on_enemy_side(coords):
        return coords[1] > 4

    threats = [a for a in getAntList(gameState, enemy, (QUEEN, WORKER, DRONE, SOLDIER, R_SOLDIER)) if on_my_side(a.coords)]
    defenders = list(myAttackers)

    defenderToThreatProximity = 0.0
    if threats and defenders:
        perThreat = []
        for t in threats:
            dMin = min(approxDist(d.coords, t.coords) for d in defenders)
            perThreat.append(closeness(dMin))
        defenderToThreatProximity = avg_or_zero(perThreat)

    # Enemy attackers closeness to my queen/hill
    enemyToMyQueen = 0.0
    if enemyAttackers and (myQueen is not None):
        dMin = min(approxDist(a.coords, myQueen.coords) for a in enemyAttackers)
        enemyToMyQueen = closeness(dMin)

    enemyToMyHill = 0.0
    if enemyAttackers and (myHill is not None):
        dMin = min(approxDist(a.coords, myHill.coords) for a in enemyAttackers)
        enemyToMyHill = closeness(dMin)

    # My attackers closeness to enemy queen/hill
    myAtkToEnemyQueen = 0.0
    if myAttackers and (enemyQueen is not None):
        vals = [closeness(approxDist(a.coords, enemyQueen.coords)) for a in myAttackers]
        myAtkToEnemyQueen = avg_or_zero(vals)

    myAtkToEnemyHill = 0.0
    if myAttackers and (enemyHill is not None):
        vals = [closeness(approxDist(a.coords, enemyHill.coords)) for a in myAttackers]
        myAtkToEnemyHill = avg_or_zero(vals)

    # Attackers on enemy side
    myAttackersOnEnemySide = safe_ratio(len([a for a in myAttackers if on_enemy_side(a.coords)]), max(1, len(myAttackers)))

    # Food near workers
    foodsNearWorkers = 0.0
    if foods:
        if myWorkers:
            nearCount = 0
            for f in foods:
                minWorkerDist = min(approxDist(w.coords, f.coords) for w in myWorkers) if myWorkers else 999
                if minWorkerDist <= 2:
                    nearCount += 1
            foodsNearWorkers = safe_ratio(nearCount, len(foods))
        else:
            foodsNearWorkers = 0.0

    # Workers carrying fraction
    carryingWorkers = len([w for w in myWorkers if getattr(w, "carrying", False)])
    workersCarryingFrac = safe_ratio(carryingWorkers, len(myWorkers))

    # Build feature vector (24 total, all in [0,1])
    features = []
    # 1-3: food levels and delta
    features.append(safe_ratio(myInv.foodCount, 11))
    features.append(safe_ratio(enemyInv.foodCount, 11))
    features.append((float(myInv.foodCount - enemyInv.foodCount) + 11.0) / 22.0)
    # 4-5: hill capture health normalized
    features.append(safe_ratio(myHill.captureHealth if myHill is not None else 0, 3))
    features.append(safe_ratio(enemyHill.captureHealth if enemyHill is not None else 0, 3))
    # 6-7: capped ant counts
    features.append(cap_norm(len(myAnts), 20))
    features.append(cap_norm(len(enemyAnts), 20))
    # 8-10: composition shares
    features.append(safe_ratio(len(myWorkers), len(myAnts)))
    features.append(safe_ratio(len(myAttackers), len(myAnts)))
    features.append(safe_ratio(len(enemyAttackers), len(enemyAnts)))
    # 11-14: worker/food and carrier/drop closeness (avg and best)
    features.append(avg_or_zero(nonCarryingCloseness))
    features.append(avg_or_zero(carryingCloseness))
    features.append(max(nonCarryingCloseness) if nonCarryingCloseness else 0.0)
    features.append(max(carryingCloseness) if carryingCloseness else 0.0)
    # 15-16: threats on my side and defender proximity
    features.append(safe_ratio(len(threats), len(enemyAnts)))
    features.append(defenderToThreatProximity)
    # 17-18: enemy attackers proximities to my queen/hill
    features.append(enemyToMyQueen)
    features.append(enemyToMyHill)
    # 19-20: my attackers proximities to enemy queen/hill
    features.append(myAtkToEnemyQueen)
    features.append(myAtkToEnemyHill)
    # 21: my attackers positioned on enemy side
    features.append(myAttackersOnEnemySide)
    # 22: worker count target (normalized to 5)
    features.append(cap_norm(len(myWorkers), 5))
    # 23: fraction of foods that are near any worker (<=2)
    features.append(foodsNearWorkers)
    # 24: fraction of workers that are carrying
    features.append(workersCarryingFrac)

    return features

##
# expandNode
#
# Description: Expands a node to include all valid moves from the GameState in the given node
#
# Parameters:
#   node - a node
#
# Return: A list of all the new nodes that were created.
##
def expandNode(node):
    moves = listAllLegalMoves(node.gameState)
    nodes = []
    for move in moves:
        newNode = Node(node, move, getNextState(node.gameState, move), node.depth + 1, None)
        nodes.append(newNode)
    return nodes



##
# utility
#
# Description: Calculates the utility of a given game state on a scale of 0 to 1
# Reminder: Do not use the board variable
#
# Parameters:
#   gameState - a game state
#
# Return: The utility of the state
#
##
def utility(gameState):
        # Some ideas: from Josh:
        # Food difference - this should absolutely play a decently large role.
        # enemy ants - if the enemy has lots of ants and we don't, that's bad.

        # Constants
        me = gameState.whoseTurn
        enemy = 1 - me
        myInv = getCurrPlayerInventory(gameState)
        enemyInv = getEnemyInv(enemy, gameState)
        myAnts = getAntList(gameState, me, (WORKER,DRONE,SOLDIER,R_SOLDIER,QUEEN))
        utility = 0.0
        # If I win in this game state; instant win lose
        if gameState.phase == PLAY_PHASE:   #v libby trick
            if getWinner(gameState) == me or \
                    len(getAntList(gameState, enemy, (QUEEN,))) == 0 or \
                    myInv.foodCount == 11 or \
                    enemyInv.getAnthill().captureHealth == 0:
                return 0.0  # cost 2 win?
            elif getWinner(gameState) == enemy or \
                    len(myAnts) == 0 or \
                    enemyInv.foodCount == 11 or \
                    myInv.getAnthill().captureHealth == 0:
                return float(100.0) # float('inf') # cost 2 lose? / best thing ever

        # estimate moves for queen, food, capture hill,... maybe soldiers?


        # food stuff - 60% of total utility
        foodScore = foodUtility(gameState, myInv, enemyInv, me)
        # print(f"Food Score: {foodScore}")
        if foodScore:
            utility += foodScore * 0.6

        # defense stuff - 30% of total utility
        defenseScore = defenseUtility(gameState, me)
        # attack stuff - 10% of total utility
        attackScore = attackUtility(gameState, myInv, enemyInv, me)

        if defenseScore > attackScore:
            utility += defenseScore * 0.4
        else:
            utility += attackScore * 0.4

        # print(f"Utility: {utility}")



        utility = (1.0 - utility) #* 8.7
        return utility


## foodUtility
# Description: Calculates the utility of the food situation in a game state
# Includes worker utility
#
# Parameters:
#   gameState - a game state
#   myInv - the inventory of the current player
#   enemyInv - the inventory of the enemy
#   me - the id of the current player
#
# Return: The utility of the food situation
##
def foodUtility(gameState, myInv, enemyInv, me):
    utility = 0.0
    # Food Weights - 90% of total utility
    if myInv.foodCount is not None and enemyInv.foodCount is not None:
        foodScore = 0.5
        foodScore += (myInv.foodCount / 11) * 0.5 # This is on a scale of 0 - 1 - good, now multiply by multiplier
        foodScore -= (enemyInv.foodCount / 11) * 0.5
        # print(f"Food Score: {foodScore}")
        utility += foodScore * 0.97


        # Some help from ChatGPT
        workerScore = 0.0
        # Get my workers
        myWorkers = getAntList(gameState, me, (WORKER,))
        tunnels = myInv.getTunnels()
        anthill = myInv.getAnthill()
        foodList = getConstrList(gameState, None, (FOOD,))
        numWorkers = len(myWorkers)
        # print(f"Workers: {myWorkers}")
        # print(f"Num Workers: {numWorkers}")

        # If we have no workers, score is 0
        if numWorkers == 0:
            return 0.0

        # If we have too many workers, aka not good
        if numWorkers > 2:
            utility -= 0.1

        # Avoid division by zero; if no workers, score remains 0
        if numWorkers > 0:
            # Precompute drop sites
            dropSites = []
            if anthill:
                dropSites.append(anthill.coords)
            if tunnels:
                dropSites.extend([t.coords for t in tunnels])

            # Normalization constants keep per-worker contribution in [0,1]
            maxFoodDist = 8.0
            maxDropDist = 8.0

            for i, w in enumerate(myWorkers):
                contrib = 0.0

                if w.carrying:
                    # If at drop site: full contribution
                    if dropSites and any(w.coords == d for d in dropSites):
                        contrib = 1.0
                    else:
                        # Positive baseline for carrying so picking up is attractive
                        if dropSites:
                            closestDrop = min(approxDist(w.coords, d) for d in dropSites)
                            progressToDrop = max(0.0, min(1.0, 1.0 - (closestDrop / maxDropDist)))
                        else:
                            progressToDrop = 0.0
                        # Baseline 0.5 plus progress up to 1.0 max
                        contrib = 0.5 + 0.5 * progressToDrop
                else:
                    # Not carrying: incentivize getting closer to nearest food, but cap at 0.5
                    if foodList:
                        closestFood = min(approxDist(w.coords, f.coords) for f in foodList)
                        towardFood = max(0.0, min(1.0, 1.0 - (closestFood / maxFoodDist)))
                        contrib = 0.5 * towardFood
                    else:
                        contrib = 0.0

                # Clamp and average across workers
                contrib = max(0.0, min(1.0, contrib))
                workerScore += contrib / numWorkers
                # print(f"Worker {i} contrib: {contrib}")

        # print(f"Worker Score: {workerScore}")
        # Ensure workerScore in [0,1]
        workerScore = max(0.0, min(1.0, workerScore))
        utility += (workerScore * 0.03)
        utility = min(utility, 1.0)
        return utility


## defenseUtility
# Description: Calculates the utility of the defense situation in a game state
#
# Parameters:
#   gameState - a game state
#   myInv - the inventory of the current player
#   enemyInv - the inventory of the enemy
#   me - the id of the current player
#
# Return: The utility of the attack situation
##
def defenseUtility(gameState, me):
    enemy = 1 - me
    def on_my_side(coords):
        y = coords[1]
        return (y <= 4)

    # Enemy ants on my side are threats
    threats = [a for a in getAntList(gameState, enemy, (QUEEN, WORKER, DRONE, SOLDIER, R_SOLDIER)) if on_my_side(a.coords)]
    # My attack-capable ants
    defenders = getAntList(gameState, me, (DRONE, SOLDIER, R_SOLDIER))

    # If no threats on my side, defense is perfect
    if not threats:
        return 1.0
    # If there are threats but no defenders, defense is bad
    if not defenders:
        return 0.0

    # Encourage defenders to be close to threats
    # 0 distance -> 1.0 score; distance >= maxDist -> 0.1 score
    maxDist = 10.0
    total = 0.0
    for t in threats:
        minDist = min(approxDist(d.coords, t.coords) for d in defenders)
        score = 1.0 - min(minDist / maxDist, 10.0)
        total += score

    proximityScore = total / len(threats)
    return max(0.0, min(1.0, proximityScore))


## attackUtility
# Description: Calculates the utility of the defense situation in a game state
#
# Parameters:
#   gameState - a game state
#   myInv - the inventory of the current player
#   enemyInv - the inventory of the enemy
#   me - the id of the current player
#
# Return: The utility of the attack situation
##
def attackUtility(gameState, myInv, enemyInv, me):
    enemy = 1 - me
    maxDist = 10.0
    total = 0.0
    def on_enemy_side(coords):
        y = coords[1]
        return y > 4

    # Enemy ants on the enemy side
    attackable = [a for a in getAntList(gameState, enemy, (QUEEN, WORKER, DRONE, SOLDIER, R_SOLDIER)) if on_enemy_side(a.coords)]
    # My attack-capable ants
    attackers = getAntList(gameState, me, (DRONE, SOLDIER, R_SOLDIER))

    enemyAnthill = enemyInv.getAnthill()
    if enemyAnthill:
        enemyAnthill = enemyAnthill.coords
    else:
        return 1.0

    enemyQueen = getAntList(gameState, enemy, (QUEEN,))[0]
    if enemyQueen:
        enemyQueen = enemyQueen.coords
    else:
        return 1.0

    # If there are no enemy's, attack is perfect
    if not attackable:
        for t in attackers:
            # minDist = approxDist(enemyAnthill, t.coords) # min(approxDist(d.coords, t.coords) for d in attackers)
            minDist = approxDist(enemyQueen, t.coords)
            score = 1.0 - min(minDist / maxDist, 10.0)
            total += score

        proximityScore = total / len(attackers) if total != 0 or len(attackers) != 0 else 0.0
        return max(0.0, min(1.0, proximityScore))
    # If there are threats but no attackers, attackable is bad
    if not attackers:
        return 0.0

    # Encourage attackers to be close to threats
    # 0 distance -> 1.0 score; distance >= maxDist -> 0.1 score
    for t in attackable:
        minDist = approxDist(enemyAnthill, t.coords) # min(approxDist(d.coords, t.coords) for d in attackers)
        # minDist = approxDist(enemyQueen, t.coords)
        score = 1.0 - min(minDist / maxDist, 10.0)
        total += score

    proximityScore = total / len(attackable) if total != 0 or len(attackable) != 0 else 0.0
    return max(0.0, min(1.0, proximityScore))


##
# bestMove
#
# Description: Searches a given list of game nodes to find the highest utility move
#
# Parameters:
#   gameState - a game state
#   moves - a list of moves
#
# Return: The state with the highest utility
#
def bestMove(nodes, ann):
    # Initialize the best node with the first node's utility
    bestNodes = [nodes[0]]

    # Iterate through nodes to find the one with the highest utility
    for node in nodes:
        if node.evaluation is None:
            mapping = mappingFunction(node.gameState)
            node.evaluation = ann.forward(mapping)[0] + node.depth
        if (node.evaluation - node.depth < bestNodes[0].evaluation - bestNodes[0].depth):
            bestNodes = [node]
        elif (node.evaluation - node.depth == bestNodes[0].evaluation - bestNodes[0].depth):
            bestNodes.append(node)

    return random.choice(bestNodes)


##
#AIPlayer
#Description: The responsibility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "Neural Network")
        self.playerId = inputPlayerId
        self.ann = ANN(24, 240, 1, 0.01, 400, 0.0001, "weights_and_biases_240.npy")


    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            enemyTunnel = getConstrList(currentState, None, (TUNNEL,))[0]
            enemyHill = getConstrList(currentState, None, (ANTHILL,))[0]

            # find all spots on enemy side of board that are empty
            furthestCoords = []
            for i in range(0, 10):
                for j in range(6, 10):
                    if currentState.board[i][j].constr == None:
                        furthestCoords.append((i,j))

            # sort spots by distance from enemy tunnel
            furthestCoords.sort(key=lambda x:
                        abs(enemyTunnel.coords[0] - x[0]) + abs(enemyTunnel.coords[1] - x[1]) +
                        abs(enemyHill.coords[0] - x[0]) + abs(enemyHill.coords[1] - x[1]))
            moves = []
            # add the two furthest spots to the moves list
            moves.append(furthestCoords[-1])
            moves.append(furthestCoords[-2])
            return moves
        else:
            return [(0, 0)]

    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##

    def getMove(self, currentState):
        frontierNodes = []
        expandedNodes = []
        rootNode = Node(None, None, currentState, 0, None)
        frontierNodes.append(rootNode)

        for i in range(3): # 3 is the depth of the search
            bestNode = bestMove(frontierNodes, self.ann)
            frontierNodes.remove(bestNode)
            expandedNodes.append(bestNode)
            newNodes = expandNode(bestNode)
            frontierNodes.extend(newNodes)

        bestNode = bestMove(frontierNodes, self.ann)
        while bestNode.depth > 1:
            bestNode = bestNode.parent

        # append the mapping to the file
        # with open("mapping.csv", "a") as f:
        #     for _, m in enumerate(mapping):
        #         f.write(f"{m},")
        #     f.write(f"{utility}\n")
        return bestNode.move


    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocations - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):
        #method templaste, not implemented
        pass


# Remove print statements for final version
# print("-------------------------------- STARTING TESTS -------------------------------- ")
totalTests = 4
passedTests = 0
# BEST MOVE TEST
# print("| Beginning bestMove test")
# nodes = []
# for i in range(10):
#     node = Node(None, None, GameState.getBlankState(), 1, None)
#     nodes.append(node)
#     node.evaluation = i / 10 + node.depth
# bestNode = bestMove(nodes, ann)

# if bestNode.evaluation == 1.0:
#     # print(f"| BestMove test passed. Value was {bestNode.evaluation}, expected 1.9")
#     passedTests += 1
# else:
#     print(f"| BestMove test failed. Value was {bestNode.evaluation}, expected 1.9")


# UTILITY TEST
# print("| Beginning utility test")
gameState = GameState.getBlankState()
util = utility(gameState)
if not 0.0 <= util <= 1.0:
    print(f"| ERROR: utility() returned {util}, expected 0.4")
else:
    # print(f"| Utility test passed. Value was {util}, expected 0.4")
    passedTests += 1


# FOOD UTILITY TEST
# print("| Beginning food utility test")
gameState = GameState.getBasicState()
util = foodUtility(gameState, getCurrPlayerInventory(gameState), getEnemyInv(0, gameState), 0)
if not 0.0 <= util <= 1.0:
    print(f"| ERROR: foodUtility() returned {util}, expected 0.0")
else:
    # print(f"| Food utility test passed. Value was {util}, expected 0.0")
    passedTests += 1


# DEFENSE UTILITY TEST
# print("| Beginning defense utility test")
gameState = GameState.getBasicState()
util = defenseUtility(gameState, 0)
if not 0.0 <= util <= 1.0:
    print(f"| ERROR: defenseUtility() returned {util}, expected 1.0")
else:
    # print(f"| Defense utility test passed. Value was {util}, expected 1.0")
    passedTests += 1

# print(f"|----------------------- Passed {passedTests} out of {totalTests} tests --------------------------- ")
# print("-------------------------------- ENDING TESTS -------------------------------- ")