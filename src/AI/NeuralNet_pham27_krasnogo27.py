# Neural Network AI (Food-Focused), HW 5b, CS-421
# Authors:
# - Trenton Pham
# - Joshua Krasnogorov

# The goal of this assignment is to build a neural network that copies the utility function
# Currently, the utility function is not used to assign utility


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
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, alpha, batch_size, stop_threshold, weights_and_biases_file, use_pretrained=False):
        # Initialize weights and biases
        self.weights_and_biases_file = weights_and_biases_file
        
        # I'm loading backweights from the bottom of the file
        if use_pretrained:
            self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = get_pretrained_weights()
        elif self.weights_and_biases_file:
            # if the file exists, great, load it up
            if os.path.exists(self.weights_and_biases_file):
                with open(weights_and_biases_file, "rb") as f:
                    data = np.load(f)
                    self.w1 = data["w1"]
                    self.b1 = data["b1"]
                    self.w2 = data["w2"]
                    self.b2 = data["b2"]
                    self.w3 = data["w3"]
                    self.b3 = data["b3"]
            else:
                # if it doesn't exist, create new weights and biases and a new file
                print(f"Weights and biases file {self.weights_and_biases_file} does not exist, creating new weights and biases and a new file")
                self.w1 = np.random.rand(input_size, hidden_size1) * 2 - 1 # -1 to 1
                self.b1 = np.random.rand(1, hidden_size1) * 2 - 1
                self.w2 = np.random.rand(hidden_size1, hidden_size2) * 2 - 1
                self.b2 = np.random.rand(1, hidden_size2) * 2 - 1
                self.w3 = np.random.rand(hidden_size2, output_size) * 2 - 1
                self.b3 = np.random.rand(1, output_size) * 2 - 1
                # save weights to new file
                with open(self.weights_and_biases_file, "wb") as f:
                    np.savez(f, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, w3=self.w3, b3=self.b3)
        else:
            # if no file is specified, create new weights and biases
            self.w1 = np.random.rand(input_size, hidden_size1) * 2 - 1 # -1 to 1
            self.b1 = np.random.rand(1, hidden_size1) * 2 - 1
            self.w2 = np.random.rand(hidden_size1, hidden_size2) * 2 - 1
            self.b2 = np.random.rand(1, hidden_size2) * 2 - 1
            self.w3 = np.random.rand(hidden_size2, output_size) * 2 - 1
            self.b3 = np.random.rand(1, output_size) * 2 - 1

        # other parameters
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
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

        # Hidden layer 2
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        # Output layer
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = self.sigmoid(self.z3)

        return self.a3

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
        error_output = (y_output - self.a3) * self.sigmoid_derivative(self.a3)

        # Calculate weights and biases for output layer
        dw3 = np.dot(self.a2.T, error_output) / n
        db3 = np.sum(error_output, axis=0, keepdims=True) / n

        # Hidden layer 2 error
        error_hidden2 = np.dot(error_output, self.w3.T) * self.sigmoid_derivative(self.a2)

        # Calculate weights and biases for hidden layer 2
        dw2 = np.dot(self.a1.T, error_hidden2) / n
        db2 = np.sum(error_hidden2, axis=0, keepdims=True) / n

        # Hidden layer 1 error
        error_hidden1 = np.dot(error_hidden2, self.w2.T) * self.sigmoid_derivative(self.a1)

        # Calculate weights and biases for hidden layer 1
        dw1 = np.dot(x_input.T, error_hidden1) / n
        db1 = np.sum(error_hidden1, axis=0, keepdims=True) / n

        # Update weights and biases
        self.w1 += self.alpha * dw1
        self.b1 += self.alpha * db1
        self.w2 += self.alpha * dw2
        self.b2 += self.alpha * db2
        self.w3 += self.alpha * dw3
        self.b3 += self.alpha * db3

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
                        np.savez(f, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2, w3=self.w3, b3=self.b3)
                
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
# Return: A vector of features (10 features)
##
def mappingFunction(gameState):
    
    # Helper function
    def closeness(distance, max_dist=10.0):
        try:
            return max(0.0, min(1.0, 1.0 - (float(distance) / float(max_dist))))
        except Exception:
            return 0.0
    
    # Get basic info
    me = gameState.whoseTurn
    enemy = 1 - me
    myInv = getCurrPlayerInventory(gameState)
    enemyInv = getEnemyInv(enemy, gameState)
    workers = getAntList(gameState, me, (WORKER,))
    foods = getConstrList(gameState, None, (FOOD,))
    
    # Feature vector (10 features)
    features = []
    
    # FEATURE 1: My food count (normalized) - MOST IMPORTANT (90% weight in utility)
    features.append(float(myInv.foodCount) / float(FOOD_GOAL))
    
    # FEATURE 2: Number of workers carrying food (normalized)
    carryingCount = sum(1 for w in workers if getattr(w, "carrying", False))
    features.append(float(carryingCount) / 2.0)  # Normalize by reasonable max (2 workers)
    
    # FEATURE 3: Total number of workers (for worker count bonus)
    features.append(float(len(workers)) / 3.0)  # Normalize by reasonable max (3)
    
    # FEATURE 4: Are we in danger of having 0 workers? (binary)
    features.append(1.0 if len(workers) == 0 else 0.0)
    
    # Get drop sites
    dropSites = []
    if myInv.getAnthill() is not None:
        dropSites.append(myInv.getAnthill().coords)
    tunnels = myInv.getTunnels()
    if tunnels:
        dropSites.extend([t.coords for t in tunnels])
    
    # FEATURE 5: Average proximity of carrying workers to drop sites
    carryingProximity = []
    for w in workers:
        if getattr(w, "carrying", False) and dropSites:
            dmin = min(approxDist(w.coords, d) for d in dropSites)
            carryingProximity.append(closeness(dmin))
    features.append(sum(carryingProximity)/len(carryingProximity) if carryingProximity else 0.0)
    
    # FEATURE 6: Average proximity of non-carrying workers to food
    nonCarryingProximity = []
    for w in workers:
        if not getattr(w, "carrying", False) and foods:
            dmin = min(approxDist(w.coords, f.coords) for f in foods)
            nonCarryingProximity.append(closeness(dmin))
    features.append(sum(nonCarryingProximity)/len(nonCarryingProximity) if nonCarryingProximity else 0.0)
    
    # FEATURE 7: Is queen blocking anthill? (binary)
    myQ = myInv.getQueen()
    myH = myInv.getAnthill()
    queenBlocking = 0.0
    if myQ and myH and workers:
        carryingWorkers = [w for w in workers if getattr(w, "carrying", False)]
        if carryingWorkers and myQ.coords == myH.coords:
            queenBlocking = 1.0
    features.append(queenBlocking)
    
    # FEATURE 8: Danger level - enemy attackers near our queen/hill (minor factor, 2% weight)
    enemyAtk = getAntList(gameState, enemy, (DRONE, SOLDIER, R_SOLDIER))
    danger = 0.0
    if enemyAtk and myQ and myH:
        dq = closeness(min(approxDist(a.coords, myQ.coords) for a in enemyAtk)) if myQ else 0.0
        dh = closeness(min(approxDist(a.coords, myH.coords) for a in enemyAtk)) if myH else 0.0
        danger = max(dq, dh)
    features.append(danger)
    
    # FEATURE 9: Food progress score (combines food + carrying for direct correlation)
    foodProgress = float(myInv.foodCount) + 0.5 * float(carryingCount)
    features.append(foodProgress / float(FOOD_GOAL))
    
    # FEATURE 10: Overall worker efficiency (combined proximity score)
    allWorkerProximity = carryingProximity + nonCarryingProximity
    features.append(sum(allWorkerProximity)/len(allWorkerProximity) if allWorkerProximity else 0.0)
    
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
# Description: A utility function that returns a float in [0, 1]
#     where 0 is closer to winning and 1 is closer to losing.
#
# Parameters:
#   gameState - a game state
#
# Return: The utility of the state (0.0 = best, 1.0 = worst)
##
def utility(gameState):
    # Terminal checks (fast path)
    me = gameState.whoseTurn
    enemy = 1 - me
    myInv = getCurrPlayerInventory(gameState)
    enemyInv = getEnemyInv(enemy, gameState)

    if gameState.phase == PLAY_PHASE:
        w = getWinner(gameState)
        if w == me or \
           len(getAntList(gameState, enemy, (QUEEN,))) == 0 or \
           myInv.foodCount >= FOOD_GOAL or \
           enemyInv.getAnthill().captureHealth <= 0:
            return 0.0
        if w == enemy or \
           len(getAntList(gameState, me, (QUEEN,))) == 0 or \
           enemyInv.foodCount >= FOOD_GOAL or \
           myInv.getAnthill().captureHealth <= 0:
            return 1.0

    # closeness helper - used to normalize distances
    def closeness(dist, maxd=10.0):
        try:
            return max(0.0, min(1.0, 1.0 - float(dist)/float(maxd)))
        except Exception:
            return 0.0

    # Simplified food-focused approach: just maximize OUR food
    # Food in inventory + food being carried = total food progress
    workers = getAntList(gameState, me, (WORKER,))
    
    # Count food in inventory
    foodInInventory = float(myInv.foodCount)
    
    # Count food being carried (almost delivered!)
    carryingWorkers = sum(1 for w in workers if getattr(w, "carrying", False))
    
    # Simple score: food collected + bonus for food in transit
    foodProgress = foodInInventory + 0.5 * float(carryingWorkers)
    
    # Normalize to [0,1] range
    foodScore = foodProgress / float(FOOD_GOAL)
    foodScore = min(1.0, foodScore)
    
    # Heavily penalize having no workers - can't gather food without them
    numWorkers = len(workers)
    workerCountBonus = 0.0
    if numWorkers == 0:
        # No workers = can't win, so huge penalty
        workerCountBonus = -0.5
    elif numWorkers >= 1:
        # Have at least 1 worker gives a small bonus
        workerCountBonus = 0.05
        # Too many workers wastes food, slight penalty
        if numWorkers > 2:
            workerCountBonus = 0.0
    
    # Worker positioning - encourage moving toward food/drop sites
    foods = getConstrList(gameState, None, (FOOD,))
    dropSites = []
    if myInv.getAnthill() is not None:
        dropSites.append(myInv.getAnthill().coords)
    tunnels = myInv.getTunnels()
    if tunnels:
        dropSites.extend([t.coords for t in tunnels])

    workerProximity = []
    for w in workers:
        if getattr(w, "carrying", False):
            # Worker carrying - get to drop site ASAP
            if dropSites:
                dmin = min(approxDist(w.coords, d) for d in dropSites)
                workerProximity.append(closeness(dmin, maxd=10.0))
        else:
            # Worker not carrying - get to food ASAP
            if foods:
                dmin = min(approxDist(w.coords, f.coords) for f in foods)
                workerProximity.append(closeness(dmin, maxd=10.0))
    
    progressScore = sum(workerProximity)/len(workerProximity) if workerProximity else 0.0

    # Penalize queen blocking anthill when worker is carrying food
    queenBlockingPenalty = 0.0
    myQ = myInv.getQueen()
    myH = myInv.getAnthill()
    if myQ and myH and workers:
        carryingWorkers = [w for w in workers if getattr(w, "carrying", False)]
        if carryingWorkers and myQ.coords == myH.coords:
            # Queen is on anthill and we have a worker with food, penalize
            queenBlockingPenalty = 0.3

    # Light safety factor, keep its weight small to stay food-focused
    enemyAtk = getAntList(gameState, enemy, (DRONE, SOLDIER, R_SOLDIER))
    danger = 0.0
    if enemyAtk and myQ and myH:
        dq = closeness(min(approxDist(a.coords, myQ.coords) for a in enemyAtk)) if myQ else 0.0
        dh = closeness(min(approxDist(a.coords, myH.coords) for a in enemyAtk)) if myH else 0.0
        danger = max(dq, dh)  # higher = closer, more dangerous

    # Combine to a "goodness" score in [0,1]
    # Food dominates everything - worker proximity is just a tiebreaker
    good = 0.90*foodScore + 0.08*progressScore + 0.02*(1.0 - danger) + workerCountBonus - queenBlockingPenalty
    good = max(0.0, min(1.0, good))

    # Inverse the score
    return 1.0 - good

##
# bestMove
#
# Description: Searches a given list of game nodes to find the best move
#
# Parameters:
#   gameState - a game state
#   moves - a list of moves
#
# Return: The node with the best utility
#
# Flag to control whether to use the ANN or simple utility
USE_NN = True  # default to utility

def bestMove(nodes, ann):
    # Initialize the best node with the first node's utility
    bestNodes = [nodes[0]]

    # Iterate through nodes to find the one with the highest utility
    for node in nodes:
        if node.evaluation is None:
            if USE_NN:
                mapping = mappingFunction(node.gameState)
                node.evaluation = ann.forward(mapping)[0] + node.depth
            else:
                node.evaluation = utility(node.gameState) + node.depth
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
        super(AIPlayer,self).__init__(inputPlayerId, "NeuralNet_pham27_krasnogo27")
        self.playerId = inputPlayerId
        # ANN: 10 inputs -> 20 hidden -> 10 hidden -> 1 output
        # Smaller network for faster training with focused features
        # Using pretrained weights from the bottom of the file
        self.ann = ANN(10, 20, 10, 1, 0.01, 400, 0.01, None, use_pretrained=True)


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

##
# get_pretrained_weights
#
# Description: Returns the pretrained weights and biases for the neural network
#
# Return: Tuple of (w1, b1, w2, b2, w3, b3) as numpy arrays
##
def get_pretrained_weights():
    
    w1 = np.array([
        [-0.13001366, -1.61836163, -0.66078776, -0.93548684, -0.34376706,  0.63968308,
          0.30409159, -0.0481603,   1.60094128, -1.64241976,  0.29053202,  1.47764153,
          0.27775632, -1.42828928,  1.37570331, -1.20379876,  1.64607116, -1.51543838,
          0.59879015,  0.03580319],
        [ 0.62527478,  0.24283355,  0.79070149, -1.07774257,  0.07584705, -0.17920652,
          0.80293631,  0.0105644,  -0.69933081,  0.55127924,  0.36508247, -0.01089418,
          0.66371024, -1.21394682,  0.22948248, -0.54594035,  0.47314592,  0.4914025,
         -0.59262816, -0.45700881],
        [ 0.7210711,   0.46288835,  0.31746221,  1.02238894,  0.56620372,  0.58988507,
          0.22929416, -0.68389179, -0.14644915,  1.2950877,   0.76878202, -0.19602037,
         -0.05160044,  0.44656323,  0.4441633,   0.04535139, -0.72740166,  1.0136483,
         -0.61914679,  1.03172363],
        [ 0.86673395,  0.95660563,  1.81708188, -0.41963305,  0.76619336, -1.15093773,
          0.71297495,  0.55600526, -1.09759515,  1.20965254, -0.60214272, -1.11417312,
          0.14364463,  1.43129129,  0.20242281,  1.3067704,   0.40798139, -0.16189563,
          0.13909438,  0.52466132],
        [-1.21582872,  0.75725841, -0.22179661,  0.48102187,  0.67204367,  0.84262138,
          0.13952048,  0.94440237,  0.5822222,   0.52227336,  0.750123,    0.76749574,
          0.54249767,  0.80962187, -0.44628445,  0.84856765,  0.06841216,  0.12012559,
         -1.03607859,  0.17225784],
        [ 0.75566007,  0.58895693,  0.46948662, -0.83966976,  0.71051075,  0.23710482,
         -0.31583715, -0.5309459,  -0.21434126,  0.32680462, -0.19516359,  0.53308756,
          0.29101821,  0.60853727,  0.76791186,  0.31137169,  0.77907552, -0.53423427,
         -0.32163453,  0.96624318],
        [-0.35381977, -0.89017526,  0.83987434, -0.28246743, -0.81927175, -0.52996776,
          0.9276768,  -0.54755018,  0.46637463, -0.71627428,  0.48305161,  0.6105662,
         -0.95216402,  0.12539068, -0.6195966,   0.37061933,  0.02468914,  0.34865011,
         -0.52089384,  0.13334175],
        [ 0.11145006, -0.58473893,  0.05429431,  0.62336017, -0.45078268,  0.51727088,
          0.65334524, -0.70070925,  0.210459,   -0.19326054, -1.34517389,  0.1746113,
         -0.5261251,  -0.93506219, -0.71646069,  0.43353358, -0.50803716,  0.74519935,
         -0.04028251,  0.1650555],
        [ 0.2703572,  -0.6861606,  -0.79802907, -0.44781245, -1.32192663, -0.28226373,
          0.09829075,  0.58626503,  1.47304576, -1.62811866,  1.06221143, -0.12601302,
          0.428069,   -0.90530397,  1.17364498, -0.69742669, -0.21737664, -0.52781432,
         -0.84570912,  0.06599087],
        [-0.57319394, -0.30931904, -0.88702827, -0.31334515, -0.75296527,  0.54495895,
         -0.73836185,  0.06291448,  0.15161729,  0.89269069, -0.2214813,  -0.59452276,
          0.7313611,  -0.61393935, -1.08775415, -0.44932087,  0.55311204, -1.02752768,
         -0.43906816, -0.81605432]
    ])


    b1 = np.array([[-0.80245955,  0.8561455,   0.96213933,  0.39326979,  0.51464652, -0.12797615,
                      0.46800163,  0.07029437, -2.75203909,  0.61664651, -0.86859549,  0.33072255,
                      0.37158542,  1.90271385, -0.06574476,  0.78842093, -1.0135619,   1.68666272,
                      0.46331509, -0.40592326]])
    
    w2 = np.array([
        [-5.66539474e-01, -3.84621564e-01, -9.77216002e-01, -8.39628583e-01,
         -1.10858364e-01,  8.57490442e-01, -7.65002207e-01, -7.06167737e-01,
         -7.37821029e-01, -1.77424743e-03],
        [-3.14718810e-01, -5.92831710e-01,  1.17955799e-01, -3.27711538e-01,
          1.90880994e+00,  9.19799100e-01,  4.25351452e-01, -1.25551178e-01,
          8.39536815e-01, -1.01327804e+00],
        [-9.61929254e-01, -9.77924559e-01,  9.32336567e-01, -4.99693500e-01,
          2.08881033e+00,  1.98989072e+00, -8.64625419e-01,  3.65191747e-01,
          5.10760479e-01, -1.04176050e+00],
        [-1.31972839e-01, -7.12669836e-01,  6.08837858e-01,  6.85658160e-02,
          9.69957779e-01,  8.37141394e-01,  1.33506362e-01,  7.61531724e-01,
          9.07884397e-01,  4.96149592e-01],
        [ 4.85289221e-01, -8.46796838e-01,  9.90988150e-01, -9.61677960e-01,
          1.79018650e+00,  5.37731093e-01,  9.02208978e-01,  9.69171161e-01,
          9.42209515e-01,  3.73930999e-01],
        [ 6.29363062e-01,  1.45348104e+00, -3.46512377e-01,  3.48050536e-01,
          5.45583133e-01, -9.52513203e-01,  1.02816004e+00,  7.16454222e-01,
          4.22818290e-01,  1.01097990e+00],
        [-1.93419587e-01,  8.48724031e-01,  9.51185618e-01, -8.34988270e-01,
         -1.57816192e-01,  9.44611461e-01, -5.66725546e-01,  4.95721447e-01,
         -7.76280707e-01,  2.94041419e-01],
        [ 8.49349518e-01, -3.42696663e-01,  1.21391821e-01,  1.46387508e-01,
         -8.89746317e-01,  2.83558960e-01,  2.53178485e-01,  9.20789417e-01,
          6.24768630e-01, -6.46615854e-01],
        [ 2.12898575e-01,  1.00667676e+00,  2.44547119e-01, -8.68571803e-01,
         -2.46182200e+00, -2.61530392e+00, -7.74374342e-02,  1.05400639e-01,
         -4.43250157e-02,  6.13699839e-01],
        [-9.41567290e-01,  2.25652474e-01,  8.15439560e-01, -6.21774445e-02,
          2.39162973e+00,  1.47312992e+00, -3.82264233e-01,  7.76958443e-01,
          5.30543760e-01,  1.45466937e-01],
        [-5.88306777e-01,  1.25627468e+00, -1.04737320e+00,  4.79079404e-02,
         -1.16377670e+00, -1.36869985e+00, -8.10838314e-01,  3.93733540e-01,
          9.75753982e-01,  8.39404989e-01],
        [-4.02926797e-01,  1.25407862e+00,  6.79578716e-01, -2.77023679e-01,
         -6.90206384e-01, -4.24989467e-01, -3.71955765e-01, -5.87274189e-01,
          8.60443112e-01,  1.02446218e+00],
        [-2.33342112e-01,  1.65160087e-01, -4.89318003e-01,  5.56549028e-01,
         -9.99501546e-01,  6.81748935e-01,  1.07856335e+00, -5.48527335e-01,
          7.14205484e-01, -2.93000630e-01],
        [ 2.61671643e-01, -1.19219626e+00, -5.50435012e-01, -4.90165225e-01,
          2.35920802e+00,  2.09634391e+00,  4.44339267e-01,  1.85868619e-02,
         -1.14432564e-01, -1.09037514e+00],
        [-9.78135825e-01,  1.63736628e+00, -4.63188799e-01, -7.01549719e-01,
         -6.82159276e-01, -1.19355320e+00,  4.44301670e-01, -7.36415489e-02,
         -3.98372652e-01,  8.63914611e-01],
        [ 2.77062214e-01, -7.49447603e-01,  3.91522044e-01,  1.90572310e-01,
          1.41867199e+00,  1.67090277e+00,  8.96893636e-02, -8.43220517e-02,
         -9.30865643e-01, -5.74268212e-01],
        [ 6.07618482e-01,  6.67326316e-01, -3.66463590e-01,  4.81397219e-01,
         -1.53676795e+00, -4.04507483e-01,  3.58058705e-01,  1.05046616e+00,
          6.54139825e-01, -7.00888422e-01],
        [ 9.70323235e-01, -7.98648472e-02,  5.43242528e-01, -4.83849032e-02,
          1.33511208e+00,  1.82736082e+00, -4.49960391e-01,  8.99638295e-01,
         -5.76896866e-01, -5.06647638e-01],
        [-5.90075997e-01, -1.23389385e-01,  3.20688122e-01, -9.84634120e-01,
          8.80159126e-01,  6.04065467e-01, -1.74236922e-01,  3.88885366e-01,
         -6.68100682e-01,  2.57820324e-02],
        [-5.41967458e-01,  5.09373335e-01, -6.90361594e-01,  8.10248678e-01,
          1.00035048e+00, -9.66544690e-02, -3.53549453e-01,  1.64617839e-01,
          6.01060538e-01, -1.40748783e-01]
    ])
    
    b2 = np.array([[-0.49528339,  0.50960815, -0.5004728,  -0.95396656,  1.15770996,  0.23110796,
                      0.2470093,   1.04743169, -0.46725536,  0.27509979]])
    
    w3 = np.array([
        [-0.2206747],
        [-3.58627054],
        [ 0.72355352],
        [-0.17918001],
        [ 4.13303779],
        [ 3.5864522],
        [-0.59632921],
        [-1.27283098],
        [-0.74421043],
        [-2.87497061]
    ])
    
    b3 = np.array([[-1.06043897]])

    
    return w1, b1, w2, b2, w3, b3
