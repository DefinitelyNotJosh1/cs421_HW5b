import random
import math
import sys
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *


##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##

#HW 2 methods here
def expandNode(node):
    state = node["state"]
    move_list = listAllLegalMoves(state)
    node_list = []
    for move in move_list:
        new_node = createNode(move, state, node["depth"] + 1, node)
        node_list.append(new_node)
    return node_list

def createNode(move, state, depth, node):
    new_node = {
    "move": move,
    "state": getNextState(state, move),
    "depth": depth,
    "parent": node,
    "evaluation": utility(getNextState(state, move)) + depth
    # evaluation is the sum of the utility and the depth
    }
    return new_node

def utility(state): #to be done later along with the unit tests
    bad = 0
    worker_part = 0
    soldier_part = 0
    queen = getAntList(state, state.whoseTurn, (QUEEN,))[0] # get queen
    if (len(getAntList(state, state.whoseTurn, (WORKER,))) > 0):
        worker = getAntList(state, state.whoseTurn, (WORKER,))[0] # get worker
    else:
        worker = None
        worker_part = 0
    tunnel = getConstrList(state, state.whoseTurn, (TUNNEL,))[0]
    food = getCurrPlayerFood(state.whoseTurn, state)
    if worker is None and state.inventories[state.whoseTurn].foodCount > 0: #if no workers and available food, make one
        return bad
    elif worker is None: return bad
    else:
         if not (worker.carrying): #WORKER: go to food if not already carrying it
            worker_part = (10-stepsToReach(state, worker.coords, food[0].coords)) + 35*(state.inventories[state.whoseTurn].foodCount)
         if (worker.carrying): #WORKER: go home if carrying food
            worker_part = (10-stepsToReach(state, worker.coords, tunnel.coords)) + 10 + 35*(state.inventories[state.whoseTurn].foodCount)
    anthill = getConstrList(state, state.whoseTurn, (ANTHILL,))[0]
    if (len(getAntList(state, state.whoseTurn, (R_SOLDIER,))) > 0):
        soldier = getAntList(state, state.whoseTurn, (R_SOLDIER,))[0]
    else: soldier = None
    if (queen.coords == anthill.coords): return bad #QUEEN: get off anthill
    if not (soldier == None): #SOLDIER: if a soldier exists and you killed the queen or worker, good
        if (len(getAntList(state, 1 - state.whoseTurn, (WORKER,))) < 1) or (len(getAntList(state, 1 - state.whoseTurn, (QUEEN,))) < 1):
            soldier_part += 6000
        else: #SOLDIER: go towards the worker but also a bit towards the squeen so it values vertical distance
            soldier_part = 600-stepsToReach(state, soldier.coords, getAntList(state, 1 - state.whoseTurn, (WORKER,))[0].coords)-stepsToReach(state, soldier.coords, getAntList(state, 1 - state.whoseTurn, (QUEEN,))[0].coords)
        if (len(getAntList(state, 1 - state.whoseTurn, (WORKER,))) < 1) and (len(getAntList(state, 1 - state.whoseTurn, (QUEEN,))) == 1):
            soldier_part += 600-stepsToReach(state, soldier.coords, getAntList(state, 1 - state.whoseTurn, (QUEEN,))[0].coords)
            #SOLDIER: stay a bit away from the queen
            #SOLDIER: if you damaged the queen, good
        if (len(getAntList(state, 1 - state.whoseTurn, (QUEEN,))) == 1):
            soldier_part +=  (2000-200*getAntList(state, 1 - state.whoseTurn, (QUEEN,))[0].health)
            if (stepsToReach(state, soldier.coords, getAntList(state, 1 - state.whoseTurn, (QUEEN,))[0].coords) < 3):
                soldier_part = 0
        else:
            soldier_part += 6000
        for drone in getAntList(state, 1 - state.whoseTurn, (DRONE,)):
            if stepsToReach(state, soldier.coords, drone.coords) < 2:
                return bad
    else:
        soldier_part = 0
    ret = (worker_part + soldier_part)
    ret = ret ** 0.2
    return 1.0 / (1.0 + math.exp(-ret))

neural = [] #weights for hidden nodes
neural_output = [] #weights for output node
deltas = [] #delta values for hidden nodes
memory = []

def run_neural(state):
    mapping = return_map(state)
    ret = 0
    for i in range(len(neural_output) - 1):
        summate= 0
        for i in range(len(neural) - 1):
            mapping[i] = neural[i+1]
    return ret

def return_map(state):
    pass

def input1(state):
    
    return 0

def input2(state):
    pass

def input3(state):
    pass

def learn(actual, expected, state):
    error = expected - actual
    d = actual * (1 - actual) * error
    mapping = return_map(state)

	for i in range(len(neural2)):
		if i > 0: neural2[i] = neural2[i] + 0.1 * d * n_input2[i-1]
		else: neural2[i] = neural2[i] + 0.1 * d * 1
		pass
	#hidden nodes: err is output * weight: neural2[i] * neural[i][j]
	for i in range(len(neural)):
		for j in range(len(neural[0])):
			d2 = neural2[i+1] * d
			if j > 0:
				neural[i][j] = neural[i][j] + a * d2 * n_input[j-1]
				#neural[i][j] = neural[i][j] + a * neural[i][j] * error * neural2[i+1] * (1 - neural2[i+1]) * n_input[j-1]
			else: 
				neural[i][j] = neural[i][j] + a * d2 * 1
				#neural[i][j] = neural[i][j] + a * neural[i][j] * error * neural2[1+1] * (1 - neural2[i+1]) * 1
	return error

def flashback():
    for a in range(len(memory)): #for every memory
        i = memory[random.randint(0, a)] #take a random memory and remove it from memory
        memory.remove(i)
        learn(run_neural(i), utility(i), i) #learn based on memory
    pass

def bestMove(nodes): #find best move in a given list of nodes
    best_utility = -1 #intitialize at the move that takes 999 moves to win wweeww
    best_move = None

    for node in nodes:
        utility = node["evaluation"]
        move = node["move"]
        print(utility)
        #if (utility > best_utility): # rank their utility and take the best
        #    best_utility = utility
        #    best_move = move
        if (utility > best_utility): #rank the number of moves to reach goal from moves and take the smallest wweeww
            best_utility = utility
            best_move = node

    return best_move

class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##

    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "neural")

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
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
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
            #HW 2 methods here



    def getMove(self, currentState):
#        legal_moves = listAllLegalMoves(currentState)
#        node_list = []
#        for i in legal_moves:
#            legal_node = createNode(i, currentState, 0, None)
#            node_list.append(legal_node)
#        print(node_list)
#        return bestMove(node_list)["move"]
        legal_moves = listAllLegalMoves(currentState)
        node_list = []

        frontiernodes = []
        expandednodes = []
        #root node from current state
        parent_node = createNode(Move(END, None, None), currentState, 0, "Test")
        frontiernodes.append(parent_node)

        for i in range(3):
            best_frontier = bestMove(frontiernodes) # find best node in frontiernodes
            frontiernodes.remove(best_frontier) # remove it from frontier nodes
            expandednodes.append(best_frontier) # append to expanded nodes
            nodes = expandNode(best_frontier)# expand best frontier
            for j in nodes:
                frontiernodes.append(j) #append the new frontiers to frontier

        best_frontier_new = bestMove(frontiernodes)

        depth = best_frontier_new["depth"]
        while True:
            if best_frontier_new["parent"] is parent_node:
                break
            else:
                best_frontier_new = best_frontier_new["parent"]


        #for move in legal_moves:
        #    nextState = getNextState(currentState, move)
        #    depth = 1   # depth stays 1 for HW A
        #    node = createNode(move, nextState, depth, currentState, parent_node)
        #    node_list.append(node)

        #return bestMove(node_list)["move"]
        print(best_frontier_new["evaluation"])
        return best_frontier_new["move"]

    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
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
