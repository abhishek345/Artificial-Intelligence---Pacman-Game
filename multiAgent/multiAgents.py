# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        curFood = currentGameState.getFood()
        curFoodList = curFood.asList()
        curPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newFoodList = newFood.asList()
        
		
        ghostPositions = successorGameState.getGhostPositions()
        distance = float("inf")
        scared = newScaredTimes[0] > 0
        for ghost in ghostPositions:
          d = manhattanDistance(ghost, newPos)
          distance = min(d, distance)
        
        distance2 = float("inf")        
        distance3 = float("-inf")
        distance4 = float("inf")
        for food in newFoodList:
          d = manhattanDistance(food, newPos)
          d0 = manhattanDistance(food, curPos)
          distance2 = min(d, distance2)
          distance3 = max(d, distance3)

        cond = len(newFoodList) < len(curFoodList)
        count = len(newFoodList)
        if cond:
          count = 10000
        if distance < 2:
          distance = -100000
        else:
          distance = 0
        if count == 0:
          count = -1000
        if scared:
          distance = 0
        return distance + 1.0/distance2 + count - successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        v = float("-inf")
        bestAction = []
        agent = 0
        actions = gameState.getLegalActions(agent)
        successors = [(action, gameState.generateSuccessor(agent, action)) for action in actions]
	def minimax(agent, agentList, state, depth, evalFunc):
  
		if depth <= 0 or state.isWin() == True or state.isLose() == True:
			return evalFunc(state)
    
		if agent == 0:
			v = float("-inf")
		else:
			v = float("inf")
          
		actions = state.getLegalActions(agent)
		successors = [state.generateSuccessor(agent, action) for action in actions]
		for j in range(len(successors)):
			successor = successors[j];
    
			if agent == 0:
      
				v = max(v, minimax(agentList[agent+1], agentList, successor, depth, evalFunc))
			elif agent == agentList[-1]:
      
				v = min(v, minimax(agentList[0], agentList, successor, depth - 1, evalFunc))
			else:
     
				v = min(v, minimax(agentList[agent+1], agentList, successor, depth, evalFunc))
  
		return v

        for successor in successors:
            temp = minimax(1, range(gameState.getNumAgents()), successor[1], self.depth, self.evaluationFunction)
            
            if temp > v:
              v = temp
              bestAction = successor[0]
        return bestAction
        
  
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agentCount = gameState.getNumAgents()

        def multiminimax(state, depth, agentIndex, alpha, beta):
          legalActions = state.getLegalActions(agentIndex)
          if depth == 0 or len(legalActions) == 0:
            return (None, self.evaluationFunction(state))

          succAgentIndex = (agentIndex + 1) % agentCount
          succDepth = depth
          if succAgentIndex == 0: succDepth -= 1

          resultAction = None
          if agentIndex == 0:
            resultValue = float("-inf")
            for action in legalActions:
              succState = state.generateSuccessor(agentIndex, action)
              (_, succValue) = multiminimax(succState, succDepth, succAgentIndex, alpha, beta) 
              if succValue > resultValue:
                (resultAction, resultValue) = (action, succValue)
              if resultValue > beta: break
              alpha = max(alpha, resultValue)
          else:
            resultValue = float("inf")
            for action in legalActions:
              succState = state.generateSuccessor(agentIndex, action)
              (_, succValue) = multiminimax(succState, succDepth, succAgentIndex, alpha, beta) 
              if succValue < resultValue:
                (resultAction, resultValue) = (action, succValue)
              if resultValue < alpha: break
              beta = min(beta, resultValue)

          return (resultAction, resultValue)

        result = multiminimax(gameState, self.depth, 0, float("-inf"), float("inf"))
        return result[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState, depth, numAgent):
          if (depth==0):
            v=self.evaluationFunction(gameState)
            return v        
          v=-99999
          legalMoves = gameState.getLegalActions(0)
          if len(legalMoves)==0:
            return self.evaluationFunction(gameState)
          else:
            for move in legalMoves:
              succState=gameState.generateSuccessor(0, move)
              succVal=exp_value(succState, depth, numAgent, 1)
              v=max(v,succVal)
          return v

        def exp_value(gameState, depth, numAgent, agentNum):          
          legalMoves = gameState.getLegalActions(agentNum)
          if len(legalMoves)==0:
            return self.evaluationFunction(gameState)
          v=99999
          ttl=0
          succVal=99999
          for move in legalMoves:
            succState=gameState.generateSuccessor(agentNum, move)
            if agentNum==numAgent-1:
              succVal=max_value(succState, depth-1, numAgent)
              ttl+=succVal
            else:
              succVal=exp_value(succState, depth, numAgent, (agentNum+1))
              ttl+=succVal
          if ttl==0:
            return 0
          else:
            return ttl/(len(legalMoves))
        numAgent=gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(0)
        v=-99999        
        best_action=gameState.getLegalActions(0)[0]
        for move in legalMoves:
          succState=gameState.generateSuccessor(0, move)          
          tmp=exp_value(succState, self.depth, numAgent, 1)
          if tmp>v:
            v=tmp
            best_action=move
        return best_action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
	
    "*** YOUR CODE HERE ***"
	
    pos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood() #food available from current state
    currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates ]
    
    foodDist=9999
    ghostDist=9999
    scaredGhostDist=9999
    ghostCount=0
    scaredGhostCount=0

    for food in currentFood.asList():
      tmp=float(abs(pos[0] - food[0]) + abs(pos[1] - food[1]))
      if tmp<foodDist:
        foodDist=tmp

    for i, ghost in enumerate(ghostStates):
      gp=ghost.getPosition()
      tmp=abs(pos[0] - gp[0]) + abs(pos[1] - gp[1])
      if scaredTimes[i]>0:
        scaredGhostDist+=tmp
        scaredGhostCount=0
      else:
        ghostDist+=tmp
        ghostCount=0
    if scaredGhostCount>0:      
      scaredGhostDist=float(scaredGhostDist/scaredGhostCount)
    if ghostCount>0:      
      ghostDist=float(ghostDist/ghostCount)
    score=(2.0/(foodDist)+(-5.0)/(ghostDist)+(3.0)/(scaredGhostDist))+currentGameState.getScore()
    return score	

# Abbreviation
better = betterEvaluationFunction

