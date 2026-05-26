from kep.constants import *


class CycleFinder(object):

	def __init__(self, graph, characterization):
		self.graph = graph
		self.cycles = []
		self.chains = []
		self.max_depth = -1
		self.visited = None

		self.cycle_nodes = []
		self.altruistic_nodes = []

		for row in characterization:
			if characterization[row][COL_PAIR_DESCRIPTION].lower() == "altruist":
				self.altruistic_nodes.append(characterization[row][COL_DONOR_ID])
			else:
				self.cycle_nodes.append(characterization[row][COL_DONOR_ID])

		self.altruistic_set = set(self.altruistic_nodes)

		if self.graph.inverted:
			self.altruistic_nodes.sort(reverse=True)
			self.cycle_nodes.sort(reverse=True)
		else:
			self.altruistic_nodes.sort()
			self.cycle_nodes.sort()

	def find_cycles(self, max_cycle):
		"""
		Initiates a depth-first search to discover all cycles in the graph 
		up to max_cycle length.
		"""
		self.max_depth = max_cycle
		self.visited = [False] * self.graph.nv

		for v in self.cycle_nodes:

			if v >= self.graph.nv: continue

			self.stack = [v]
			self.root = v
			self.depth = 0

			self.visited[v] = True

			for u in self.graph.adjList[v]:
				if u < self.root:
					continue
				if u == self.root:
					self.cycles.append(list(self.stack))
				elif not self.visited[u] and u not in self.altruistic_set:
					self.visited[u] = True
					self.find_cycles_step(u)
					self.visited[u] = False

			self.visited[v] = False

	def find_cycles_step(self, u):
		"""
		A recursive helper method to build and validate cycles 
		up to the defined maximum depth.
		"""
		if self.depth < self.max_depth - 1:

			self.stack.append(u)
			self.depth += 1

			for i in self.graph.adjList[u]:
				if i == self.root:
					self.cycles.append(list(self.stack))
				elif i > self.root and not self.visited[i] and i not in self.altruistic_set:
					self.visited[i] = True
					self.find_cycles_step(i)
					self.visited[i] = False

			self.stack.pop()
			self.depth -= 1

	def find_chains(self, max_chain):
		"""
		Initiates a depth-first search to discover all valid chains starting 
		from altruistic nodes up to max_chain length.
		"""
		self.max_depth = max_chain - 1
		self.visited = [False] * self.graph.nv

		for v in self.altruistic_nodes:
			self.stack = [v]
			self.root = v
			self.depth = 0

			self.visited[v] = True

			for u in self.graph.adjList[v]:
				if not self.visited[u] and u not in self.altruistic_set:
					self.visited[u] = True
					self.find_chains_step(u)
					self.visited[u] = False

			self.visited[v] = False


	def find_chains_step(self, u):
		"""
		A recursive helper method to build chains 
		originating from an altruistic root up to the maximum depth.
		"""
		self.stack.append(u)
		self.depth += 1

		self.chains.append(list(self.stack))

		if self.depth < self.max_depth:
			for i in self.graph.adjList[u]:
				if not self.visited[i] and i not in self.altruistic_set:
					self.visited[i] = True
					self.find_chains_step(i)
					self.visited[i] = False

		self.stack.pop()
		self.depth -= 1
