import os


class Digraph(object):

	def __init__(self):
		"""
		Initializes an empty directed graph.
		"""
		self.nv = -1
		self.ne = -1

		self.verts = None # list of vertices
		self.incList = None # incidence list (nodes that point to a given node)
		self.adjList = None # adjacency list (nodes that a given node points to)
		self.inverted = False

	def load(self, path):
		"""
		Loads graph structure and edge data, populating 
		the vertices, incidence list, and adjacency list.
		"""
		if not os.path.exists(path):
			raise FileNotFoundError(f"File not found: {path}")

		with open(path, 'r') as f:
			header = f.readline().strip()
			if not header:
				raise ValueError("File is empty or invalid.")
			li = header.split(",")

			self.nv = int(li[0]) # number of vertices
			self.ne = int(li[1]) # number of edges

			print (f"Graph has {self.nv} nodes and {self.ne} arcs.")

			self.verts = list(range(self.nv))
			self.incList = [[] for i in range(self.nv)]
			self.adjList = [[] for i in range(self.nv)]

			for i in range(self.ne):
				line = f.readline()
				if not line:
					raise ValueError(f"Warning: End of file reached at iteration {i}. Expected {self.ne} edges.")
				parts = line.strip().split(",")

				if len(parts) < 2:
					continue  # Skip malformed lines

				orig = int(parts[0])
				dest = int(parts[1])
				#weight = int(parts[2]) #In future versions, we may use weights for the compatibilities

				self.incList[dest].append(orig)
				self.adjList[orig].append(dest)

	def showIncList(self):
		"""
		Prints the incidence list (incoming edges) for all vertices in the graph 
		to the console.
		"""
		print("\nPrinting Incoming List.")
		for i, li in enumerate(self.incList):
			print(i, li)
		print()

	def showAdjList(self):
		"""
		Prints the adjacency list (outgoing edges) for all vertices in the graph 
		to the console.
		"""
		print("\nPrinting Adjacency List.")
		for i, li in enumerate(self.adjList):
			print(i, li)
		print()

	def reverse(self):
		"""
		Sort vertices and adjacency/incidence lists in descending order.
		"""
		self.verts.sort(reverse=True)
		for el in self.adjList:
			el.sort(reverse=True)
		self.inverted = not self.inverted

	def to_dot_file(self, filename):
		"""
		Exports the current graph structure to a .dot file for visualization.
		"""
		with open(filename, 'w') as f:
			f.write("digraph G {\n")
			f.write("  node [shape=circle];\n")

			# Iterate through all nodes and their adjacency lists
			for u in range(self.nv):
				if u < len(self.adjList):
					for v in self.adjList[u]:
						# Write the directed edge: u -> v
						f.write(f'  "{u}" -> "{v}";\n')

			f.write("}\n")
		print(f"Graph exported to {filename}")
