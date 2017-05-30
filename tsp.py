import random
import math
import copy

tsp = lambda: None
tsp.node_list = []
tsp.start_node = None
vehicle_capacity = None
tsp.mutation_rate = 0.7
tsp.generations = 5
tsp.population_size = 5
tsp.tournament_size = 2
elitism = True

class GA(object):
  def crossover(self, parent_A, parent_B):
    
    child_route = Route()
    
    for x in range(1,len(child_route.route)):
      child_route.route[x] = None
    
    start_pos = random.randint(1,len(parent_A.route))
    end_pos = random.randint(1,len(parent_A.route))
    
    if start_pos < end_pos:
      for x in range(start_pos, end_pos):
        child_route.route[x] = parent_A.route[x]
    
    elif start_pos > end_pos:
      for x in range(end_pos, start_pos):
        child_route.route[x] = parent_A.route[x]      

    for i in range(len(parent_B.route)):
      if not parent_B.route[i] in child_route.route:
        for x in range(len(child_route.route)):
          if child_route.route[x] == None:
              child_route.route[x] = parent_B.route[i]
              break

    child_route.calculate_length()
    return child_route

  def mutate(self, mutate_route):

    if random.random() < tsp.mutation_rate:
        if len(mutate_route.route) > 1:
          index_A = random.randint(1,len(mutate_route.route)-1)
          index_B = random.randint(1,len(mutate_route.route)-1)

          if index_A == index_B:
              return mutate_route

          node_A = mutate_route.route[index_A]
          node_B = mutate_route.route[index_B]

          mutate_route.route[index_B] = node_A
          mutate_route.route[index_A] = node_B
          mutate_route.calculate_length()
    return mutate_route
  
  def evolve(self,initial_population):

    descendant_population = Population(size=initial_population.size, initialise=True)
    
    # Number of routes carried to new population.
    elitism_offset = 0

    # Set the first of the new population to the fittest of the old.
    if elitism:
        descendant_population.population[0] = initial_population.fittest
        elitismOffset = 1

    # Goes through the new population and fills it with the child of two tournament winners from previous populations
    for x in range(elitism_offset, descendant_population.size):

        tournament_parent_A = self.tournament(initial_population)
        tournament_parent_B = self.tournament(initial_population)

        # Create a child of these parents
        tournament_child = self.crossover(tournament_parent_A, tournament_parent_B)

        # Fill the population up with children
        descendant_population.population[x] = tournament_child

    # Mutate
    for route in descendant_population.population:
        if random.random() < tsp.mutation_rate:
            route = self.mutate(route)

    # Update the fittest route:
    descendant_population.get_fittest()

    return descendant_population



  def tournament(self,current_population):
    tournament_population = Population(size=tsp.tournament_size, initialise=False)
    
    for i in range(tsp.tournament_size-1):
        tournament_population.population.append(random.choice(current_population.population))
    
    return tournament_population.get_fittest()



class Population(object):
  def __init__(self, size, initialise):
      self.population= []
      self.size = size

      if initialise:
          for x in range(0,size):
              new_route = Route()
              self.population.append(new_route)
          fittest = self.get_fittest()
          # fittest.print_route()

  def get_fittest(self):
      sorted_by_fitness = sorted(self.population, key=lambda x: x.length, reverse=False)
      self.fittest = sorted_by_fitness[0]
      return self.fittest


class Node(object):
  def __init__(self, name, x, y, demand=None):
    self.name = name
    self.x = x
    self.y = y
    self.demand = demand

    tsp.node_list.append(self)

    self.distances = {self.name:0.0}

  def calculate_distances(self): 
      for node in tsp.node_list:
          dist = self.euclidean(self.x, self.y, node.x, node.y)
          self.distances[node.name] = dist

  def euclidean(self,x1,y1,x2,y2):
      return pow(pow(x1-x2,2) + pow(y1-y2,2),0.5)
    
  def print_distances(self):
    for node in self.distances:
      print "Distance to node %s is %d" % ( node, self.distances[node])

  def print_attributes(self):
    print "I am Node %s at (%d , %d) with demand of %d " % ( self.name, self.x, self.y, self.demand)

class Route(object):
  def __init__(self, route=None):
    self.route = []
    self.route = sorted(tsp.node_list[1:], key=lambda *args: random.random())
    self.route.insert(0,tsp.start_node)
    self.length = 0.00
    if route:
      self.route = route
    self.calculate_length()
    # self.print_route()

  def calculate_length(self):
    self.length = 0.00
    self.capacity = 0
    for idx, node in enumerate(self.route):
      current = node
      next_node = self.route[(idx + 1) % len(self.route)]
      self.length += node.distances[next_node.name]
      self.capacity += next_node.demand

  def print_route(self):
    path = 'Route: '
    coord = 'Coordinates: |'
    for node in self.route:
      path += node.name + ','
      coord +=  str(node.x) + ',' + str(node.y) + '|'

    path = path[:-1]
    print path
    print coord
    print "Length: %d" %  (self.length)
    print "Capacity: %d" % (self.capacity)

def GA_loop(route):
  r = Route(route)
  the_population = Population(tsp.population_size,True)
  the_population.population[-1] = r
  the_population.get_fittest()
  initial_length = the_population.fittest.length
  # print initial_length
  # print "tbaba"
  min =  initial_length
  # print min
  pop = None
  best_route = the_population.fittest.route
  for i in range(1,tsp.generations):
    the_population = GA().evolve(the_population)
    # print i
    if the_population.fittest.length < min:
      best_route = copy.deepcopy(the_population.fittest.route)
      pop = copy.deepcopy(the_population.fittest)
      # print the_population.fittest.length
  # if pop:
   # print pop.length
  return best_route



# def parse(filename):
#   file = filename
#   dimension = None
#   capaciy = None
#   section = None
#   for x in open(file, 'r'):
#       line = x.split()
      
#       if line[0] == 'DIMENSION':
#         dimension = line[2]
#       elif  line[0] == 'CAPACITY':
#         vehicle_capacity = line[2]

#       elif  line[0] == 'NODE_COORD_SECTION':
#         section = line[0]

#       elif  line[0] == 'DEMAND_SECTION':
#         section = line[0]
#         counter = 0
      
#       elif section == 'DEMAND_SECTION':
#         tsp.node_list[counter].demand = int(line[1])
#         counter = counter + 1 

#       elif section == 'NODE_COORD_SECTION':
#         current = Node(line[0],int(line[1]),int(line[2]))

# def initialise_map(filename):
#   parse(filename)
#   for node in tsp.node_list:
#     node.calculate_distances()
#   for node in tsp.node_list:
#     if node.name == '1':
#       tsp.start_node = node
#     node.print_attributes()
#     # node.print_distances()
#   population = Population(population_size,True)

def local_tsp(route,start):
  # global tsp.node_list
  # route = []
  # path = [1, 82, 88, 210, 22, 191, 242, 40, 223, 15, 47, 217, 36, 120, 240, 201, 142]
  # for x in tsp.node_list:
  #   if int(x.name) in path:
  #     route.append(x)
  #     print x.name
  # tsp.node_list = route
  # return GA_loop()
  if start:
    tsp.mutation_rate = 0.7
    tsp.generations = 20
    tsp.population_size = 10
    tsp.tournament_size = 3
  
  else:
    tsp.mutation_rate = 0.7
    tsp.generations = 5
    tsp.population_size = 5
    tsp.tournament_size = 2

  tsp.node_list = route
  tsp.start_node = route[0]
  optimal = GA_loop(route)
  # output_route(optimal)
  return optimal

def output_route(route):
  arrow = "->"
  route_string = ''
  for x in route.route:
    route_string += x.name + arrow
  route_string += '1'
  print route_string


# if __name__ == '__main__':
#   filename = "fruitybun250.vrp"
#   initialise_map(filename)
#   optimal = local_tsp()
#   output_route(optimal)





