import random
import math
import copy
from sets import Set
from collections import deque
import time
import tsp
import pdb
from hashlib import md5
import multiprocessing
import time

node_list = []
node_map = {}
dimension = None
start_node = None

vehicle_capacity = None

population_size = 100
tournament_size = 10
bloom = None
generations = 30000

elitism = True
elite = 1

mutation_rate = 0.1
crossover_rate = 0.9
cross_hill_rate = 3
mutate_hill_rate = 20
tabu = 1
exit_time = 0

class BloomFilter:
    def __init__(self, num_bytes, num_probes, items, iterable=() ):
        self.array = bytearray(num_bytes)
        self.num_probes = num_probes
        self.num_bins = num_bytes * 8
        self.items = items
        self.update(iterable)

    def get_probes(self, key):
        random = Random(key).random
        return (int(random() * self.num_bins) for _ in range(self.num_probes))

    def update(self, keys):
        self.items += 1
        
        if self.items > 500:
          self.array = bytearray(4 * 1024)
          self.items = 1

        for key in keys:
            for i in self.get_probes(key):
                self.array[i//8] |= 2 ** (i%8)

    def __getitem__(self, key):
        return all(self.array[i//8] & (2 ** (i%8)) for i in self.get_probes(key))

class BloomFilter_32k(BloomFilter):
    # 32kb (2**18 bins), 13 probes. Holds 13,600 entries with 1 error per 10,000.

    def __init__(self, iterable=()):
        BloomFilter.__init__(self, 4 * 1024, 1, 0, iterable)

    def get_probes(self, key, md5=md5, int=int, range13=tuple(range(1))):
        h = int(md5(key.encode()).hexdigest(), 16)
        for _ in range13:
            yield h & 32767    # 2 ** 18 - 1
            h >>= 15

class GA(object):

  def crossover(self, parent_1, parent_2):
    while True:
      child_routes = []
      # if random.random() < 0.5:
      if len(parent_1.routes) <= len(parent_2.routes):
        routes_size = len(parent_1.routes)
      else:
        routes_size = len(parent_2.routes)
      
      used = Set()
      seen = Set()
      
      for routeIndex in range(routes_size):
        route_exist = True
        if routeIndex < len(parent_1.routes) and routeIndex < len(parent_2.routes):
          # if len(parent_1.routes[routeIndex]) <= len(parent_2.routes[routeIndex]):
          if random.random() < 0.5:
            parent_A = parent_1
            parent_B = parent_2
          else:
            parent_A = parent_2
            parent_B = parent_1
        elif routeIndex < len(parent_1.routes) :
          parent_A = parent_1
          parent_B = parent_2
          route_exist = False
        else:
          parent_A = parent_2
          parent_B = parent_1
          route_exist = False

        routeA = parent_A.routes[routeIndex]
        if route_exist:
          routeB = parent_B.routes[routeIndex]
        
        sizes = len(parent_A.routes[routeIndex])
        
        child_route = [None]*sizes
        # print child_route
        # print len(routeA)
        # print len(routeB)
        child_route =  copy.copy(parent_A.routes[routeIndex])
        for node_index in range(1,len(child_route)):
          child_route[node_index] = None
        
        capacity = 0

        start_pos = random.randint(1,len(routeA))
        end_pos = random.randint(1,len(routeA))
        
        for x in range(1, len(routeA)):
          seen.add(node_map[routeA[x].name])
        for x in range(1, len(routeB)):
          seen.add(node_map[routeB[x].name])
      
        if start_pos < end_pos:
          for x in range(start_pos, end_pos):          
            if node_map[routeA[x].name] not in used and routeA[x].demand + capacity <= vehicle_capacity:
              child_route[x] = node_map[routeA[x].name]
              used.add(node_map[routeA[x].name])
              capacity += routeA[x].demand
          
        elif start_pos > end_pos:
          for x in range(end_pos, start_pos):
            if routeA[x] not in used  and routeA[x].demand + capacity <= vehicle_capacity:
              child_route[x] = node_map[routeA[x].name]
              used.add(node_map[routeA[x].name])     
              capacity += routeA[x].demand
        
        if route_exist == True:
          for i in range(1,len(routeB)):
            # print routeB[i].demand + capacity 
            if node_map[routeB[i].name] not in used and routeB[i].demand + capacity <= vehicle_capacity:
              end_of_list = True          
              for x in range(1,len(child_route)):
                if child_route[x] is None :
                  child_route[x] = node_map[routeB[i].name]
                  used.add(node_map[routeB[i].name])
                  capacity += routeB[i].demand
                  end_of_list = False
                  break
              
              if end_of_list == True:
                  child_route.append(node_map[routeB[i].name])
                  used.add(node_map[routeB[i].name])
                  capacity += routeB[i].demand            

        child_route = [x for x in child_route if x is not None]
        
        child_routes.append(child_route)
      
      capacities = range(len(child_routes))
      child = Route(child_routes,capacities)
      child.calculate_capacity()
      leftover = list(seen - used)
      
      leftover = deque(leftover)
      
      random_ind = range(0,len(child_routes))
      random_ind = sorted(random_ind, key=lambda *args: random.random())

      extra = []
      flag = 0
      while leftover:
        node = leftover.popleft()
        for x in random_ind:
          if child.capacities[x] + node.demand <= vehicle_capacity:
            child.routes[x].append(node_map[node.name])
            child.capacities[x] += node.demand
            flag = 1
            break
        if flag == 0:  
          extra.append(node_map[node.name])

      if len(extra) > 0:
        extra.insert(0,start_node)
        child.routes.append(extra)
        extra_capacity = 0
        child.capacities.append(extra_capacity)

      child.calculate_length()
      child.str()

      if len(seen) == 249 or len(leftover) == 0:
        if child.is_node_full():
         return child    

  def index_mutate(self, mutate_route, result):
    
    
    index_route_A = result[0]
    index_node_A = result[1]
    index_route_B = result[2]
    index_node_B = result[3]


    route_A = mutate_route.routes[index_route_A]
    route_B = mutate_route.routes[index_route_B]
    
    node_A = route_A[index_node_A]
    node_B = route_B[index_node_B]


    mutate_route.routes[index_route_A][index_node_A] = node_map[node_B.name]
    mutate_route.routes[index_route_B][index_node_B] = node_map[node_A.name]
    
    mutate_route.calculate_length()

    return mutate_route

  def mutate(self, mutate_route):      
    while True:
      index_route_A = random.randint(0,len(mutate_route.routes)-1)
      index_route_B = random.randint(0,len(mutate_route.routes)-1)
      
      if len(mutate_route.routes[index_route_A]) > 1 and len(mutate_route.routes[index_route_B]) > 1:

        index_node_A = random.randint(1,len(mutate_route.routes[index_route_A])-1)
        index_node_B = random.randint(1,len(mutate_route.routes[index_route_B])-1)

        if index_route_A == index_route_B and index_node_A == index_node_B:
            return None

        route_A = mutate_route.routes[index_route_A]
        route_B = mutate_route.routes[index_route_B]

        node_A= route_A[index_node_A]
        node_B = route_B[index_node_B]
        # node_A_name = route_A[index_node_A].name
        # node_B_name = route_B[index_node_B].name
       
        cap_A = mutate_route.capacities[index_route_A]
        cap_B = mutate_route.capacities[index_route_B]
        
        if index_route_A == index_route_B :

          mutate_route.routes[index_route_A][index_node_A] = node_map[node_B.name]
          mutate_route.routes[index_route_B][index_node_B] = node_map[node_A.name]


          x = []
          x.extend([mutate_route.length, index_route_A, index_node_A, index_route_B, index_node_B])

          mutate_route.routes[index_route_A][index_node_A] = node_map[node_A.name]
          mutate_route.routes[index_route_B][index_node_B] = node_map[node_B.name]


          return x

        elif cap_A - node_A.demand + node_B.demand <= vehicle_capacity:
          if cap_B - node_B.demand + node_A.demand <= vehicle_capacity:
            # print mutate_route.length
            mutate_route.routes[index_route_A][index_node_A] = node_map[node_B.name]
            mutate_route.routes[index_route_B][index_node_B] = node_map[node_A.name]
            x = []
            x.extend([mutate_route.length, index_route_A, index_node_A, index_route_B, index_node_B])
            mutate_route.routes[index_route_A][index_node_A] = node_map[node_A.name]
            mutate_route.routes[index_route_B][index_node_B] = node_map[node_B.name]
            return x

    return None

  def evolve(self,initial_population):

    descendant_population = Population(size=initial_population.size, initialise=True)

    if elitism:
      for x in range(elite):
        descendant_population.population[x] = copy.copy(initial_population.population[int(x)])
    
    for i in range(elite, descendant_population.size):

      tournament_parent_A = self.tournament(initial_population)

      if random.random() < crossover_rate:

        tournament_parent_B = self.tournament(initial_population)
        
        if tabu == 0:
          tournament_child = self.crossover(tournament_parent_A, tournament_parent_B)
          descendant_population.population[i] = tournament_child
          
          tournament_parent_B = self.tournament(initial_population)
        
        elif tabu == 1:
          generated_children = []
          for y in range(0,cross_hill_rate):
            tournament_child = self.crossover(tournament_parent_A, tournament_parent_B)
            generated_children.append(tournament_child)
          
          generated_children.sort(key=lambda x:x.length,reverse=False)
          descendant_population.population[i] = generated_children[0]

        elif tabu == 2:

          generated_children = []
          score = float('inf')

          for y in range(0,cross_hill_rate):
            tournament_child = self.crossover(tournament_parent_A, tournament_parent_B)
            generated_children.append(tournament_child)

          generated_children.sort(key=lambda x:x.length,reverse=False)

          chosen_child = None

          
          for child in generated_children:
            if not bloom[child.string]:
              chosen_child = child
              bloom.update(chosen_child.string)
              break

          if not chosen_child:
            chosen_child = child


          descendant_population.population[i] = chosen_child

      else:
        descendant_population.population[i] = copy.copy(tournament_parent_A)

    # Mutate
    for routes in descendant_population.population:
      if random.random() < mutation_rate:
        score = float('inf')
        best = []
        result = []
        routes.calculate_length()
        for x in range(0,mutate_hill_rate):
          result = self.mutate(routes)
          
          if result is not None:
            if result[0] < score:
              score = result[0]
              best = result[1:]
        
        if best:
          routes = self.index_mutate(routes, best)

    descendant_population.get_fittest()

    return descendant_population

  def tournament(self,current_population):
    tournament_population = Population(size=tournament_size, initialise=False)
    
    for i in range(tournament_size-1):
        tournament_population.population.append(random.choice(current_population.population))
    
    return tournament_population.get_fittest()

class Population(object):
  def __init__(self, size, initialise, start=False):
      self.population= []
      self.size = size
      if start:
        for x in range(0,size):
          new_route = Route(init=True,start=True)
          self.population.append(new_route)
        fittest = self.get_fittest()
        # print fittest.length

      elif initialise:
        for x in range(0,size):
          new_route = Route(init=False)
          self.population.append(new_route)
        fittest = self.get_fittest()
        # fittest.print_route()

  def get_fittest(self):
      self.population.sort(key=lambda x: x.length, reverse=False)
      self.fittest = self.population[0]
      return self.fittest

class Node(object):
  def __init__(self, name, x, y, demand=None):
    self.name = name
    self.x = x
    self.y = y
    self.demand = demand

    node_list.append(self)

    self.distances = {self.name:0.0}

  def calculate_distances(self): 
      for node in node_list:
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
  def __init__(self, routes=None, capacities=None, init=False, start=False):
    self.routes = []
    self.capacities = []
    self.string = ''
    self.depot = start_node
    self.length = 0.0
    if init is False:
      if routes:
        self.routes = routes
        self.capacities = capacities
        self.calculate_length()
        self.str()
      # else:
        # self.generate_random_route()   
    if init is True:
      self.generate_cluster_route(start)
      # self.generate_angled_route(start)
      self.str()

  def generate_cluster_route(self,start):
    search_space = copy.copy(node_list[1:])
    
    while search_space:
        
      min_distance = float('inf')
      index = 0
      
      current_route = []
      current_route.append(self.depot)
      
      cluster_seed = random.choice(search_space)
      current_route.append(cluster_seed)
      current_node = cluster_seed
      capacity = current_node.demand

      search_space.remove(cluster_seed)


      while search_space:
        min_distance = float('inf')
        # print len(search_space)
        for idx , node in enumerate(search_space):
          if node.name is not current_node.name:
            if current_node.distances[node.name]< min_distance:
              next_node = node
              min_distance = current_node.distances[node.name]
              index = idx

        capacity += next_node.demand
        
        if capacity > 500:
          break

        # print "cap:"  + str(capacity)

        current_node = next_node
        current_route.append(next_node)

        # print "index:"  + str(index)
        # print "search:"  + str(len(search_space))

        # print "\n"

        del search_space[index]
      
      if start:
        current_route = tsp.local_tsp(current_route,start)

      self.routes.append(current_route)
      self.capacities.append(capacity)
    self.calculate_length()

  def str(self):
    self.string = ''
    for route in self.routes:
        self.string += ','.join([node.name for node in route])
        self.string += ','
    self.string = self.string[:-1]
    # print self.string 
    # print "\n"
  
  def is_node_full(self):
    a = Set()
    for x in self.routes:
      for y in x:
        a.add(node_map[y.name])
    if len(a) < 250:
      return False
    else:
      return True

  def is_valid(self):
    self.calculate_length()
    
    for x in self.capacities:
      if x > 500:
        return False

    return True

  def generate_angled_route(self,start): 
    route = copy.copy(node_list[1:]) 
    route.sort(key=lambda x:x.polar_angle,reverse=False)

    index = random.randint(0,len(route))
    route = rotate(route,index)
    
    routes = []
    current_route = []
    current_route.append(self.depot)
    sum = 0
    
    for node in route:
      capacity = node.demand
      sum += capacity
      if sum > vehicle_capacity:
        self.capacities.append(sum-capacity)
        if start:
          current_route = tsp.local_tsp(current_route,start)
        self.routes.append(current_route)
        current_route = []
        current_route.append(self.depot)
        current_route.append(node)
        sum = capacity
      else:
        current_route.append(node)
    
    if start: 
      current_route = tsp.local_tsp(current_route,start)
    
    self.routes.append(current_route)
    self.capacities.append(sum)
    self.calculate_length()

  def generate_distanced_route(self,start): 
    route = copy.copy(node_list[1:]) 
    route.sort(key=lambda x:x.polar_dist,reverse=False)

    # route.sort(key=lambda x:x.polar_dist,reverse=False)
    # route = sorted(node_list[1:])
    index = random.randint(0,len(route))
    route = rotate(route,index)
    
    routes = []
    current_route = []
    current_route.append(self.depot)
    sum = 0
    
    for node in route:
      capacity = node.demand
      sum += capacity
      if sum > vehicle_capacity:
        self.capacities.append(sum-capacity)
        if start:
          current_route = tsp.local_tsp(current_route,start)
        self.routes.append(current_route)
        current_route = []
        current_route.append(self.depot)
        current_route.append(node)
        sum = capacity
      else:
        current_route.append(node)
    
    if start: 
      current_route = tsp.local_tsp(current_route,start)
    
    self.routes.append(current_route)
    self.capacities.append(sum)
    self.calculate_length()
  
  def generate_random_route(self):
    route = sorted(node_list[1:], key=lambda *args: random.random())
    # route = sorted(node_list[1:])

    routes = []
    current_route = []
    current_route.append(self.depot)
    sum = 0
    for node in route:
      capacity = node.demand
      sum += capacity
      if sum > vehicle_capacity:
        self.capacities.append(sum-capacity)
        self.routes.append(current_route)
        current_route = []
        current_route.append(self.depot)
        current_route.append(node)
        sum = capacity
      else:
        current_route.append(node) 
    self.routes.append(current_route)
    self.capacities.append(sum)
    self.calculate_length()

  def calculate_capacity(self):
    for idx, route in enumerate(self.routes):
      self.capacities[idx] = 0
      for idn, node in enumerate(route):
        if node is not None:
          current_node = node
          self.capacities[idx] += current_node.demand
    
  def calculate_length(self):
    self.length = 0.0
    capacity = 0
    for idx, route in enumerate(self.routes):
      self.capacities[idx] = 0
      for idn, node in enumerate(route):
        if node is not None:
          current_node = node
          self.capacities[idx] += current_node.demand
          
          next_node = route[(idn + 1) % len(route)]
          if next_node is not None:
            self.length += node.distances[next_node.name]
    
  def print_routes(self):
    coord = 'Coordinates: |'
    for idx, route in enumerate(self.routes):
      path = 'Route %d : ' % (idx)
      for node in route:
        if node is None:
          path += 'None' + ','
        else :
          path += node.name + ','
        # coord +=  str(node.x) + ',' + str(node.y) + '|'
      path = path[:-1]
      # print coord
      print path
      print "Capacity : %d" % (self.capacities[idx])
    print "Length: %d" %  (self.length)
    print '\n'

def parse_solution(name):
  file = open(name, 'r')
  line = file.readline()
  line = file.readline()
  line = file.readline()
  line = file.readline()
  arrow = "->"
  lines = file.readlines()

  file.close()
  
  lines = [x.strip() for x in lines]
  route = []
  routes = []
  for x in lines:
    x = x.split(arrow)
    x = x[:-1]
    
    route = []

    for y in x:
      route.append(node_map[y])
    routes.append(route)


  cap = range(len(routes))


  r = Route(routes=routes, capacities=cap)

  return r

def parse(filename):
  global vehicle_capacity, dimension, start_node
  file = filename
  dimension = None
  capaciy = None
  section = None
  for x in open(file, 'r'):
      line = x.split()
      
      if line[0] == 'DIMENSION':
        dimension = int(line[2])
      elif  line[0] == 'CAPACITY':
        vehicle_capacity = int(line[2])
        # vehicle_capacity += 5

      elif  line[0] == 'NODE_COORD_SECTION':
        section = line[0]

      elif  line[0] == 'DEMAND_SECTION':
        section = line[0]
        counter = 0
      
      elif section == 'DEMAND_SECTION':
        node_list[counter].demand = int(line[1])
        counter = counter + 1 

      elif section == 'NODE_COORD_SECTION':
        current = Node(line[0],int(line[1]),int(line[2]))
        if line[0] is '1':
          start_node = current
        current.polar_angle = math.degrees(math.atan2(current.y-start_node.y,current.x-start_node.x))
        current.polar_dist = math.sqrt(math.pow(current.x-start_node.x,2) + math.pow(current.y-start_node.y,2)) 

def GA_loop(previous_solution):
  global vehicle_capacity, mutation_rate,crossover_rate
 
  i = 0
  min = float('inf')
  y = 0
  m = 0
  if previous_solution:
    the_population = Population(size=population_size, initialise=False)
 
    for x in range(population_size):
      the_population.population.append(copy.deepcopy(previous_solution))

    the_population.get_fittest()
 
    initial = copy.deepcopy(previous_solution)

    the_population.population[-1] = copy.copy(previous_solution)
    the_population.get_fittest()

    min = copy.copy(the_population.fittest.length)
    output(the_population.fittest, 'phase2.txt')

  else:
    the_population = Population(size=population_size, initialise=True, start=True)
  
    while the_population.fittest.length > 6600:
      the_population = Population(size=population_size, initialise=True, start=True)
      the_population.get_fittest()

    initial = copy.deepcopy(the_population.fittest)

  for x in range(generations):
    if time.time() < exit_time:
      the_population = GA().evolve(the_population)
      # print i
      i+=1

      if the_population.fittest.length < min :
        if the_population.fittest.is_valid():
          the_population.fittest.calculate_length()
          if the_population.fittest.length < min :
            output(the_population.fittest,'phase2.txt')
            min = copy.copy(the_population.fittest.length)
            actual_min = min
            y = i
            # index = str(i)
            # index += ', '
            # index += str(the_population.fittest.length)
            # index += '\n'
            # file_best.write(index)
            # print "".join("\t"+str(the_population.fittest.length))

      if previous_solution:
        if i > 1 and i < 50:
          mutation_rate = 0.7

        if i == 50:
          mutation_rate = 0.2
          the_population.population[-1] = initial
          the_population.get_fittest()

        if i % 200 == 0 and m == 0:
          vehicle_capacity = 510
          mutation_rate = 0.7
          m = i

        if m + 200 == i:
          m = 0
          mutation_rate = 0.2
          vehicle_capacity = 500

def output(route, name):
  line = []
  arrow = "->"
  line.append("login im13557 65091\n")
  line.append("name Iman Malik\n")
  line.append("algorithm Two-Phased Genetic Algorithm\n")
  line.append("cost " + str("%.3f" % route.length) + "\n")

  file = open(name, 'w')
  for x in line:
    file.write(x)
  for r in route.routes:
    route_string = ''
    if len(r) > 1:
      for node in r:
        route_string += node.name
        route_string += arrow
      route_string += "1\n"
      file.write(route_string)
  file.close()

def rotate(l, n):
    return l[-n:] + l[:-n]

def initialise_map(filename):
  global start_node, bloom, node_map
  parse(filename)
  for node in node_list:
    node.calculate_distances()
    node_map[node.name] = node

  bloom = BloomFilter_32k()

def phases():
  GA_loop(False)
  previous_solution = parse_solution("phase2.txt")
  GA_loop(previous_solution)

if __name__ == '__main__':
  global_time = time.time()
  exit_time = global_time + 60 * 30

  filename = "fruitybun250.vrp"
  initialise_map(filename)
  phases()
  txt = 'phase2.txt'
  txt_opn = open(txt)
  print txt_opn.read()

