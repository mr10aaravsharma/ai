
---------------------------------------------dfs
def dfs(graph, closed, open, start_node, goal_node):
    path = []
    open.append(start_node)

    while open:
        extracted_node = open.pop(0)
        path.append(extracted_node)
        print(extracted_node)

        if extracted_node == goal_node:
            closed.append(extracted_node)
            return f'Goal reached with path : {path}', open, closed

        open = graph[extracted_node] + open
        closed.append(extracted_node)
        print('Current open list :' ,open)
        print('Current closed list :' ,closed)

        for neighbour_node in graph[extracted_node] :
            if neighbour_node not in graph.keys():
                return (f'The node {neighbour_node} does not exist')
                
    return "Goal node does not exist"

graph1 = {
5 : [3, 7],
3 : [2, 4],
7 : [8],
2 : [],
4 : [8],
8 : []
}
closed1 = []
open1 = []
ans = dfs(graph1, closed1, open1, 5, 8)
print(ans[0])
print('Final open list is : ',ans[1])
print('Final closed list is : ',ans[2])




---------------------------------------------------bfs
def bfs(graph, closed, open, start_node, goal_node):
    path = []
    open.append(start_node)

    while open:
        extracted_node = open.pop(0)
        path.append(extracted_node)
        closed.append(extracted_node)
        print(extracted_node)

        if extracted_node == goal_node:
            return f'Goal reached with path : {path}', open, closed

        for neighbour_node in graph[extracted_node]:
            if neighbour_node not in graph.keys():
                print(f'The node {neighbour_node} does not exist')
                continue
            if neighbour_node not in closed:
                open.append(neighbour_node)
        print('Current open list :' ,open)
        print('Current closed list :' ,closed)
    return "Goal node does not exist"

graph1 = {
5 : [3, 7],
3 : [2, 9, 4],
7 : [8],
2 : [],
4 : [2],
8 : []
}
closed1 = []
open1 = []
ans = bfs(graph1, closed1, open1, 5, 8)
print(ans[0])
print('Final open list is : ',ans[1])
print('Final closed list is : ',ans[2])






-------------------------------------------------------------------dfid
def iterative_deepening_dfs(graph, start, target):
    depth = 1
    bottom_reached = False

    while not bottom_reached:
        result, bottom_reached = iterative_deepening_dfs_rec(graph, start, target, 0, depth)

        if result is not None:
            return result

        depth *= 2
        print("Increasing depth to " + str(depth))

    return None

def iterative_deepening_dfs_rec(graph, node, target, current_depth, max_depth):
    print("Visiting Node " + str(node))
    
    if node == target:
        print("Found the node we're looking for!")
        return node, True

    if current_depth == max_depth:
        print("Current maximum depth reached, returning...")

        if len(graph[node]) > 0:
            return None, False
        else:
            return None, True

    bottom_reached = True

    for i in range(len(graph[node])):
        result, bottom_reached_rec = iterative_deepening_dfs_rec(graph, graph[node][i], target, current_depth + 1, max_depth)

        if result is not None:
            return result, True

        bottom_reached = bottom_reached and bottom_reached_rec
        
    return None, bottom_reached

graph1 = {
5 : [3, 7],
3 : [2, 4],
7 : [8],
2 : [],
4 : [8],
8 : []
}
iterative_deepening_dfs(graph1,5,8)






------------------------------------------------------------astar
from collections import deque
class Graph:
   def __init__(self, adjac_lis):
       self.adjac_lis = adjac_lis
   def get_neighbors(self, v):
       return self.adjac_lis[v]
   # This is heuristic function which is having equal values for all nodes
   def h(self, n):
       H = {
           'A': 1,
           'B': 1,
           'C': 1,
           'D': 1
       }
       return H[n]
   def a_star_algorithm(self, start, stop):
       # In this open_lst is a lisy of nodes which have been visited, but who's
       # neighbours haven't all been always inspected, It starts off with the start
 #node
       # And closed_lst is a list of nodes which have been visited
       # and who's neighbors have been always inspected
       open_lst = set([start])
       closed_lst = set([])
       # poo has present distances from start to all other nodes
       # the default value is +infinity
       poo = {}
       poo[start] = 0
       # par contains an adjac mapping of all nodes
       par = {}
       par[start] = start
       while len(open_lst) > 0:
           n = None
           # it will find a node with the lowest value of f() -
           for v in open_lst:
               if n == None or poo[v] + self.h(v) < poo[n] + self.h(n):
                   n = v;
           if n == None:
               print('Path does not exist!')
               return None
           # if the current node is the stop
           # then we start again from start
           if n == stop:
               reconst_path = []
               while par[n] != n:
                   reconst_path.append(n)
                   n = par[n]
               reconst_path.append(start)
               reconst_path.reverse()
               print('Path found: {}'.format(reconst_path))
               return reconst_path
           # for all the neighbors of the current node do
           for (m, weight) in self.get_neighbors(n):
             # if the current node is not presentin both open_lst and closed_lst
               # add it to open_lst and note n as it's par
               if m not in open_lst and m not in closed_lst:
                   open_lst.add(m)
                   par[m] = n
                   poo[m] = poo[n] + weight
               # otherwise, check if it's quicker to first visit n, then m
               # and if it is, update par data and poo data
               # and if the node was in the closed_lst, move it to open_lst
               else:
                   if poo[m] > poo[n] + weight:
                       poo[m] = poo[n] + weight
                       par[m] = n
                       if m in closed_lst:
                           closed_lst.remove(m)
                           open_lst.add(m)
           # remove n from the open_lst, and add it to closed_lst
           # because all of his neighbors were inspected
           open_lst.remove(n)
           closed_lst.add(n)
       print('Path does not exist!')
       return None
 
adjac_lis = {
   'A': [('B', 1), ('C', 3), ('D', 7)],
   'B': [('D', 5)],
   'C': [('D', 12)]
}
graph1 = Graph(adjac_lis)
graph1.a_star_algorithm('A', 'D')






-------------------------------------------------hill climbing
import copy
 
visited_states = []
 
# heuristic fn - number of misplaced blocks as compared to goal state
def heuristic(curr_state,goal_state):
   goal_=goal_state[3]
   val=0
   for i in range(len(curr_state)):
       check_val=curr_state[i]
       if len(check_val)>0:
           for j in range(len(check_val)):
               if check_val[j]!=goal_[j]:
                   # val-=1
                   val-=j
               else:
                   # val+=1
                   val+=j
   return val
 
# generate next possible solution for the current state
def generate_next(curr_state,prev_heu,goal_state):
   global visited_states
   state = copy.deepcopy(curr_state)
   for i in range(len(state)):
       temp = copy.deepcopy(state)
       if len(temp[i]) > 0:
           elem = temp[i].pop()
           for j in range(len(temp)):
               temp1 = copy.deepcopy(temp)
               if j != i:
                   temp1[j] = temp1[j] + [elem]
                   if (temp1 not in visited_states):
                       curr_heu=heuristic(temp1,goal_state)
                       # if a better state than previous state of found
                       if curr_heu>prev_heu:
                           child = copy.deepcopy(temp1)
                           return child
  
   # no better soln than current state is possible
   return 0
 
def solution_(init_state,goal_state):
   global visited_states
 
   # checking if initial state is already the final state
   if (init_state == goal_state):
       print (goal_state)
       print("solution found!")
       return
  
   current_state = copy.deepcopy(init_state)
  
   # loop while goal is found or no better optimal solution is possible
   while(True):
 
       # add current state to visited to avoid repetition
       visited_states.append(copy.deepcopy(current_state))
 
       print(current_state)
       prev_heu=heuristic(current_state,goal_state)
 
       # generate possible better child from current state
       child = generate_next(current_state,prev_heu,goal_state)
      
       # No more better states are possible
       if child==0:
           print("Final state - ",current_state)
           return
          
       # change current state to child
       current_state = copy.deepcopy(child) 
 
def solver():
   # maintaining a global visited to save all visited and avoid repetition & infinite loop condition
   global visited_states
   # inputs
   init_state = [[],[],[],['B','C','D','A']]
   goal_state = [[],[],[],['A','B','C','D']]
   # goal_state = [[],[],[],['A','D','C','B']]
   solution_(init_state,goal_state)
 
solver()





--------------------------------------------perceptron
import numpy as np

X1 = np.array([1, -2, 0, -1], dtype=np.float)
X2 = np.array([0, 1.5, -0.5, -1], dtype=np.float)
X3 = np.array([-1, 1, 0.5, -1], dtype=np.float)

X = np.array([X1, X2, X3], dtype=np.float)
W = np.array([1, -1, 0, 0.5], dtype=np.float)

d = np.array([-1, -1, 1], dtype=np.float)

c = 0.1
epochs = 1

for i in range(epochs):
    print("Iteration ", i+1)
    for j in range(len(X)):
        net = np.dot(X[j], W)
        
        if (net <= 0):
            op = -1
        elif net > 0:
            op = 1
        
        error = d[j] - op
       
        dW = c*error*X[j]
        W += dW
        
    print("W after ", i+1, " epochs ", W)    
print("Final W after ", epochs, "epochs:")
print(W)





---------------------------------------------genetic
import random

def genetic(p):
    x=[]
    fx=[]
    for i in range(4):
        s= p[i]
        count= s.count("1")
        x.append(count)
        val= count/10.0
        fx.append(val)
    total= sum(fx)
    avg= total/4.0
    real=[]
    exp=[]
    actual_count=[]
    for i in range(4):
        r= fx[i]/total
        e= fx[i]/avg
        real.append(r)
        exp.append(e)
        ac= round(e)
        actual_count.append(ac)
    print("\nInitial Pop\t x\t f(x)\t f(x)/sum\t f(x)/avg\t Actual Count")
    for i in range(4):
        print(p[i]," "+str(x[i])," "+str(fx[i]),"\t%.3f" %real[i],"\t%.3f\t"%exp[i],actual_count[i], sep=" ")
    min_count= min(exp)
    min_index= exp.index(min_count)
    print("String {} with count {} is rejected\n".format(p[min_index],actual_count[min_index]))
    max_count= max(exp)
    max_index= exp.index(max_count)
    p[min_index]=p[max_index]
    x[min_index]=x[max_index]
    fx[min_index]=fx[max_index]
    real[min_index]=real[max_index]
    exp[min_index]=exp[max_index]
    actual_count[min_index]=exp[max_index]
    return mate(p)
 
def mate(p):
    selection= [0,1,2,3]
    selected= [0,0,0,0]
    mates={}
    first= random.randint(1, 3)
    mates[0]= first
    selected[0]=first
    selected[first]=0
    del selection[0]
    del selection[first-1]
    mates[selection[0]]=selection[1]
    selected[selection[0]]= selection[1]
    selected[selection[1]]= selection[0]
    return crossover(p,mates,selected)
 
def crossover(p,mates,selected):
    new_pop=p.copy()
    for k in mates.keys():
        first= p[k]
        s1_p1= first[0:5]
        s1_p2= first[5:10]
        second= p[mates[k]]
        s2_p1= second[0:5]
        s2_p2= second[5:10]
        str1= s1_p1+s2_p2
        str2= s2_p1+s1_p2
        new_pop[k]=str1
        new_pop[mates[k]]=str2
    print("\n.....Performing Crossover. ... ")
    print("\nPopulation Mate \tCrossover Site \tNew Pop x f(x)")
    for i in range(4):
        print(p[i],selected[i],5," "+new_pop[i],new_pop[i].count("1"),new_pop[i].count("1")/10.0,sep=" ")
    print()
    return mutation(new_pop)
 
def mutation(p):
    before= p.copy()
    loc= random.randint(0,3)
    initial= p[loc]
    mutation= initial.replace("0","1",1)
    p[loc]=mutation
    print("\n.....Performing Mutation .... \n")
    print("\nDue to mutation String {} in New Pop becomes {}\n".format(initial,mutation))
    print("\nBefore mutation x1 \tf(x1)\tAfter mutation \tx2 \tf(x2)")
    k=0.0
    for i in range(4):
        print(before[i]," "+str(before[i].count("1")),before[i].count("1")/10.0,p[i]," "+str(p[i].count("1")),p[i].count("1")/10.0,sep=' ')
        k=k+p[i].count("1")/10.0
    print("Fitness: ",k)
    return p
 
p1= ["0000011100","1000011111","0110101011","1111111011"]
print("------------------First Generation -------------------")
p2= genetic(p1)
print("\n------------------Second Generation -------------------")
p3= genetic(p2)






---------------------------------prolog
male(sarthak).
male(sunil).
male(sahat).
male(anil).
male(motilal).

female(meenakshi).
female(phoola).
female(tanvi).

parent(sunil, sarthak).
parent(meenakshi, sarthak).

parent(sunil, sahat).
parent(meenakshi, sahat).

parent(motilal, anil).
parent(phoola, anil).

parent(motilal, sunil).
parent(phoola, sunil).

parent(anil, tanvi).

father(X,Y):- parent(X,Y), male(X).
mother(X,Y):- parent(X,Y), female(X).
brother(X,Y):- parent(Z,X), parent(Z,Y), X\==Y, male(X).
grandmother(X,Y):- mother(X,Z), parent(Z,Y).
uncle(X,Y):- brother(X,Z), father(Z,Y).
wife(X,Y):- parent(X,Z), parent(Y,Z), female(X).
cousin(X,Y) :- parent(Z,X), parent(W,Y), brother(Z,W), Z\==W.



