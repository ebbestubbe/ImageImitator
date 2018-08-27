import numpy as np
import random
import matplotlib.pyplot as plt
import time
import math
#Function objects: call takes a matrix of paramaters. shape[0] is the number of candidates, shape[1] is the number of dimensions

class func_eggholder():
    def __init__(self):
        self.name="Eggholder"
        self.n_calls=0
        self.n_dim = 2
        self.bound_lower = -512
        self.bound_upper = 512
    def call(self,x):
        assert(x.shape[1] == self.n_dim)
        """
        print((x >=self.bound_lower))
        if(not (x >=self.bound_lower).all()):
            print(x)

        print((x <=self.bound_upper))        
        if(not (x <=self.bound_upper).all()):
            print(x)
        """
        assert((x >=self.bound_lower).all())
        assert((x <=self.bound_upper).all())
        
        self.n_calls+=1*x.shape[0]
        return -(x[:,1]+47)*np.sin(np.sqrt(np.abs(x[:,0]/2+x[:,1]+47))) - x[:,0]*np.sin(np.sqrt(np.abs(x[:,0]-x[:,1]-47)))

class func_rosenbrock():
    def __init__(self,n_dim):
        self.name="Rosenbrock"
        self.n_calls = 0
        self.n_dim = n_dim
        self.bound_lower = -10
        self.bound_upper = 10
    def call(self,x):
        self.n_calls+=1*x.shape[0]
        n = x.shape[1] # number of dimensions
        s = np.zeros(x.shape[0])
        #s+=100*np.power(x[:,1] - np.power(x[:,0],2),2) + np.power((1-x[:,0]),2)
        #s+=100*np.power(x[:,2] - np.power(x[:,1],2),2) + np.power((1-x[:,1]),2)
        for i in range(n-1):
            s+=100*np.power(x[:,i+1] - np.power(x[:,i],2),2) + np.power((1-x[:,i]),2)
            
        return s
class func_sphere():
    def __init__(self,n_dim):
        self.name = "Sphere"
        self.n_dim = n_dim
        self.n_calls = 0
        self.bound_lower = -10
        self.bound_upper = 10
    def call(self,x):
        self.n_calls+=1*x.shape[0]
        return np.sum(np.power(x,2),axis=1)

class func_rastrigin():
    def __init__(self,n_dim):
        self.name = "Rastrigin"
        self.n_calls = 0
        self.n_dim = n_dim
        self.bound_lower = -5.12
        self.bound_upper = 5.12
    def call(self,x):
        self.n_calls+=1*x.shape[0]
        return 10*x.shape[1] + np.sum(np.power(x,2) - 10*np.cos(2*np.pi*x),axis=1)




def crossover(x0,x1):
    crosspoint = random.randint(0,x0.shape[0])
    c0 = np.concatenate([x0[:crosspoint], x1[crosspoint:] ])
    c1 = np.concatenate([x1[:crosspoint], x0[crosspoint:] ])
    return c0,c1

def GA(n_pop,n_iterations,func,mutation_scale = 0.1, mutation_add = 0.01):
    start = time.clock()
    func.n_calls = 0
    n_dim = func.n_dim
    A = np.random.rand(n_pop,n_dim)*(func.bound_upper-func.bound_lower) + func.bound_lower
    max_calls = n_iterations*n_pop
    
    fitness = np.empty(n_pop)
    best_y = np.inf
    best_ys = []
    best_xs = []
    n_calls = []
    for i in range(n_iterations):
        #evaluate fitness
        fitness = func.call(A)
        p = fitness.argsort()
        fitness = fitness[p]
        A = A[p]
        #log
        if(fitness[0] < best_y):
            best_y = fitness[0]
            best_xs.append(A[0])
            best_ys.append(fitness[0])
            n_calls.append(func.n_calls)
        #select and crossover
        n_select = 4*n_pop//10
        if(n_select%2 == 1):
            n_select+=1
        n_children = n_select
        n_replace = n_pop-n_children-n_select
        
        A_selected = A[:n_select,:]
        A_crossed = np.zeros(A_selected.shape)
        for j in range(n_select//2):
            c0, c1 = crossover(A_selected[j*2],A_selected[j*2+1])
            A_crossed[j*2,:] = c0
            A_crossed[j*2+1,:] = c1
        A_replace = np.random.rand(n_replace,n_dim)*(func.bound_upper-func.bound_lower) + func.bound_lower
        A = np.concatenate([A_selected,A_crossed,A_replace])
        mutate = np.random.choice(n_pop,n_pop//3,replace=False)
        #mutate by replacing with random candidate
        for j in mutate:
            if(j == 0):
                continue
            A[j,:] = A[j,:]*np.random.rand(n_dim)*((2*mutation_scale) - mutation_scale+1.0) + np.random.rand(n_dim)*(2*mutation_add) - mutation_add

            #A[j,:] += np.random.rand(n_dim)*(2*mutation_add) - mutation_add
            #A[j,:] = A[j,:]*np.random.rand(n_dim)*((2*mutation_scale) - mutation_scale+1.0)
        #for j in range(n_select//2):
        A = clamp(A,func.bound_lower,func.bound_upper)
    
    #evaluate for the last iteration as well
    fitness = func.call(A)
    p = fitness.argsort()
    fitness = fitness[p]
    A = A[p]
    if(fitness[0] < best_y):
        best_y = fitness[0]
        best_xs.append(A[0])
        best_ys.append(fitness[0])
        n_calls.append(func.n_calls)

    end = time.clock()
    #print("numpy took : " + str(end-start))
    
    #print(best_xs[-1])
    #print(best_y)
    #print(n_calls)
    #plt.figure()
    #plt.plot(n_calls,np.log(best_ys),'.')
    #plt.show()
    return best_xs[-1],n_calls, best_ys
def MOGA(n_pop,n_iterations,func,mutation_scale = 0.1, mutation_add = 0.01):
    start = time.clock()
    func.n_calls = 0
    n_dim = func.n_dim
    #func = func_rosenbrock()
    #make population, rows are population members, columns are dimensions
    A = np.random.rand(n_pop,n_dim)*(func.bound_upper-func.bound_lower) + func.bound_lower

    #get the fitness cumulative probability of the chromosomes eq (3)
    #If we assume no fitness collisions, we can do this.
    Cfi = np.arange(1,n_pop+1)/n_pop
    Cfi = Cfi[::-1]

    Cfisum = np.sum(Cfi)
    Cfi = Cfi.reshape(n_pop,1)
    alpha_i = 1-Cfi #probability to be chosen for mutation.
    n_mutation = np.int64(np.squeeze(np.ceil(alpha_i*n_dim))) # number of loci to undergo mutation
    
    best_y = np.inf
    best_ys = []
    best_xs = []
    n_calls = []
    
    for i in range(n_iterations):
        fitness = func.call(A)
        p = fitness.argsort() # get the sorting permutation        
        #Sort fitness and candidates
        fitness = fitness[p]
        A = A[p]
        #print("i: " + str(i))
        #print("candidate best: " + str(A[0]))
        #print("fitness best: " + str(fitness[0]))
        #print("fitness mean: " + str(np.average(fitness)))
        #print("fitness std: " + str(np.std(fitness)))

        if(fitness[0] < best_y):
            best_y = fitness[0]
            best_xs.append(A[0])
            best_ys.append(fitness[0])
            n_calls.append(func.n_calls)    
        dim_avg = np.average(A,axis=0)
        zero_avg = A-dim_avg

        power = np.power(zero_avg,2)
        power_scaled = np.multiply(power,Cfi)
        summed = np.sum(power_scaled,axis=0)
        sigma = np.sqrt(np.divide(summed,Cfisum))
        #print("sigma: " + str(sigma))

        #find the sorting of the dimensions, such that the least informative(highest sigma) is "first"
        #We want to do this without actually reordering the array
        sigma_order = (-sigma).argsort() #argsort is ascending. Sorting by negative gives descending

        randoms = np.random.rand(n_pop) #random numbers to compare with
        true_if_mutate = np.less(randoms,alpha_i.reshape(n_pop,)) #True/false array of which to mutate
        indices_to_mutate = np.atleast_1d(np.squeeze(np.argwhere(true_if_mutate))) #indices to mutate
        #print(sigma)
        for j in indices_to_mutate:
            for k in range(n_mutation[j]):
                #A[j,sigma_order[k]] = random.random()*(func.bound_upper-func.bound_lower) + func.bound_lower #replace with new member(is this how we mutate?)
                #This seems counterintuitive. But if we add random noise / scale the candidate, the standard deviation increases, making it more likely to increase again. making candidates bigger/smaller than before, eventually leading to overflow. There needs to be some mechanic that "kills" the dimensions or candidates running amok. perhaps just clipping to min/max?, what if we dont know min/max?
                
                A[j,sigma_order[k]] *= random.gauss(1.0,sigma[sigma_order[k]])
                A[j,sigma_order[k]] += random.gauss(0.0,sigma[sigma_order[k]])
                
        A = clamp(A,func.bound_lower,func.bound_upper)
    fitness = func.call(A)
    p = fitness.argsort()
    fitness = fitness[p]
    A = A[p]
    if(fitness[0] < best_y):
        best_y = fitness[0]
        best_xs.append(A[0])
        best_ys.append(fitness[0])
        n_calls.append(func.n_calls)
    #print(best_xs[-1])
    #print(best_y)
    #print(n_calls)
    end = time.clock()
    #print("MOGA runtime : " + str(end-start))
    return best_xs[-1],n_calls, best_ys
def test(func):
    n_pop = 100
    #n_dim = 10
    n_iterations = 10

    mutation_scale = 0.5
    mutation_add = .1
    GA_x, GA_calls, GA_y = GA(n_pop,n_iterations,func = func,mutation_scale = mutation_scale,mutation_add=mutation_add)
    MOGA_x,MOGA_calls, MOGA_y = MOGA(n_pop,n_iterations, func=func,mutation_scale = mutation_scale,mutation_add=mutation_add)
    print(func.name)
    print("GA: " + str(GA_y[-1]))
    print("MOGA: " + str(MOGA_y[-1]))
    print("")
    """    
    plt.figure()
    plt.plot(GA_calls,np.log10(GA_y),'r.')
    plt.plot(MOGA_calls,np.log10(MOGA_y),'b.')
    plt.show()
    """

#Sets all the values in A below 'lower' set to lower. Likewise for 'upper'
def clamp(A,lower,upper):
    minwith = np.ones(A.shape)*upper
    maxwith = np.ones(A.shape)*lower
    B = np.minimum(A,minwith)
    B = np.maximum(B,maxwith)
    return B

def main():
    n_dim = 2
    test(func_rastrigin(n_dim = n_dim))
    test(func_sphere(n_dim = n_dim))
    test(func_eggholder())
    
    
    #call()
    #manual_call()

    """
    n_pop = 1000
    n_dim = 10
    n_iterations = 100

    mutation_scale = 0.5
    mutation_add = .1
    func = func_rastrigin()
    GA_x, GA_calls, GA_y = GA(n_pop,n_dim,n_iterations,func = func,mutation_scale = mutation_scale,mutation_add=mutation_add)
    MOGA_x,MOGA_calls, MOGA_y = MOGA(n_pop,n_dim,n_iterations, func=func,mutation_scale = mutation_scale,mutation_add=mutation_add)
    print("GA: " + str(GA_y[-1]))
    print("MOGA: " + str(MOGA_y[-1]))
    """
    

if __name__ == "__main__":
    """
    func = func_eggholder() 

    a = np.array([[3.1249,1.3942],[1,0],[1,1]])
    print(func.call(a))
    b = np.array([[1.3942,3.1249],[-2,0],[-3,1]])
    print(func.call(b))
    print(func.n_calls)
    """
    main()	