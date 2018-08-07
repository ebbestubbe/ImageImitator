import numpy as np
from skimage import io, draw
from image import Image
from circle import Circle, generate_random_circle
from circles import Circles
import random
import time
import matplotlib.pyplot as plt
"""
def make_image(shape,parameters,colors):
    image = np.ones(shape)*0.5

    for i in range(parameters.shape[0]):
        p0,p1,r = parameters[i,:]
        rr,cc = draw.ellipse(p0,p1,r,r,shape)
        image[rr,cc] += colors[i]
    image = normalize_image(image)
    return image

def normalize_image(image):
    #maxwith = np.zeros(shape)
    #image = np.maximum(image,maxwith)

    image = image - np.amin(image)
    image = image/np.amax(image)
    return image
""" 
def draw_circle(image,p0,p1,r,color):
    imagecopy = np.copy(image)
    rr,cc = draw.ellipse(p0,p1,r,r,imagecopy.shape)
    imagecopy[rr,cc] += color
    minwith = np.ones(imagecopy.shape)
    maxwith = np.zeros(imagecopy.shape)
    imagecopy = np.minimum(imagecopy,minwith)
    imagecopy = np.maximum(imagecopy,maxwith)
    
    return imagecopy
	#print(image)

def testbw():

	testfolder = 'testimages\\'
	imagename = 'bwtest.png'
	image = io.imread(testfolder + imagename,as_gray = True)

	image = draw_circle(image,10,10,5,0.2)
	image = draw_circle(image,15,20,5,0.9)

	io.imsave(testfolder + imagename + '_mod.png',image)

def make_error_image(target,trial):
    error = np.power(target-trial,2.0)
    return error

def sum_image(image):
    return np.sum(image)

def get_error(target,trial):
    error_image = make_error_image(target,trial)
    return sum_image(error_image)

def test_zero_error():
    testfolder = 'testimages\\'
    targetname = 'matchtarget.png'
    trialname = 'matchtrial_0error.png'
    target = io.imread(testfolder + targetname,as_gray=True)
    trial = io.imread(testfolder + trialname, as_gray= True)

    error_image = make_error_image(target,trial)
    total_error = sum_image(error_image)
    print("total error: " + str(total_error))
    io.imsave(testfolder + 'error_zero_image.png',error_image)

def test_error():
    testfolder = 'testimages\\'
    targetname = 'matchtarget.png'
    trialname = 'matchtrial.png'
    target = io.imread(testfolder + targetname,as_gray=True)
    trial = io.imread(testfolder + trialname, as_gray= True)

    error_image = make_error_image(target,trial)
    total_error = sum_image(error_image)
    print("total error: " + str(total_error))
    io.imsave(testfolder + 'error_image.png',error_image)

def random_solve():
    testfolder = 'testimages\\'
    targetname = 'Rose.jpeg'
    target = io.imread(testfolder + targetname, as_gray=True)
    trial = np.zeros(target.shape)
    iterations = 10000
    besterror = target.shape[0]*target.shape[1]
    print(besterror)
    for i in range(iterations):
        print(i)
        for j in range((iterations-i)//500 + 1):

            p0 = random.randint(0,target.shape[0])
            p1 = random.randint(0,target.shape[1])
            radius = random.randint(1,max(target.shape))
            color = random.random()*2.0-1.0
            new_trial = draw_circle(trial,p0,p1,radius,color)
        
        new_trial_error = sum_image(make_error_image(target,new_trial))
        if(new_trial_error < besterror):
            print("improved to: " + str(new_trial_error))
            trial = new_trial
            besterror = new_trial_error
    #print(trial)

    io.imsave(testfolder + 'solution_rose2.png',trial)
    io.imsave(testfolder + 'solution_error_rose2.png',make_error_image(target,trial))    
    

def test_full_creation():
    testfolder = 'testimages\\'
    targetname = 'matchtarget.png'
    target = io.imread(testfolder + targetname, as_gray=True)
    start_time = time.time()
    
    n_circles = 10000
    trial_image = Image(target.shape)    
    for i in range(n_circles):
        p0 = random.randint(0,target.shape[0])
        p1 = random.randint(0,target.shape[1])
        r = random.randint(1,max(target.shape))
        color = random.random()*2-1.0
        trial_image.add_circle(Circle(p0,p1,r,color))
        
    image_to_save = trial_image.get_normalized_image()
    end_time = time.time()-start_time
    print("end_time: " + str(end_time))
    io.imsave(testfolder + 'trial_build.png',image_to_save)

def random_solve_build():
    testfolder = 'testimages\\'
    targetname = 'Rose.jpeg'
    target = io.imread(testfolder + targetname, as_gray=True)
    besterror = target.shape[0]*target.shape[1]
    
    n_circles = 10
    trial = Image(target.shape)
    it = []
    errors = []
    for j in range(10):
        print("adding: " + str(j))
        n_before = trial.ncircles
            
        for i in range(n_circles):
                
            circle = generate_random_circle(target.shape)
            trial.add_circle(circle)
            trial_normalized = trial.get_normalized_image()
            error = get_error(trial_normalized,target)
            if(error<besterror):
                print("improved to: " + str(error))
                #it.append(i)
                #errors.append(error)
                besterror = error
            else:
                trial.pop_circle()
        n_after = trial.ncircles()
        
        print("removing:")
        if(n_after != n_before):
            for i in reversed(range(trial.ncircles())):
                circle = trial.pop_circle(i);

                trial_normalized = trial.get_normalized_image()
                error = get_error(trial_normalized,target)
                if(error<besterror):
                    print("improved to: " + str(error))    
                    besterror = error
                else:
                    trial.add_circle(circle)
        """
        for i in reversed(range(len(trial.circles))):
            trial_normalized = trial.get_normalized_image()
            error = get_error(trial_normalized,target)
            
            c_old = trial.pop_circle(i)
            trial_normalized = trial.get_normalized_image()
            error_smaller = get_error(trial_normalized,target)
            
            while(c_old.r>2):
        """

        
        
    print("number of circles: "  + str(trial.ncircles()))
    #plt.plot(it,errors,'.')
    #plt.savefig(testfolder + 'errortrace_rose.png')
    io.imsave(testfolder + 'random_solution_rose.png',trial.get_normalized_image())
    io.imsave(testfolder + 'randomsolution_error_rose.png',make_error_image(target,trial.get_normalized_image()))    

def shuffle_solver():
    testfolder = 'testimages\\'
    targetname = 'Rose.jpeg'
    target = io.imread(testfolder + targetname, as_gray=True)
    besterror = target.shape[0]*target.shape[1]
    
    n_circles = 5
    iterations = 4
    
    it = []
    errors = []
    circles = []
    for i in range(n_circles):
        circles.append(generate_random_circle(target.shape))

    trial = Image(target.shape,circles)

    for i in range(iterations):
        c_old = trial.pop_circle(random.randint(0,n_circles-1))
        
        p0 = random.randint(0,target.shape[0])
        p1 = random.randint(0,target.shape[1])
        c = Circle(p0,p1,c_old.r,c_old.color)
        trial.add_circle(c)

        trial_normalized = trial.get_normalized_image()
        error = get_error(trial_normalized,target)
        if(error<besterror):
            print("improved to: " + str(error))
            it.append(i)
            errors.append(error)
            besterror = error
        else:
            trial.pop_circle()
            trial.add_circle(c_old)
    print("number of circles: "  + str(len(trial.circles)))
    plt.plot(it,errors,'.')
    plt.savefig(testfolder + 'shuffle_errortrace_rose.png')
    io.imsave(testfolder + 'shuffle_solution_rose.png',trial.get_normalized_image())
    io.imsave(testfolder + 'shuffle_solution_error_rose.png',make_error_image(target,trial.get_normalized_image()))

def genetic_algorithm_solver():
    testfolder = 'testimages\\'
    targetname = 'Rose.jpeg'
    target = io.imread(testfolder + targetname, as_gray=True)

    n_pop = 10
    n_circles = 300
    trials = [None for i in range(n_pop)]
    fitness = [None for i in range(n_pop)]
    #for j in range(n_pop):
    for j in range(n_pop):
        circles = Circles()
        for i in range(n_circles):
            circles.add_circle(generate_random_circle(target.shape))
        trial = Image(target.shape,circles,background = np.sum(target)/(target.shape[0]*target.shape[1]))
        trials[j] = trial
        #fitness.append(get_error(target,trial.get_normalized_image()))
        #print(fitness[j])
    #io.imsave(testfolder + 'genetic_0.png',trials[0].get_normalized_image())
    #io.imsave(testfolder + 'genetic_1.png',trials[-1].get_normalized_image())
        

    #population = []

    #initialize population
    max_iterations = 1000
    for i in range(max_iterations):
        print("i: " + str(i))
        
        if(i%10 == 0):
            io.imsave(testfolder + 'genetic_rose_' + str(i) +'.png',trials[0].get_normalized_image2())
        #evaluate fitness
        for j in range(n_pop):
            fitness[j] = get_error(target,trials[j].get_image())
        fitness,trials = (list(t) for t in zip(*sorted(zip(fitness,trials))))
        
        print("fit best: " + str(fitness[0]) + "; fit worst: " + str(fitness[-1]) + "fit avg: " + str(np.mean(fitness)))
        #print("fit: " + str(fitness[0]))
        
        #select
        new_trials = []
        n_select = 4*n_pop//10
        if(n_select%2 == 1):
            n_select+=1
        n_children = n_select
        n_keep = n_pop-n_children-n_select
        

        #crossover    
        for j in range(n_select//2):
            #print("cross 1 " + str(2*j))
            #print("cross 2 " + str(2*j+1))
            child0,child1 = crossover(trials[2*j],trials[2*j+1])
            new_trials.append(trials[2*j])
            new_trials.append(trials[2*j+1])
            new_trials.append(child0)
            new_trials.append(child1)
            
        for j in range(n_select,n_select + n_keep):
            #print("keep " + str(j)) #first
            new_trials.append(trials[j])
        #for j in range(n_select + n_keep,n_children + n_select + n_keep):
            #print("children " + str(j)) #first
        
        mutate = np.random.choice(n_pop,n_pop//3,replace=False)
        for j in mutate:
            if(j == 0):
                continue
            for circle in new_trials[j].circles.circles:
                circle.r+= random.randint(0,2)*2-1
                circle.r = min([max([circle.r,1]),30])
                circle.color = circle.color*random.choice([0.95,1.05])
        
        trials = new_trials
    
    for j in range(n_pop):
        fitness[j] = get_error(target,trials[j].get_image())
    fitness,trials = (list(t) for t in zip(*sorted(zip(fitness,trials))))
        
    io.imsave(testfolder + 'genetic_rose_worst.png',trials[-1].get_normalized_image2())
    
    io.imsave(testfolder + 'genetic_rose_best.png',trials[0].get_normalized_image2())



def crossover(image0, image1):
    #print(len(image0.circles.circles))
    #point = random.randint(0,len(image1.circles.circles))
    circles_child0 = []
    circles_child1 = []
    for i in range(len(image0.circles.circles)):
        flip = random.randint(0,2)
        if(flip == 0):
            circles_child0.append(image0.circles.circles[i])
            circles_child1.append(image1.circles.circles[i])
        else:
            circles_child0.append(image1.circles.circles[i])
            circles_child1.append(image0.circles.circles[i])
    #make Circles wrapper:
    circles_child0_wrapper = Circles(circles_child0)
    circles_child1_wrapper = Circles(circles_child1)
    
    child0 = Image(image0.get_shape(),circles_child0_wrapper)
    child1 = Image(image0.get_shape(),circles_child1_wrapper)
    return child0,child1
    
if __name__ == "__main__":
	#shuffle_solver()
    #random_solve_build()
    genetic_algorithm_solver()
    #test_full_creation()
    #genetic_algorithm_solver()