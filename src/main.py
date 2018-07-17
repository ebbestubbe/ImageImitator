import numpy as np
from skimage import io, draw
from image import Image
from circle import Circle, generate_random_circle
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
    
    n_circles = 10000
    trial = Image(target.shape)
    #for j in range(10):
    #print("adding: " + str(j))
    #n_before = len(trial.circles)
    it = []
    errors = []
    for i in range(n_circles):
        circle = generate_random_circle(target.shape)
        trial.add_circle(circle)
        trial_normalized = trial.get_normalized_image()
        error = get_error(trial_normalized,target)
        if(error<besterror):
            print("improved to: " + str(error))
            it.append(i)
            errors.append(error)
            besterror = error
        else:
            trial.pop_circle()
    #n_after = len(trial.circles)
        """
        print("removing:")
        if(n_after != n_before):
            for i in reversed(range(len(trial.circles))):
                circle = trial.pop_circle(i);

                trial_normalized = trial.get_normalized_image()
                error = get_error(trial_normalized,target)
                if(error<besterror):
                    print("improved to: " + str(error))    
                    besterror = error
                else:
                    trial.add_circle(circle)
        """
    print("number of circles: "  + str(len(trial.circles)))
    plt.plot(it,errors,'.')
    plt.savefig(testfolder + 'errortrace_rose.png')
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

if __name__ == "__main__":
	shuffle_solver()
    #random_solve_build()
    #test_full_creation()