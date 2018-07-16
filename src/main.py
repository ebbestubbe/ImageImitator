import numpy as np
from skimage import io
from skimage import draw
import random
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
    


if __name__ == "__main__":
	random_solve()