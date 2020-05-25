## Queries about Endgame assignment

1. What action space to take? Whether to take action pasce like session 7 code, angle of rotation or something else.

2. What state space to take? Just current state image or more than one consecutive images or we should include some other things like orientation as in session  7code.

3. In case of critic, how to combine state with action for input to critic network. Because state will have an image while action will be a float 


## WorkFlow

1. Design environment  
    a. render(): In this step we will have to figure out how to plot car image over citymap.png taking into account car's rotation and coordinates(location). As suggested by Rohan we can use opencv library to do this.  
    
    b. step(a) : In this step given any action car should take that action and calculate new coordinate(location) and rotation. Later render() should be able to create visualization. In session 7 code, car coordinate calculation, angle calculation, car movement calculation etc is already available we just have to verify coordinate systems because for kivy origin is lower-left corner while opencv,numpy has upper-left corner as origin so accordingly all the calculations will change.
    
    c. verify (a) and (b) : We can easily verify working of steps (a) and (b) by randomly taking actions among say (+5 deg,0 deg, -5 deg) rotation. We should be able to generate consistent movement of car over the citymap.png canvas similar to what we saw in kivy.
    
    d. reward system: We will have to figure out entire reward system. We can follow same system as session 7 code. In that case we should implement methods to detect road, sand, wall, goal etc. Reward values we may have to adjust later seeing how our algo performs
    
    e. episodes & reset: If you check the codes shared with us, we have episodes. Eg. in case of walker2d if the walker falls down episode ends and env.step() returns done as True. We should also have similar episode system for endgame assignment. eg. car gets onto sand and doesn't come out for say 100 steps or car is stuck to the wall for say 50 steps we can end the episode. We must also have a reset() method which will reset car to on location which on road.
    
2. TD3 design:
    a. states selection: We must figure out what states to feed into TD3 networks. Rohan gave some hints in the lecture. We can take current frame and distance to target as state. 
    b. state representation: Rohan said we can have a crop of say 80x80 out of MASK1.png with car location as center of crop. Over this crop we can plot some kind of arrow or any minimalistic representation to indicate car's rotation (which way car is facing). He also said that since MASK1.png has only two values we can represent state using a boolean numpy array with say True representing road and False representing sand. Doing this will save lots of space and training will also be fast. We will have to figure out what works for us.
    c. action space: This seems to be straight forward. Rohan clearly mentioned in lecture that taking angle as action is good enough. So we can take 1D action representing angle. This will have continuous float value. We will have to decide max_action also which we can set to 3 or 5 degrees.
    d. model selection: Rohan said any simple model which can achieve upto 99% accuracy on MNIST should be good. 
3. Miscellaneous:  
    
    a. Video creation: final output video
    b. Model inference : No training just evaluation


### Resources

#### Useful repos

1. https://github.com/lzhan144/Solving-CarRacing-with-DDPG?files=1. 
This repo solves CarRacing-V0 from gym environment. It is similar to our problem statement. Input state is an image  while output is actions in continuous space.


