# Question 1:

__What are logits?__

  Logits are 'unnormalized' output values made by the last output layer before applying an activation function. Say your running a network for
  the MNIST dataset then your network could produce and output like this [2.3, 0.7, -0.8, 1.5, 3.3, -1.4, 0.3, 2.1, 0.9, -1.9]. Each number c-
  orrisponding to one of the 0-9 outputs. After this is done you might add an activation function to these. Which leads to the next question.
  
__What's the softmax function?__ 

  __The Softmax Function:__ 
        ![image](image1.svg)
        
  Using the same data from before you can apply the softmax function to a set of numbers and change them into probabilities that sum to 1.
  Also seen here -> ![image](image2.png)
