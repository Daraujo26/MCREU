# June 14, 2023

__Break down the following line of code:__

    1: x = rearrange(x, '(n1 n2) h w c -> (n1 h) (n2 w) c', n1=2, n2=2)

The rearrange function, a function declared in the einops package, is used to __reposition__ multi-dimentional tensors. As stated in the einops <a href="https://einops.rocks/api/rearrange/">definition</a>, rearrange is used to transpose, reshape, stack, etc. While taking three parameters: <i> tensor, pattern, and axes.</i> 
In the code above, the function takes the four dimensional tensor (x), and reshapes or transposes it into a three dimensional tensor to prepare for cases such as image display, model training, and more. The goal in the code was to display 4 images, using matplotlib, in a 2x2 grid.
The pattern parameter aka. '(n1 n2) h w c -> (n1 h) (n2 w) c', 'labels' the current four dimensions of x as (n1 n2), h, w, c. Note that respectively, n1=2, n2=2, h=32, w=32, and c=3. n1 & n2 are declared (2x2 grid), h=height (32 pixels), w=width (32 pixels), and c=channels (Red, Green, Blue). Then based off those values the tensor gets reordered into the declared format at the end of the pattern string
(after the arrow, (n1 h) (n2 w) c). This then returns your new tensor, a three dimensional matrix. Notice there is no rearrange axes parameter since were using the defaulted zero axes.

__Example for four images:__

    1: import torchvision
    2: import matplotlib.pyplot as plt
    3: import numpy as np
    4: from einops import rearrange
    5: trainset = torchvision.datasets.CIFAR10(root='.', train=True, download=True)
    6: a, b, c, d = [trainset[i][0] for i in range(4)]
    7: x = np.stack((a, b, c, d))
    8: x.shape
    out: (4, 32, 32, 3)
    9: x = rearrange(x, '(n1 n2) h w c -> (n1 h) (n2 w) c', n1=2, n2=2)
    10: x.shape
    out: (64, 64, 3)
  Now you can show your 4 2x2 images...
  
    11: plt.imshow(x)
    out: <matplotlib.image.AxesImage at 0x7fa47d909f10>
    12: plt.show()
