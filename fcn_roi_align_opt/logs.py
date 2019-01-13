import numpy as np

dice_loss = 0.99
for i in range(200):
    step = np.random.uniform(0.0001, 0.9, 1)[0]
    neg = np.random.randint(0, 3)
    pos = np.random.randint(2, 6)
    dice_loss
    print('Epoch {},  val dice loss: {}'.format(i+1, ) )
