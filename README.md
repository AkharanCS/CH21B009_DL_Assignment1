sample

# GD
num_iterations = 1000
lr = 0.01
for i in range(num_iterations):
    Loss = 0
    dw = list(nn1.weight_grad_cumm)
    db = list(nn1.bias_grad_cumm)
    for j in range(1000):
        x = train_images[j].reshape(28*28,1)/255
        y = train_labels[j].reshape(10,1)
        yhat = nn1.forward_pass(x)
        Loss += loss(y,yhat)
        w_grad_now,b_grad_now = nn1.backpropagation(x,y)
        for l in range(len(w_grad_now)):
            dw[l] += w_grad_now[l]
            db[l] += b_grad_now[l]
    for k in range(len(nn1.weights)):
        nn1.weights[k] -= lr*dw[k]
        nn1.bias[k] -= lr*db[k]
    if i%10 == 0:
        print(f"Iteration {i}, Cost: {Loss}")