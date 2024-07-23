
def gradient_descent(x,y,w_in,b_in,lr,num_iters,cost_function,gradient_function):
    '''
    Performs Gradient Descent to fit w,b and update them within num_iters gradient steps with a learning rate lr.

    Args:
    x : training data, there are m data points
    y : target values
    w_in, b_in : initial values
    lr : alpha or learning rate of the algorithm
    num_iters : number of iterations or steps to be taken
    cost_function : function call to calculate the cost function at every step of the gradient descent
    gradient_function : function call to produce the gradient

    Returns:
    w : updated value of w after running gradient descent
    b : updated value of b after running gradient descent
    J_hist : histroy of cost function values
    p_hist : histroy of all parameter values during every iteration
    '''

    J_hist,p_hist = [],[]
    w = w_in       # inital value for the weight given to Feature 1
    b = b_in       # inital value of interecept of the linear regression line

    for i in range(num_iters):           # performing for fixed number of iterations

        # calculating the derivatives of the cost function wrt w and b
        dj_dw,dj_db = gradient_function(x,y,w,b)

        w = w -lr * dj_dw
        b = b - lr * dj_db

        if i<1000:      # a maximum of 1000 iterations
            J_hist.append(cost_function(x,y,w,b))
            p_hist.append([w,b])

        if i%100 == 0:
            print(f'iteration {i} : Cost Function : {J_hist[-1]:0.3f} | ',
                  f'dj_dw : {dj_dw} | dj_db : {dj_db}',
                  f'w : {w} | b : {b}')
    return w,b, J_hist,p_hist
