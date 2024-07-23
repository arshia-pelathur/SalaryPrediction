def gradient(x,y,w,b):
    '''
    Computes the gradient for w and b for Gradient Descent

    Args:
    x : Data having a shape of m(size of training data)
    y : target values
    w,b : model parameters

    Returns:
    djdw : gradient of cost function wrt w
    djdb : gradient of cost function wrt b
    '''
    m = x.shape[0]
    djdw = 0
    djdb = 0

    for i in range(m):
        f_wb = w*x[i] + b
        djdw_temp = (f_wb - y[i]) * x[i]      # derivative of cost wrt w
        djdb_temp = (f_wb - y[i])             # derivate of cost wrt b
        djdw+=djdw_temp
        djdb+=djdb_temp
    
    djdw = djdw/m
    djdb = djdb/m

    return djdw,djdb


def cost(x,y,w,b):
    '''
    Computes the cost function

    Args:
    x : Data having a shape of m(size of training data)
    y : target values
    w,b : model parameters

    Returns:
    total_cost : total cost function for a certain w and b value
    '''
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w*x[i]+b
        cost = cost+(f_wb - y[i])**2
    total_cost = cost/(2*m)
    return total_cost

