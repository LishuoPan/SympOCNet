import numpy as np
import matplotlib.pyplot as plt

def dist(w, v, p, qr):
    l2 = np.sum((w - v) ** 2)
    d = np.sum((p - v) * (w - v), axis = -1, keepdims = True) / l2
    t = np.maximum(np.zeros_like(d), np.minimum(np.ones_like(d), d))
    projection = v + np.matmul(t, (w - v))
    return np.sqrt(np.sum((p - projection) ** 2, axis = -1)) - qr

def plot_heat(q_pred, net, filename, num_interpolate, num_total, y_train, y_test):
    num_drone = net.dim // 2
    x = np.linspace(-5,5,100)
    y = np.linspace(-5,5,100)
    xx, yy = np.meshgrid(x, y)
    zz = []
    for i, w in enumerate(net.ws):
        w = np.array([w])
        v = w + np.array([[np.cos(net.angles[i]), np.sin(net.angles[i])]]) * net.ql[i]
        h = lambda p: dist(w, v, p, net.qr)
        p = np.concatenate([xx.reshape([-1,1]), yy.reshape([-1,1])], axis = -1)
        zz.append(h(p).reshape([100,100]))
    zz = np.min(np.array(zz), axis = 0)
    
    # convert y_train and y_test to numpy
    y_train['bd'] = y_train['bd'].detach().cpu()
    y_test['bd'] = y_test['bd'].detach().cpu()
    
    plt.figure(figsize = [4*num_interpolate,4*2])
    for index in range(num_total):
        ax = plt.subplot(2,num_interpolate,index+1)
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))
        plt.contour(xx,yy,zz,[0])
        for i in range(y_train['bd'].size()[0]):
            for j in range(num_drone):
                ax.add_patch(plt.Circle((y_train['bd'][i, 0, 2*j], y_train['bd'][i, 0, 2*j+1]), net.dr, fill = True, color = 'yellow', linestyle = '--'))
        for i in range(num_drone):
            ax.add_patch(plt.Circle((y_test['bd'][index,0,2*i], y_test['bd'][index,0,2*i+1]), net.dr, fill = False, color = 'black', linestyle = '--'))
            ax.add_patch(plt.Circle((y_test['bd'][index,1,2*i], y_test['bd'][index,1,2*i+1]), net.dr, fill = False, color = 'red', linestyle = '-'))
            plt.plot(q_pred[index,:,2*i], q_pred[index,:,2*i+1])
    plt.savefig('figs/'+filename+'.pdf')
    

def plot_anime(q_pred, net, filename):
    from matplotlib import animation
    from random import randint
    num_drone = net.dim // 2
    x = np.linspace(-5,5,100)
    y = np.linspace(-5,5,100)
    xx, yy = np.meshgrid(x, y)
    zz = []
    for i, w in enumerate(net.ws):
        w = np.array([w])
        v = w + np.array([[np.cos(net.angles[i]), np.sin(net.angles[i])]]) * net.ql[i]
        h = lambda p: dist(w, v, p, net.qr)
        p = np.concatenate([xx.reshape([-1,1]), yy.reshape([-1,1])], axis = -1)
        zz.append(h(p).reshape([100,100]))
    zz = np.min(np.array(zz), axis = 0)
    fig = plt.figure(figsize = [4,4])
    ax = plt.axes(xlim=(-5, 5), ylim=(-5,5))
    plt.contour(xx,yy,zz,[0])
    colors = []
    for i in range(num_drone):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    drones = []
    for i in range(num_drone):
        drones.append(plt.Circle((q_pred[0, 2*i], q_pred[0, 2*i+1]), net.dr, fill = False, color = colors[i]))
    def init():
        for i in range(num_drone):
            ax.add_patch(drones[i])
        return drones 
    def animate(i):
        for j in range(num_drone):
            drones[j].center = (q_pred[i,2*j], q_pred[i,2*j+1])
        return drones
    frames = q_pred.shape[0]
    delay_time = 20000 // q_pred.shape[0]
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=delay_time, blit=True, repeat = False)
    anim.save('figs/'+filename+'.gif', writer='imagemagick', fps=30)
    plt.show()
    plt.close()
    
def plot_cost_constraint(data, net, loss, filename, print_every):
    cost = loss[:,-2]
    hmin = loss[:,-1]
    iterations = np.arange(0, print_every * len(cost), print_every)
    plt.figure()
    plt.plot(iterations, cost, label = 'cost', c = 'b')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/'+filename+'/cost.pdf')
    plt.show()
    plt.close()
    plt.figure()
    plt.plot(iterations, hmin, label = 'constraint', c = 'r')
    plt.plot(iterations, 0*hmin, '.', markersize = 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/'+filename+'/constraint.pdf')
    plt.close()