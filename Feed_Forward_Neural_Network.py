import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.optimize
import sys



NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_OUTPUT = 10
NUM_LAYERS = NUM_HIDDEN_LAYERS +  2  
NUM_HIDDEN_NODES = 64
NUM_HIDDEN = NUM_HIDDEN_LAYERS * [ NUM_HIDDEN_NODES ]

#Initialising Random hyperparameters values to tune the model 
epoch_size = [50,100,150,200,300]
mini_batch_size = [16,32,64128]
learningRate = [0.0001,0.0002,0.01]
alpha_values = [0.0001,0.05]
num_layers=[5,7,9]
num_units_hidden_layers =[64,128]


#defining Variables to store final Model Parameters
model_epoch=0
model_learningRate=0
model_regularization_strength=0
model_minibatch_size=0
model_hidden_layers= 0
model_units_hidden_layers= 0

#Loading data and Splitting the data into coreesponding SubSets 
def load_data():

    #Load the traning and testing Data 
    X_tr = np.load("fashion_mnist_train_images.npy")
    ytr = np.load("fashion_mnist_train_labels.npy")
    X_te = np.load("fashion_mnist_test_images.npy")
    yte = np.load("fashion_mnist_test_labels.npy")

    #Check the shape of input data
    #print("RAW DATA -> X_train: {xt}, X_test: {xv}, y_train: {yt}, y_tets: {yv}".format(xt=X_tr.shape,xv=X_te.shape,yt= ytr.shape,yv=yte.shape))'

    #lets split the training data set first into validation and training data 
    data = np.c_[X_tr,ytr]
    np.random.shuffle(data) 


    row,col=data.shape
    split_per = int(0.2*row)            #split 20%-80% into val and train
    data_train = data[split_per:,:]
    data_val = data[0:split_per,:]
 

    y_train = data_train[:,-1]

    X_train = data_train[:,:-1]
    y_val = data_val[:,-1] 
    X_val = data_val[:,:-1]
    
    #Split the data into training and Validation sets using sklearn
    #X_train, X_val, y_train, y_val = train_test_split(X_tr, ytr, test_size=0.2,random_state=42)

    #Check the shape of the split data
    #print("SPLIT RAW DATA -> X_train: {xt}, X_val: {xv}, ,X_test: {xe}, y_train: {yt}, y_val: {yv}, y_tets: {ye}".format(xt=X_train.shape,xv=X_val.shape,xe=X_te.shape,yt= y_train.shape,yv=y_val.shape,ye=yte.shape))

    #Pre process the ground truth label data to convert it into one hot encoding vector format 
    #Convert to one hot encoding Vector
    def one_hot_encode_labels(label_data,num_classes):
      
      n=len(label_data) 
      y_test=[]
      for label in label_data:
        one_hot_vector = np.zeros(num_classes)
        one_hot_vector[int(label)] = 1
        y_test.append(one_hot_vector)

      y_test_new=np.asarray(np.reshape(y_test,(n,10)))
      return y_test_new
 
    #checking for y_te  and y_tr and y_val
    y_te_reshaped= one_hot_encode_labels(yte,10)
    y_tr_reshaped=one_hot_encode_labels(y_train,10)
    y_val_reshaped= one_hot_encode_labels(y_val,10)
    return X_train,y_tr_reshaped,X_te,y_te_reshaped, X_val,y_val_reshaped

def unpack (weights):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN[0]
    W = weights[start:end]
    Ws.append(W)

    # Unpack the weight matrices as vectors
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i]*NUM_HIDDEN[i+1]
        W = weights[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN[-1]*NUM_OUTPUT
    W = weights[start:end]
    Ws.append(W)

    # Reshape the weight "vectors" into proper matrices
    Ws[0] = Ws[0].reshape(NUM_INPUT,NUM_HIDDEN[0])
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN[i], NUM_HIDDEN[i-1])
    Ws[-1] = Ws[-1].reshape(NUM_HIDDEN[-1],NUM_OUTPUT)

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN[0]
    b = weights[start:end]
    bs.append(b.reshape(1,len(b)))

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i+1]
        b = weights[start:end]
        bs.append(b.reshape(1,len(b)))

    start = end
    end = end + NUM_OUTPUT
    b = weights[start:end]
    bs.append(b.reshape(1,len(b)))

    return Ws, bs

def ReLU(Z):
    Z = np.array(Z)
    Z[Z<=0]=0
    return Z

#Calculates ReLu derivative
def relu_derv(z1):
    z1[z1 >0] = 1
    z1[z1<=0] = 0
    return z1

# Performs a standard form of random initialization of weights and biases
def initWeightsAndBiases(num_layers,h_nodes,num_features,num_output):
    Ws=[]
    bs=[]

    # Sampling each weight from a 0-mean Gaussian with std.dev. of 1/sqrt(numInputs).
    # Initialize biases to small positive number (0.01).
    Val_init = 1/np.sqrt(h_nodes)

    np.random.seed(0)
    for i in range(num_layers-1):
        if i==0:
            w_i=np.random.uniform(-Val_init/2,Val_init/2,(num_features,h_nodes))
            b_i=np.random.rand(1,h_nodes)
        elif i==num_layers-2:
            w_i=np.random.uniform(-Val_init/2,Val_init/2,(h_nodes,num_output))
            b_i=np.random.rand(1,num_output)
            
        else:
            w_i=np.random.uniform(-Val_init/2,Val_init/2,(h_nodes,h_nodes))
            b_i=np.random.rand(1,h_nodes)
            
            
        Ws.append(w_i)
        bs.append(b_i)

        '''
         #checking shape and size and length of Ws and bs 
         print(" Length of Ws is: {l} and length of bs is: {b}".format(l=len(Ws), b = len(bs)))
         #Printing All 4 one by one 
         print(" Ws[0] is of shape : {s} ,Ws[1] is of shape :{s1} ,Ws[2] is of shape: {s2} , Ws[3] is of shape : {s3} ".format(s=Ws[0].shape, s1=Ws[1].shape, s2=Ws[2].shape, s3=Ws[3].shape))
         print(" bs[0] is of shape : {s} ,bs[1] is of shape :{s1} ,bs[2] is of shape: {s2} , bs[3] is of shape : {s3} ".format(s=bs[0].shape, s1=bs[1].shape, s2=bs[2].shape, s3=bs[3].shape))
         '''

    return(Ws,bs)  

#Calculates the yhat values by applying Softmax as Activation Function on the Last layer
def softmax(z):
    yhat = np.zeros(z.shape)
    z_max = np.amax(z)   
    for r in range(z.shape[0]):
        sum_z = np.sum(np.exp(z[r]-z_max))
        yhat[r] = np.exp(z[r]-z_max)/sum_z 
    return yhat         

#Forward Propagation on the layers of the Network
def forward_prop(X_mini_batch,w,b,num_layers):
    #Initialialize List to store the pre and post Activation Function Values 
    h=[]
    z=[]

    for i in range(0,num_layers-1):
        if i==0:
            z_ind=np.dot(X_mini_batch,w[i])+b[i]            
        else:         
            z_ind=np.dot(h[i-1],w[i])+b[i]
           
        h.append(ReLU(z_ind))
        z.append(z_ind)

    #Calculate the last layer yhat by applying Softmax AF
    yhat=softmax(z[i])
    h[i]=yhat
    
    return yhat,h,z

#Algorithm 6.4 Back prop 
def back_propagation(Y,X,yhat,num_layers,w,b,h,z):
    n =len(X)
    g= yhat.T - Y.T
    h[num_layers-2]=yhat

    dJdb=[]     #Gradients wrt to biases
    dJdWs=[]    #Gradients wrt to Weights

    for i in range(num_layers-2,-1,-1):
        g=np.multiply(g.T,relu_derv(z[i]))
        s=np.sum(g,axis=0)
        dJdb.append(s.reshape(1,len(s)))
        if i==0:
            dJdWs.append(np.dot(g.T,X).T)
        else:
            dJdWs.append(np.dot(g.T,h[i-1]).T)

        g=np.dot(w[i],g.T)

    dJdWs= dJdWs[::-1]
    dJdb= dJdb[::-1]


    return dJdWs,dJdb

""" BACK PROP ALGORITHM IMPLEMENTATION -> ALG 6.4"""
def back_prop(X_tr,ytr,weights,num_layers):
    #Unpack the weights
    Ws,bs = unpack(weights)
    #Calculates yhat, h and z Values
    yhat,h,z = forward_prop(X_tr,Ws,bs,num_layers)

    n=len(X_tr)
    h[num_layers-2]=yhat
    n = yhat.shape[0]

    g = yhat.T - ytr.T
    
    dJdb=[]     #Gradients wrt to biases
    dJdWs=[]    #Gradients wrt to Weights

    for i in range(num_layers-2,-1,-1):
        g=np.multiply(g.T,relu_derv(z[i]))
        s=np.sum(g,axis=0)
        dJdb.append(s.reshape(1,len(s)))
        if i==0:
            dJdWs.append(np.dot(g.T,X_tr).T)
        else:
            dJdWs.append(np.dot(g.T,h[i-1]).T)
        g=np.dot(Ws[i],g.T)

    dJdWs=dJdWs[::-1]
    dJdb=dJdb[::-1]

    '''
    print("SHAPES OF BACKPROP WEIGHTS AND BIAS")
    for i in range(len(dJdWs)):
       print(dJdWs[i].shape)

    for i in range(len(dJdWs)):
        print(dJdb[i].shape)
    '''

    return (np.hstack([ dJdW.flatten() for dJdW in dJdWs ] + [ dJdb.flatten() for dJdb in dJdb ]) /n)

#Update the weights and biases after Calculating the gradients
def update_weights_and_bias(grad_w,grad_b,lr,w,b):
    w_new=[]
    b_new=[]
    for i in range(len(grad_w)):
        w_new.append(w[i]-lr*grad_w[i])
        b_new.append(b[i]-lr*grad_b[i])

    return(w_new,b_new)
    
def fce(X_b,y_b,weights,num_layers):
    #Unpack the weights and biases in their respective Size
    Ws,bs=unpack(weights)
    yhat,h,z=forward_prop(X_b,Ws,bs,num_layers)
    fce_val=0
    n = (yhat.shape[0])    
    
    for row in range(n):
        for col in range(yhat.shape[1]):
            fce_val=fce_val+y_b[row,col]*np.log(yhat[row,col])

    fce_val=-(fce_val)/n 
    return fce_val

def accuracy(X,Y,best_w,best_b,hidden_layer):

    Y_hat,h,z=forward_prop(X,best_w,best_b,hidden_layer)

    Y_hat=np.argmax(Y_hat, axis=1)
    y = np.argmax(Y, axis=1)
    acc = np.mean(Y_hat == y)
    return (acc*100)

def crossEntropy_reg(X,y,w,alpha,yhat):
    fce_val=0.
    reg=0.
    for row in range(yhat.shape[0]):
        for col in range(yhat.shape[1]):
            fce_val=fce_val+y[row,col]*np.log(yhat[row,col])
    
    fce_val=-(fce_val)/(yhat.shape[0])    
    
    for col in range(yhat.shape[1]):
        reg=reg+np.dot(w.T[col],w[:,col])
        
    fce_val=fce_val+(alpha*reg/2)
    
    return(fce_val)

def train (trainX,trainY,epoch,mini_batch,lr,reg,n_layers,h_nodes,Ws,bs):
    n=100
    #n = (X_train.shape[0])
    weights = np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])
    trajectory = np.copy(weights)

    for _ in range(epoch):  
        for i in range(int(n/mini_batch)):
            start= i*mini_batch
            end=start + (mini_batch) - 1

            X_mini = trainX[(start):(end)]
            y_mini = trainY[(start):(end)]

            #gaussian Noise in each mini batch to keep the training Examples fresh 
            mu=0.0
            std = 0.1
            noise = np.random.normal(mu, std, size = X_mini.shape)
            X_mini = X_mini + noise

            yhat_t,h_t,z_t=forward_prop(X_mini,Ws,bs,n_layers)

            
            grad_w,grad_b=back_propagation(y_mini,X_mini,yhat_t,n_layers,Ws,bs,h_t,z_t)
            
            #update the Weights and Bias matrices
            w_new,b_new=update_weights_and_bias(grad_w,grad_b,lr,Ws,bs)
            Ws = w_new
            bs= b_new

    wab = np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])
    trajectory = np.vstack([trajectory,np.copy(wab)])

    return Ws, bs , trajectory 

#Takes training data and change the hyperparameter values to train the model
def tune_hyperparams(X_train,ytr,X_val,yval):
    #here w,b is initialised weights and bias according to variance 
    opt_weights = []
    opt_bias=[]
    
    fce_minimum = sys.maxsize
    fce_max_Acc= 0

    global model_epoch
    global model_learningRate
    global model_minibatch_size
    global model_regularization_strength
    global model_hidden_layers
    global model_units_hidden_layers
    for n_layers in num_layers:
        for h_nodes in num_units_hidden_layers:
            for epoch in epoch_size:
                for batch in mini_batch_size:
                    for lr in learningRate:
                        for reg in alpha_values:
                            print("epoch: {e}, batch: {b}, learning: {l}, alpha: {a}".format(e=epoch,b=batch,l=lr,a=reg))
                            n_f= 784
                            n_c =10
                            #Initialising Random Weights and Biases
                            W,b = initWeightsAndBiases(n_layers, h_nodes,n_f,n_c)

                            #Weights and Bias for current training iteration updated with forward and backprop
                            #Forward and Backprop is done in train by implementing SGD
                            weights,bias,t = train(X_train,ytr,epoch,batch,lr,reg,n_layers,h_nodes,W,b)

                            #Testing the weights and bias on Validation Set
                            yhat_val,h_val,z_val=forward_prop(X_val,weights,bias,n_layers)
                            fce_val =crossEntropy_reg(X_val,yval,weights[-1],reg,yhat_val)
                            fce_acc_val=accuracy(X_val,yval,weights,bias,n_layers)   

                            print("Accuracy on Validation Set {acc}". format(acc=fce_acc_val))
                            print("Loss on Validation Set {l}". format(l=fce_val))
                             
                            if fce_val < fce_minimum:
                                print("----------------------------------------------------------------")
                                print("epoch: {e}, batch: {b}, learning: {l}, alpha: {a}".format(e=epoch,b=batch,l=lr,a=reg))
                                fce_minimum = fce_val
                                print("fce_minimum_val")
                                print(fce_minimum)
                                fce_max_Acc = fce_acc_val
                                print("Accuracy on Validation Set {acc}". format(acc=fce_acc_val))
                                print("----------------------------------------------------------------")
                                fce_val_Acc=fce_acc_val
                                opt_weights= weights
                                opt_bias=bias
                                model_epoch=epoch
                                model_learningRate=lr
                                model_minibatch_size=batch
                                model_regularization_strength=reg
                                model_hidden_layers= n_layers
                                model_units_hidden_layers=h_nodes

    return fce_minimum,opt_weights,opt_bias,fce_val_Acc

def final_test (X,Y,epoch,mini_batch,lr,reg,n_layers,h_nodes,w_new,b_new):
    for e in range(1):
        yhat,h,z=forward_prop(X,w_new,b_new,n_layers)
        fce =crossEntropy_reg(X,Y,w_new[-1],reg,yhat)
        acc=accuracy(X,Y,w_new,b_new,n_layers)  
        print("EPOCH ITERATION NUMBER : : {f}".format(f= e)) 
        print("Accuracy on Test Set {acc}". format(acc=acc))
        print("Loss on Test Set {l}". format(l=fce))
     
    return True

# Creates an image representing the first layer of weights (W0).
def show_W0 (W):
    Ws,bs = unpack(W)
    W = Ws[0]
    n = int(NUM_HIDDEN[0] ** 0.5)
    plt.imshow(np.vstack([
        np.hstack([ np.pad(np.reshape(W[:,idx1*n + idx2], [ 28, 28 ]), 2, mode='constant') for idx2 in range(n) ]) for idx1 in range(n)
    ]), cmap='gray'), plt.show()

def plotSGDPath (trainX, trainY, trajectory):
    # This toy plot to show a 2-d projection of the weight space
    # along with the associated loss (cross-entropy), plus a superimposed 
    # trajectory across the landscape that was traversed using SGD. Use
    # sklearn.decomposition.PCA's fit_transform and inverse_transform methods.

    pca = PCA(n_components = 2)
    zs = pca.fit_transform(trajectory)

    def toyFunction (x1, x2):
        z = [x1,x2]
        wab = pca.inverse_transform(z)
        Ws,bs = unpack(wab)
        yhat,h,z = forward_prop(trainX, Ws,bs, NUM_LAYERS)
        loss = fce(trainX,trainY,wab,NUM_LAYERS)
        return loss

    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')


    axis1 = np.linspace( min(zs[:,0]), max(zs[:,0]), 150)  
    axis2 = np.linspace( min(zs[:,1]), max(zs[:,1]), 150)
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))

    for i in range(len(axis1)):
        for j in range(len(axis2)):
            Zaxis[i,j] = toyFunction(Xaxis[i,j], Yaxis[i,j])
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    Xaxis_ = zs[:, 0]  
    Yaxis_ = zs[:, 1] 
    Zaxis_ = np.zeros(len(Xaxis_))
    for i in range(len(Xaxis)):
        Zaxis_[i] = toyFunction(Xaxis_[i], Yaxis_[i])

    ax.scatter(Xaxis_, Yaxis_, Zaxis_, color='r')
    plt.show()

if __name__== "__main__":
    # Load training data.
    # Recommendation: divide the pixels by 255 (so that their range is [0-1]), and then subtract
    # 0.5 (so that the range is [-0.5,+0.5]).

    print("Hello! Getting Started!")
    X_train,ytr,X_te,yte,X_val,yval= load_data()
    
    print("SPLIT RAW DATA -> X_train: {xt}, X_val: {xv}, ,X_test: {xe}, y_train: {yt}, y_val: {yv}, y_tets: {ye}".format(xt=X_train.shape,xv=X_val.shape,xe=X_te.shape,yt= ytr.shape,yv=yval.shape,ye=yte.shape))

    #Transforming the Data Set 
    trainX= (X_train/255 - 0.5)
    ValX = (X_val/255 - 0.5)
    testX= (X_te/255 - 0.5)
    trainY= ytr

    fce_minimum,opt_w,opt_b,fce_acc = tune_hyperparams(trainX,ytr,ValX,yval)
    print("Best Model Parameters are:")
    print("Best epoch: :"+"{:.2f}".format(model_epoch))
    print("Best learningRate: :"+"{:.7f}".format(model_learningRate))
    print("Best miniBatchSize: :"+"{:.2f}".format(model_minibatch_size))
    print("Best regularizationStrength: :"+"{:.7f}".format(model_regularization_strength))
    print("Fce Minimum Training Data: :"+"{vt}".format(vt=fce_minimum))
    print("Fce Maximum ACC: :"+"{vt}".format(vt=fce_acc))

    #Tain the data once with the model parameters on training data set 
    #W_opt,b_opt=final_train(trainX,ytr,model_epoch,model_minibatch_size,model_learningRate,model_regularization_strength,model_hidden_layers,model_units_hidden_layers,opt_w,opt_b)

    ''' Calculating the training efficiency on our test set '''
    W_opt= opt_w
    b_opt= opt_b

    
    Y=final_test(testX,yte,model_epoch,model_minibatch_size,model_learningRate,model_regularization_strength,model_hidden_layers,model_units_hidden_layers,opt_w,opt_b)

    yhat_te,h_te,z_te=forward_prop(testX,W_opt,b_opt,model_hidden_layers)
    fce_te =crossEntropy_reg(testX,yte,W_opt[-1],model_regularization_strength,yhat_te)
    acc_te=accuracy(testX,yte,W_opt,b_opt,model_hidden_layers)  

    print("Fce UnRegularized Test Data: :"+"{vt}".format(vt=fce_te))
    print("Accuracy Test Data: :"+"{vt}".format(vt=acc_te))


    """ CHECK GRADIENT FUNCTION ACC"""
    #Checking the Gradient Calculation wrt Weights and Biases Accuracy before Implementing SGD 
    #Init Random Weights and Biases
    Ws, bs = initWeightsAndBiases(NUM_LAYERS,NUM_HIDDEN_NODES,trainX.shape[1],ytr.shape[1])
    weights = np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])

    #Initialising Values for check
    num_layers= NUM_LAYERS

    #Loss with Random Initialization of Weights 
    random_loss_value= fce(trainX,ytr,weights,num_layers)
    print("fce random :: {f}".format(f=random_loss_value))
    

    
    print("Checking Gradient Function Accuracy")
    print("Function scipy optimize grad check : : {f}".format(f=scipy.optimize.check_grad(lambda weights_: fce(np.atleast_2d(trainX[0:5,:]), np.atleast_2d(trainY[0:5,:]), weights_,num_layers), \
                                    lambda weights_: back_prop(np.atleast_2d(trainX[0:5,:]), np.atleast_2d(trainY[0:5,:]), weights_,num_layers), \
                                    weights)))


    """ VISULAISE THE TRAINED WEIGTHS"""
    optimized_weights = np.hstack([ W.flatten() for W in W_opt ] + [ b.flatten() for b in b_opt ])
    show_W0(optimized_weights)
    
    # Plot the SGD trajectory
    W_opt,b_opt,trajectory = train(trainX,ytr,model_epoch,model_minibatch_size,model_learningRate,model_regularization_strength,model_hidden_layers,model_units_hidden_layers,opt_w,opt_b)
    plotSGDPath(trainX[:200,:], trainY[:200,:], trajectory)
