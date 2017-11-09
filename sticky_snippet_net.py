import tensorflow as tf
import sys
import os
import random
import time

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

MINI_BATCH_SIZE = 5
LEARNING_RATE = 0.001
EPOCHS = 15


class net:
    di = {1 : 3 , 2 : 4 , 3 : 1 , 4 : 2}
    dic_onehot = {-10 : 0 , 12 : 1 , 34 : 2 , 56 : 3 , 78 : 4 , 210 : 5}
    dinu = {'A' : '1' , 'B' : '2' , 'C' : '3' , 'D' : '4'}

    def __init__(self , data_model):
        
        #check if folder exists
        if os.path.isdir(data_model) is not True:
            print ("NO DATA")
            exit(1)
        
        self.data_import(data_model) 

    def __initialise_variables(param):
        #placeholder initialiser
        x = tf.placeholder( tf.float32 , [ None , 40 ] , name = "x")
        
        y = tf.placeholder( tf.float32 , [ None , 6 ] , name = "y")

        #weights and bias initialiser
        uniform_init = tf.contrib.layers.xavier_initializer()

        #hidden layer 1 to have 40 neurons
        W1 = tf.get_variable("W1", shape = (40, 40), dtype=tf.float32 ,initializer=uniform_init)
        b1 = tf.get_variable("b1", shape = (40), dtype=tf.float32, initializer=uniform_init)
        
        #hidden layer 2 to have 40 neurons
        W2 = tf.get_variable("W2", shape = (40, 40) , initializer=uniform_init)
        b2 = tf.get_variable("b2", shape = (40) , initializer=uniform_init)
        
        #hidden layer 3 to have 40 neurons
        W3 = tf.get_variable("W3", shape = (40, 40) , initializer=uniform_init)
        b3 = tf.get_variable("b3", shape = (40) , initializer=uniform_init)
        
        #hidden layer 4 to have 6 neurons
        W4 = tf.get_variable("W4", shape = (40 , 40) , initializer=uniform_init)
        b4 = tf.get_variable("b4", shape = (40) , initializer=uniform_init)

        #hidden layer 4 to have 6 neurons
        W5 = tf.get_variable("W5", shape = (40 , 6) , initializer=uniform_init)
        b5 = tf.get_variable("b5", shape = (6) , initializer=uniform_init)

        #hidden layers
        layer1 = tf.nn.relu(net.linear_z(W1 , x , b1))
        layer2 = tf.nn.relu(net.linear_z(W2 , layer1 , b2))
        layer3 = tf.nn.relu(net.linear_z(W3 , layer2 , b3))
        layer4 = tf.nn.relu(net.linear_z(W4 , layer3 , b4))
        layer5 = net.linear_z(W5 ,layer4 , b5)

        #output layer of the nn
        out_layer = tf.nn.softmax( layer5 , name='outlayer_to_crossentropy')
        
        #cost function
        cf = tf.nn.softmax_cross_entropy_with_logits(
            logits = layer5 , labels = y)

        #predicts if the output is equal to its expectation 
        correctness_of_prediction = tf.equal(
            tf.argmax(out_layer, 1), tf.argmax(y, 1))

        #accuracy of the 
        accuracy = tf.reduce_mean(
            tf.cast(correctness_of_prediction, tf.float32), name='accuracy')

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE) #epsilon is passed
        train = optimizer.minimize(cf)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        if param == 'training':
            return x, y, sess, train, accuracy, out_layer
        elif param == 'test':
            return x, y, out_layer, accuracy

    def linear_z( weights , x , bias):
        p = tf.add(tf.matmul(x , weights) , bias)
        return p
    
    def __convert(string):
        list1 = list(string)
        for i in range ( 0 , 40 ):
            list1[i] = net.dinu[list1[i]]
        return list1

    def actual_output(self , counter):
        iterator=0
        while net.di[self.x_train[counter][iterator]] == self.x_train[counter][40 - 1 - iterator] and iterator<=20:
            iterator = iterator + 1

        one_hot_vector = list([0, 0, 0, 0, 0, 0])
        position = 0
        if iterator%2==0:
            position = (iterator-1)* 10 + iterator

        else:
            position = (iterator)* 10 + iterator + 1
        
        if iterator>8 and iterator<20:
            one_hot_vector[net.dic_onehot[78]] = 1    
        
        else:
            one_hot_vector[net.dic_onehot[position]] = 1
        return one_hot_vector

    def data_import( self , data_model ):
        self.x_train = []
        self.y_train = []
        os.chdir(data_model)
        counter=0
        for files in os.listdir(os.getcwd()):
            f = open(files , 'r')
            s = f.readlines()
            for string in s:
                string = string[:-1]
                string = ''.join(net.__convert(string))
                if len(string) == 40:
                    list_line_reader = list(string)
                    self.x_train.append([int(x) for x in list_line_reader])      
                    self.y_train.append(self.actual_output(counter))
                    counter = counter + 1
        #randomises data
        for i in range(1, len(self.x_train)-1):
            j = random.randint(i+1, len(self.x_train)-1)
            self.swap_t(i, j)
        
    def swap_t (self , i , j):
        tempx = self.x_train[i]
        self.x_train[i] = self.x_train[j]
        self.x_train[j] = tempx
        tempy = self.y_train[i]
        self.y_train[i] = self.y_train[j]
        self.y_train[j] = tempy

      
    def mini_batch(self , model_file, xdata=None, ydata=None, params=None):
        '''Trains the data with the specified mini batch size
        '''

        if params is None:
            xdata=self.x_train
            ydata=self.y_train
            x, y, sess, train, accuracy, _ = net.__initialise_variables('training')
        
        else:
            x=params['x']
            y=params['y']
            sess=params['sess']
            train=params['train']
            accuracy=params['accuracy']

        start_time=time.time()
    
        for j in range (EPOCHS):
            training_acc=0
            print("EPOCH NUMBER: ", j)
            for k in range(0, len(xdata), MINI_BATCH_SIZE):
                current_batch_x_train = xdata[k:k+MINI_BATCH_SIZE]
                current_batch_y_train = ydata[k:k+MINI_BATCH_SIZE]

                _, acc= sess.run([train ,accuracy],
                        {x: current_batch_x_train, y: current_batch_y_train})           

                training_acc = training_acc + acc

            training_acc= training_acc * MINI_BATCH_SIZE/ len(xdata)
            if params is None:
                print ("Accuracy= ",training_acc)

        train_time=time.time() - start_time
        
        saver= tf.train.Saver()
        saver.save(sess , model_file)
        
        print("Total training time= ", train_time, "seconds")    
    
    def _5_fold_trainer(self, model_file):
        '''Performs 5-fold cross validation
        '''
        x, y, sess, train, accuracy, out_layer = net.__initialise_variables('training')
        params={}
        params['x']=x
        params['y']=y
        params['sess']=sess
        params['train']=train
        params['accuracy']=accuracy
        params['out_layer']=out_layer

        subset_size = len(self.x_train) // 5
        subsets_x = []
        subsets_y = []
        for i in range(0, len(self.x_train) , subset_size):
            subset = self.x_train[i:i+subset_size]
            subsets_x.append(subset)
            subset = self.y_train[i:i+subset_size]
            subsets_y.append(subset)

        for j in range(5):
            train_set_x = []
            train_set_y = []
            test_set_x = []
            test_set_y = []
            for i in range(5):
                if i != j:
                    train_set_x.extend(subsets_x[i])
                    train_set_y.extend(subsets_y[i])
                    
                else:
                    test_set_x=subsets_x[i]
                    test_set_y=subsets_y[i]
            self.mini_batch(model_file, train_set_x, train_set_y, params)
            acc, matrix= self.test( model_file, params, test_set_x, test_set_y)
            print(acc)

        print (matrix)

        
    def test(self, model_file, params=None, xdata=None, ydata=None) :

        if params is None:
            x, y, out_layer, acc=net.__initialise_variables('test')
            xdata=self.x_train
            ydata=self.y_train

        else :
            x=params['x']
            y=params['y']
            out_layer=params['out_layer']
            acc=params['accuracy']

        # Get Tensorflow model
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess , model_file)        
        print ("Model restored!")
        
        start_time=time.time()

        total_accuracy = sess.run(acc, {x: xdata, y: ydata})

        #the position of the '1' bit in the 1-hot vector
        prediction = tf.argmax(out_layer, 1)
        actual = tf.argmax(ydata, 1)

        pred, act = sess.run([prediction, actual],  {x: xdata, y: ydata})
        print(pred,act)
        confusion_matrix = sess.run(tf.confusion_matrix(
                act, pred), {x: xdata, y: ydata})

        TP = tf.count_nonzero(prediction * actual, dtype=tf.float32)
        TN = tf.count_nonzero((prediction - 1) * (actual - 1), dtype=tf.float32)
        FP = tf.count_nonzero(prediction * (actual - 1), dtype=tf.float32)
        FN = tf.count_nonzero((prediction - 1) * actual, dtype=tf.float32)

        tp, tn, fp, fn = sess.run([TP, TN, FP, FN],  {x: xdata, y: ydata})
        
        tpr = tp / (tp + fp)
        tnr = tn / (tn + fn)
        fpr = fp / (tp + fp)
        fnr = fn / (tn + fn)

        duration = time.time() - start_time

        if params is None:
            print ("Total number of items tested on: ", len(xdata))
            print ("Total Accuracy over testing data: ", total_accuracy)
            print ("True Positive Rate: ", tpr)
            print ("True Negative Rate: ", tnr)
            print ("False Positive Rate: ", fpr)
            print ("False Negative Rate: ", fnr)
            print("Testing time: ", duration, " seconds")
            print("Confusion Matrix:\n",confusion_matrix)

        else :
            return  total_accuracy, confusion_matrix
          

if __name__ == '__main__' :
    if len(sys.argv) != 4:
        print ("Please supply the correct arguments")
        raise SystemExit(1)

model_file = os.path.join( os.getcwd() , str(sys.argv[ 2 ]) )
model_file = model_file + '.txt'
data_model = os.path.join( os.getcwd() , str(sys.argv[ 3 ]) )

trainer = net ( data_model )

if str(sys.argv[1]) == 'train' :
    trainer.mini_batch(model_file)
elif str(sys.argv[1]) == '5fold' :
    trainer._5_fold_trainer(model_file)
else :
    trainer.test(model_file)   



