import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.util import deprecation
from tensorflow import keras
from tensorflow.keras import layers
import time
from keras import backend as K

deprecation._PRINT_DEPRECATION_WARNINGS = False

def f(x,y):
	if (np.any(x < -1) or np.any(y < -1) or np.any(x > 1) or np.any(y > 1)):
		raise Exception("Invalid value!")
	return np.cos(x + 6 * 0.35 * y) + 2 * 0.35 * (x * y)
	
def f_array(X_train):
	if (np.any((X_train < -1)) or np.any((X_train > 1))):
		raise Exception("Invalid value!")
	return np.cos(X_train[:,0] + 6 * 0.35 * X_train[:,1]) + 2 * 0.35 * (X_train[:,0] * X_train[:,1])
	
def model(X, w_h1, w_o):
	h = tf.nn.sigmoid(tf.matmul(X, w_h1))
	return tf.matmul(h, w_o)
	
def RMSE(y, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y), axis=-1))
	
class ConvergenceChecker(keras.callbacks.Callback):
	#catchall callback class for general tracking of convergence, how long it takes to get there, which epoch it was achieved on
	def on_train_begin(self, logs={}):
		self.converged = False
		self.elapsed = 0
		self.times = []
	
	def on_epoch_begin(self, batch, logs={}):
		self.times.append(time.time())

	def on_epoch_end(self, batch, logs={}):
		self.times[len(self.times)-1] = time.time() - self.times[len(self.times)-1]
		if (self.converged == False):
			self.elapsed += 1
			if (logs.get('loss') < 0.02):
				self.converged = True
				
class stop(keras.callbacks.Callback):
	#standard keras EarlyStopping does not support the idea of stopping when validation error is not decreasing any longer, at least not as far as I'm aware
	def on_train_begin(self, logs={}):
		self.patience = 10
		self.prev_loss = 0
		self.curr_loss = 0
		
	def on_epoch_end(self, batch, logs={}):
		self.curr_loss = logs.get('val_RMSE')
		if (self.curr_loss - self.prev_loss >= 0.02):
			self.patience -= 1
		else:
			self.patience = 10
			
		if (self.patience == 0):
			self.model.stop_training=True
		self.prev_loss = self.curr_loss
				
class fig6(keras.callbacks.Callback):
	#made for graphing figure 6, as the name implies: contains all relevant measures
	def on_train_begin(self, logs={}):
		self.prev_loss = 0
		self.prev_val_loss = 0
		
		self.losses = np.empty(3)
		self.val_losses = np.empty(3)
		
		self.converged = False
		self.flipMe = False
		
	def on_epoch_end(self, batch, logs={}):
		if (self.flipMe == True):
			self.flipMe = False
			self.model.save_weights('1c_after_convergence.h5')
			self.losses[2] = logs.get('loss')
			self.val_losses[2] = logs.get('val_loss')
			
		if (self.converged == False):
			self.model.save_weights('1c_prev_to_convergence.h5')
			if (logs.get('loss') <= 0.02):
				self.model.save_weights('1c_convergence.h5')
				self.converged = True
				self.flipMe = True
				
				self.losses[0] = self.prev_loss
				self.val_losses[0] = self.prev_val_loss
				
				self.losses[1] = logs.get('loss')
				self.val_losses[1] = logs.get('val_loss')
				
			self.prev_loss = logs.get('loss')
			self.prev_val_loss = logs.get('val_loss')
				
def a():
	cb = ConvergenceChecker()
	num_iters = 5
	num_epochs = 1000

	


	def MSE_table():
		converge_2 = 0
		converge_8 = 0
		converge_50 = 0

		MSE_2 = 0
		MSE_8 = 0
		MSE_50 = 0

		list_2 = [2]
		list_8 = [8]
		list_50 = [50]
	
		final_models = []
		for x in range(num_iters):
			for num_hidden in [2,8,50]:
				model = keras.Sequential()
				model.add(layers.Dense(num_hidden, activation='tanh'))
				model.add(layers.Dense(1))
				model.compile(optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.05),
							  loss='mse',
							  metrics=[RMSE])
				hist = model.fit(X_train, y_train, epochs=num_epochs, batch_size=50, verbose=0, callbacks=[cb]).history
				
				if (num_hidden == 2):
					converge_2 += cb.elapsed
					MSE_2 += hist['loss'][cb.elapsed-1]
				elif (num_hidden == 8):
					converge_8 += cb.elapsed
					MSE_8 += hist['loss'][cb.elapsed-1]
				else:
					converge_50 += cb.elapsed
					MSE_50 += hist['loss'][cb.elapsed-1]
				
				if (x == num_iters - 1):
					final_models.append(model)
				
			print("Iteration", x+1, "complete.")
				
		list_2.append(converge_2/num_iters)
		list_2.append(MSE_2/num_iters)

		list_8.append(converge_8/num_iters)
		list_8.append(MSE_8/num_iters)

		list_50.append(converge_50/num_iters)
		list_50.append(MSE_50/num_iters)

		print(list_2[0], "neurons converged after", list_2[1],"epochs on average, at an average MSE of", list_2[2], ".")
		print(list_8[0], "neurons converged after", list_8[1],"epochs on average, at an average MSE of", list_8[2], ".")
		print(list_50[0], "neurons converged after", list_50[1],"epochs on average, at an average MSE of", list_50[2], ".")
		
		return final_models
		
	def fig3(models):
		y_pred_2 = models[0].predict(X_test)
		y_pred_8 = models[1].predict(X_test)
		y_pred_50 = models[2].predict(X_test)
	
		plt.contour(X1, X2, y_test.reshape(9,9), colors='k')
		plt.contour(X1, X2, y_pred_2.reshape(9,9), colors='r')
		plt.contour(X1, X2, y_pred_8.reshape(9,9), colors='g')
		plt.contour(X1, X2, y_pred_50.reshape(9,9), colors='b')
		plt.xlabel('x1')
		plt.ylabel('x2')
		plt.annotate('black = target', (0.2, 0.9))
		plt.annotate('red = 2 neurons', (0.2, 0.8))
		plt.annotate('green = 8 neurons', (0.2, 0.7))
		plt.annotate('blue = 50 neurons', (0.2, 0.6))
		plt.show()
		
		
	
	models = MSE_table()
	fig3(models)
		
def b():
	##WARNING! This function will save 300 approx. 15KB files to the current directory
	##These are the weights, checkpointed so I can retrieve them later
	num_hidden = 8
	num_epochs = 100
	num_batches = 5
	cb = ConvergenceChecker()
	checkpoint1 = tf.keras.callbacks.ModelCheckpoint(filepath='GD@{epoch:02d}.hdf5',verbose=0,save_weights_only=True, save_freq='epoch')
	checkpoint2 = tf.keras.callbacks.ModelCheckpoint(filepath='GDM@{epoch:02d}.hdf5',verbose=0,save_weights_only=True, save_freq='epoch')
	checkpoint3 = tf.keras.callbacks.ModelCheckpoint(filepath='RMS@{epoch:02d}.hdf5',verbose=0,save_weights_only=True, save_freq='epoch')
	
	list_gd = []
	list_gdm = []
	list_RMS = []



	#initializing models, collecting data for convergence time/loss over time/time to train
	modelGD = keras.Sequential()
	modelGD.add(layers.Dense(num_hidden, activation='tanh'))
	modelGD.add(layers.Dense(1))
	
	modelGD.compile(optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.02),
						  loss='mse',
						  metrics=[RMSE])
	hist = modelGD.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batches, verbose=0, callbacks=[cb,checkpoint1]).history
	list_gd.append(cb.elapsed)			  
	list_gd.append(hist['loss'])			  
	list_gd.append(cb.times)	  
	
	
	modelGDM = keras.Sequential()
	modelGDM.add(layers.Dense(num_hidden, activation='tanh'))
	modelGDM.add(layers.Dense(1))
	modelGDM.compile(optimizer=tf.compat.v1.train.MomentumOptimizer(learning_rate = 0.02, momentum = 0.02),
						  loss='mse',
						  metrics=[RMSE])
	hist = modelGDM.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batches, verbose=0, callbacks=[cb,checkpoint2]).history
	list_gdm.append(cb.elapsed)	
	list_gdm.append(hist['loss'])	
	list_gdm.append(cb.times)	
	
	
	
	modelRMS = keras.Sequential()
	modelRMS.add(layers.Dense(num_hidden, activation='tanh'))
	modelRMS.add(layers.Dense(1))
	modelRMS.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(learning_rate = 0.02),
						  loss='mse',
						  metrics=[RMSE])
	hist = modelRMS.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batches, verbose=0, callbacks=[cb,checkpoint3]).history
	list_RMS.append(cb.elapsed)
	list_RMS.append(hist['loss'])
	list_RMS.append(cb.times)




	#Table of convergence times
	print("Gradient descent took", list_gd[0], "epochs to converge.")
	print("The momentum optimizer took", list_gdm[0], "epochs to converge.")
	print("RMS took", list_RMS[0], "epochs to converge.\n")
	
	
	
	#Evaluating models on test set via RMSE
	hist = modelGD.evaluate(x=X_test, y=y_test, batch_size=num_batches, verbose=0)
	list_gd.append(hist[1])
	
	hist = modelGDM.evaluate(x=X_test, y=y_test, batch_size=num_batches, verbose=0)
	list_gdm.append(hist[1])
	
	hist = modelRMS.evaluate(x=X_test, y=y_test, batch_size=num_batches, verbose=0)
	list_RMS.append(hist[1])
	
	
	
	#Evaluating models w/ their weights at time of convergence
	modelGD.load_weights('GD@' + str(list_gd[0]) + '.hdf5')
	modelGDM.load_weights('GDM@' + str(list_gdm[0]) + '.hdf5')
	modelRMS.load_weights('RMS@' + str(list_RMS[0]) + '.hdf5')
	hist = modelGD.evaluate(x=X_test, y=y_test, batch_size=num_batches, verbose=0)
	list_gd.append(hist[1])
	
	hist = modelGDM.evaluate(x=X_test, y=y_test, batch_size=num_batches, verbose=0)
	list_gdm.append(hist[1])
	
	hist = modelRMS.evaluate(x=X_test, y=y_test, batch_size=num_batches, verbose=0)
	list_RMS.append(hist[1])
	
	def MSE_Plot():
		#graphing MSE of 3 models over time
		plt.plot(range(num_epochs), list_gd[1], label='GradientDescentOptimizer', marker='o')
		plt.plot(range(num_epochs), list_gdm[1], label='MomentumOptimizer', marker='o')
		plt.plot(range(num_epochs), list_RMS[1], label='RMSPropOptimizer', marker='o')
		plt.xlabel('Epoch')
		plt.ylabel('MSE')
		plt.legend()
		plt.show()
	
	def CPU_Time_Plot():
		#graphing CPU time of 3 models per epoch
		x = np.arange(num_epochs)
		ax = plt.subplot(111)
		plt.bar(x=(x - 0.3), height=list_gd[2], label='GradientDescentOptimizer', width=0.3)
		plt.bar(x=x, height=list_gdm[2], label='MomentumOptimizer', width=0.3)
		plt.bar(x=(x + 0.3), height=list_RMS[2], label='RMSPropOptimizer', width=0.3)
		plt.xlabel('Epoch')
		plt.ylabel('CPU Time Taken')
		plt.legend()
		plt.show()
	
	def Max_Acc(index):
		#prints each accuracy and highlights best (different prints depending on 100 epoch variant vs. convergence time variant)
		min = list_gd[index]
		case = 0
		print("Gradient descent achieved RMSE of", list_gd[index], "on the test set.")
		
		if (list_gdm[index] < min):
			min = list_gdm[index]
			case = 1
		print("The momentum optimizer achieved RMSE of", list_gdm[index], "on the test set.")
		
		if (list_RMS[index] < min):
			min = list_RMS[index]
			case = 2
		print("RMS achieved RMSE of", list_RMS[index], "on the test set.")
		
		if (index==3):
			if (case <= 0):
				print("Gradient descent had the best accuracy at the end of training.")
			elif (case == 1):
				print("The momentum optimizer had the best accuracy at the end of training.")
			else:	
				print("RMS had the best accuracy at the end of training.")
		else: 
			if (case <= 0):
				print("Gradient descent had the best accuracy at the time of convergence.")
			elif (case == 1):
				print("The momentum optimizer had the best accuracy at the time of convergence.")
			else:	
				print("RMS had the best accuracy at the time of convergence.")
		print()
		
	
	MSE_Plot()
	CPU_Time_Plot()
	print("AFTER 100 EPOCHS")
	Max_Acc(3)
	print("AFTER CONVERGENCE")
	Max_Acc(4)
	
	
	
def c():
	num_epochs = 250
	num_batches = 50
	num_iters = 5
	es = stop()
	cb = fig6()
	MSEs = np.empty(25)
	
	def plot_avg_MSE():
		for i in range (num_iters):
			for num_hidden in range(2, 52, 2):
				model = keras.Sequential()
				model.add(layers.Dense(num_hidden, activation='tanh'))
				model.add(layers.Dense(1))
				model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(learning_rate = 0.02),
							  loss='mse',
							  metrics=[RMSE])
				model.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batches, verbose=0, callbacks=[es], validation_data=(X_val,y_val))
				mse = model.evaluate(x=X_test, y=y_test, batch_size=num_batches, verbose=0)[0]
				MSEs[(num_hidden//2) - 1] += (mse/num_iters)
				
			print("Iteration", i+1, "complete.")
		plt.plot(range(2,52,2), MSEs)
		plt.xlabel('# of Hidden Neurons')
		plt.ylabel('MSE')
		plt.show()
		
		
	
	def plot_fig6():
		model = keras.Sequential()
		model.add(layers.Dense(8, activation='tanh'))
		model.add(layers.Dense(1))
		model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(learning_rate = 0.02),
								  loss='mse',
								  metrics=[RMSE])
		hist = model.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batches, verbose=0, callbacks=[es,cb], validation_data=(X_val,y_val)).history
		
		test_MSE = []
		
		model.load_weights('1c_prev_to_convergence.h5')
		test_MSE.append(model.evaluate(x=X_test, y=y_test, batch_size=num_batches, verbose=0)[0])
		
		model.load_weights('1c_convergence.h5')
		test_MSE.append(model.evaluate(x=X_test, y=y_test, batch_size=num_batches, verbose=0)[0])
		
		model.load_weights('1c_after_convergence.h5')
		test_MSE.append(model.evaluate(x=X_test, y=y_test, batch_size=num_batches, verbose=0)[0])
		
		plt.plot(range(3), cb.losses, label='training set')
		plt.plot(range(3), cb.val_losses, label='validation set')
		plt.plot(range(3), test_MSE, label='test set')
		
		plt.hlines(y=0.02, xmin=0, xmax=2)
		plt.xlabel("2 Epochs")
		plt.ylabel("MSE")
		plt.legend()
		plt.show()
	
	def plot_fig7():
		num_epochs = 1000
		num_batches = 25
	
		model = keras.Sequential()
		model.add(layers.Dense(8, activation='tanh'))
		model.add(layers.Dense(1))
		model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(learning_rate = 0.02),
								  loss='mse',
								  metrics=[RMSE])
		model.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batches, verbose=0, callbacks=[es], validation_data=(X_val,y_val))
		y_pred_es = model.predict(X_test)
		
		model = keras.Sequential()
		model.add(layers.Dense(8, activation='tanh'))
		model.add(layers.Dense(1))
		model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(learning_rate = 0.02),
								  loss='mse',
								  metrics=[RMSE])
		model.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batches, verbose=0)
		y_pred = model.predict(X_test)
		
		plt.contour(X1, X2, y_test.reshape(9,9), colors='k')
		plt.contour(X1, X2, y_pred.reshape(9,9), colors='r')
		plt.contour(X1, X2, y_pred_es.reshape(9,9), colors='g')
		plt.xlabel('x1')
		plt.ylabel('x2')
		plt.annotate('black = target', (0.2, 0.9))
		plt.annotate('red = without early stopping', (0.2, 0.8))
		plt.annotate('green = with early stopping', (0.2, 0.7))
		plt.show()
		
	
	plot_avg_MSE()
	plot_fig6()
	plot_fig7()
	

	
u_grid = np.linspace(-1, 1, 10)
X1, X2 = np.meshgrid(u_grid, u_grid)
X_train = np.vstack((X1.flatten(), X2.flatten())).T
y_train = f_array(X_train)

u_grid = np.linspace(-1, 1, 9)
X1, X2 = np.meshgrid(u_grid, u_grid)
X_test = np.vstack((X1.flatten(), X2.flatten())).T
y_test = f_array(X_test)

X_val = np.random.uniform(low = -1, high = 1, size = (30, 2))
y_val = f_array(X_val)[np.newaxis].T

#comment this to avoid running part a
a()

#comment this to avoid running part b
b()

#comment this to avoid running part c
c()
