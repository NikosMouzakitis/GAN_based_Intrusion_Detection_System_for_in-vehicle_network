
#	author: Mouzakitis Nikolaos

#		Development of a 
#	GAN Intrusion Detection System
#
#	Reference paper:	GIDS: GAN based Intrusion Detection System for
#			               In-Vehicle Network
#
#	Details on how to acquire the datasets can be found in the reference paper.


import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf

def one_hot_vector(a):
	'''create a OHV(one-hot-vector) and return it based on 
	the hexadicimal digit string as provided argument.'''

	ret= np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
	if(a=='0'):
		ret[0]=1
	elif(a=='1'):	
		ret[1]=1
	elif(a=='2'):	
		ret[2]=1
	elif(a=='3'):	
		ret[3]=1
	elif(a=='4'):	
		ret[4]=1
	elif(a=='5'):	
		ret[5]=1
	elif(a=='6'):	
		ret[6]=1
	elif(a=='7'):	
		ret[7]=1
	elif(a=='8'):	
		ret[8]=1
	elif(a=='9'):	
		ret[9]=1
	elif(a=='a'):
		ret[10]=1
	elif(a=='b'):	
		ret[11]=1
	elif(a=='c'):	
		ret[12]=1
	elif(a=='d'):	
		ret[13]=1
	elif(a=='e'):	
		ret[14]=1
	elif(a=='f'):	
		ret[15]=1

	return ret

def create_CAN_image(s):
	'''Function which is creating the CAN Image of a given ID, by
	   subsequent calls to one_hot_vector() and returns as a numpy 
	   array the CAN Image created.'''
	
	a = one_hot_vector(s[0])
	b = one_hot_vector(s[1])
	c = one_hot_vector(s[2])

	ret=np.array([a,b,c])

	return ret

def create_64batch_Discriminator(can_image):
    '''function that creates a 64X48 batch of
    the CAN images.'''
    
    cnt=0
    for i in can_image:
        j=i.reshape(1,48)
       
        if(cnt==0):
            a=j
            cnt+=1
        else:
            a=np.concatenate((a,j),axis=0)
    return a
    
    
#start of the program
    

print("GAN Intrusion Detection System for CAN Bus case study.\nStarted.")
ids_normal_list=[]
f=open("NORMAL_IDS.txt","r")
lines=f.readlines()
f.close()

for i in lines:
	i=i[2:len(i)-1]	
	ids_normal_list.append(i)

ids_normal_list.pop()
#now we have stored in ids_list all the IDS of the normal run.

#create all the normal CAN_IMAGES(3*16 arrays)
normal_CAN_Images=[]

for i in ids_normal_list:
	normal_CAN_Images.append(create_CAN_image(i))

#create a batch of 64X16*3 2d image
#batchDiscriminator = create_64batch_Discriminator(normal_CAN_Images[0:64])
#plt.figure(1)
#plt.imshow(batchDiscriminator,cmap="binary")
#plt.title("A batch for the Discriminator D(64X48)")

normal_train_batchD = [] #training list for batches of normal
count_normal_train_batchD = 0

for i in range(0,10816,64):
    normal_train_batchD.append(create_64batch_Discriminator(normal_CAN_Images[i:i+64]))
    count_normal_train_batchD += 1
    
