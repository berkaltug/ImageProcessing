    data=[]
	data.append([5.1,3.5,1.4,0.2, "setosa"])
	data.append([4.9,3.0,1.4,0.2, "setosa"])
	data.append([4.7,3.2,1.3,0.2, "setosa"])
	data.append([4.6,3.1,1.5,0.2, "setosa"])
	data.append([6.4,3.2,4.5,1.5, "versicolor"])
	data.append([6.9,3.1,4.9,1.5, "versicolor"])
	data.append([5.5,2.3,4.0,1.3, "versicolor"])
	data.append([6.5,2.8,4.6,1.5, "versicolor"])
	data.append([7.1,3.0,5.9,2.1, "virginica"])
	data.append([7.6,3.0,6.6,2.1, "virginica"])
	data.append([7.3,2.9,6.3,1.8, "virginica"])
	data.append([6.5,3.0,5.8,2.2, "virginica"])
	
	
	# In[3]:
	
	
	def uzaklik_hesap(inp,data):
	    uzaklik=0
	    for i in range(4):
	            uzaklik=uzaklik+((inp[i]-data[i])**2)
	            uzaklik=uzaklik**.5
	    return uzaklik
	
	
	# In[4]:
	
	
	def uzakliklari_bul(inp):
	    benzerlik_dizisi1=[]
	    sayi=len(data)
	    for i in range (sayi):
	        benzerlik=uzaklik_hesap(inp,data[i])
	        benzerlik_dizisi1.append(benzerlik)
	    return benzerlik_dizisi1
	
	
	# In[5]:
	
	
	def siniflari_bul(benzerlik_dizisi, k):
	    siniflar=[]
	    yedek=[]
	    for i in range (12):
	        yedek.append(benzerlik_dizisi[i])
	    ksayac=0
	    
	    for j in range (k):
	        for i in range (12):
	            if (ksayac!=k):
	                if(benzerlik_dizisi[i]==min(yedek)):
	                    siniflar.append(data[i][4])
	                    print(i)
	                    ksayac=ksayac+1
	                    yedek.remove(min(yedek))
	    return siniflar
	
	
	# In[6]:
	
	
	def with_knn_find_class(input, k):
	    benzerlik_dizi=uzakliklari_bul(input)
	    print(benzerlik_dizi)
	    siniflar=siniflari_bul(benzerlik_dizi,k)
	    print(siniflar)
	    class_1=0
	    class_2=0
	    class_3=0
	    for i in range (k):
	        if (siniflar[i]=="setosa"):
	            class_1=class_1+1
	        if (siniflar[i]=="versicolor"):
	            class_2=class_2+1
	        if (siniflar[i]=="virginica"):
	            class_3=class_3+1
	    if (class_1>class_2 and class_1>class_3):
	        myclass="setosa"
	    if (class_2>class_1 and class_2>class_3):
	        myclass="versicolor"
	    if (class_3>class_1 and class_3>class_2):
	        myclass="virginica"
	    return myclass
	
	
	# In[7]:
	
	
	sinifsiz=[6.4,3.2,4.5,1.5]
	print(with_knn_find_class(sinifsiz,3))






KMEANS


	from mpl_toolkits.mplot3d import Axes3D
	from sklearn import datasets
	from sklearn.decomposition import PCA
import  matplotlib.pylpot as plt
	
	# import some data to play with
	iris = datasets.load_iris()
	X = iris.data[:, :2]  # we only take the first two features.
	y = iris.target
	
	from mpl_toolkits import mplot3d
	get_ipython().run_line_magic('matplotlib', 'inline')
	import numpy as np
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	
	xdata=iris.data[:,0]
	ydata=iris.data[:,2]
	zdata=iris.data[:,3]
	label=iris.target
	
	ax.scatter3D(xdata[100:149], ydata[100:149], zdata[100:149],marker='o')
	ax.scatter3D(xdata[50:99], ydata[50:99], zdata[50:99],marker='x')
	ax.scatter3D(xdata[0:49], ydata[0:49], zdata[0:49],marker='x')
	
	
	# In[2]:
	
	
	X.shape
	
	
	# In[7]:
	
	
	iris.data.shape,iris.target.shape
	
	
	# In[14]:
	
	
	for i in range(0,3):
	    print(iris.data[i],iris.target[i])
	for i in range(50,53):
	    print(iris.data[i],iris.target[i])
	for i in range(100,103):
	    print(iris.data[i],iris.target[i])
	
	
	# In[48]:
	
	
	
	
	
	# In[138]:
	
	
	#5.1 3.5 1.4 0.2
	def get_mu_s():
	    mu_0=[5,2,0]
	    mu_1=[4,4,0]
	    mu_2=[3,2,5]
	    
	    return mu_0,mu_1,mu_2
	def get_distance(mu,point):
	    x=mu[0]-point[0]
	    y=mu[1]-point[1]
	    z=mu[2]-point[2]
	
	    return (x**2+y**2+z**2)**0.5
	def get_class_for_one_instance(flower):
	    mu_s=get_mu_s()
	    d_0=get_distance(mu_s[0],flower)
	    d_1=get_distance(mu_s[1],flower)
	    d_2=get_distance(mu_s[2],flower)
	    if ((d_0 < d_1) and (d_0<d_2)):
	        return "0"
	    if ((d_1 < d_0) and (d_1<d_2)):
	        return "1"
	    if ((d_2 < d_1) and (d_2<d_0)):
	        return "2"
	def my_f_1(s_1=125):
	    x=iris.data[s_1][0]
	    y=iris.data[s_1][2]
	    z=iris.data[s_1][3]
	    my_f_1=[x,y,z]
	    r=get_class_for_one_instance(my_f_1)
	    print(r)
	for i in range(150):
	    my_f_1(i)
	
	
	# In[122]:
	
	
	def get_flower(i):
	    x=iris.data[i][0]
	    y=iris.data[i][2]
	    z=iris.data[i][3]
	    return [x,y,z]
	     
	
	
	# In[139]:
	
	
	def update_mu():
	    hata="yok"
	    mu_0_counter=0.0001
	    mu_0_sum=0
	    
	    mu_1_counter=0.0001
	    mu_1_sum=0
	    
	    mu_2_counter=0.0001
	    mu_2_sum=0
	    
	    c_1=[]
	    c_2=[]
	    c_3=[]
	        
	    for i in range(150):
	        my_flower_data=get_flower(i)
	        
	        f_class=get_class_for_one_instance(my_flower_data)
	        hata="var"
	
	        #print(f_class)
	        if(f_class=="0"):
	            c_1.append(my_flower_data)    
	        if(f_class=="1"):
	            c_2.append(my_flower_data)
	            
	        if(f_class=="2"):
	            c_2.append(my_flower_data)
	            
	    
	    
	    return c_1,c_2,c_3
	
	
	# In[143]:
	
	
	c_1,c_2,c_3=update_mu()
	
	
	# In[144]:
	
	
	len(c_1),len(c_2),len(c_3)
	

BAYES


	def generate_data():
	    
	    import math
	    import random
	    my_classes={}
	
	    kac_sinif_olsun=5
	    kac_deger_atansin=2000
	    araliklar_ne_olsun=[random.randint(5,50) for x in range(kac_sinif_olsun)]
	    
	    for s in range(kac_sinif_olsun):
	        class_name=str(s)+"_class"
	        my_classes[class_name]={}
	        aralik=araliklar_ne_olsun[s]
	        my_classes[class_name]["data"]=[ random.random()*(aralik+s*5)+(s*5) for x in range(kac_deger_atansin)]
	
	        #ortalama hesaplama ve sinifin "mean" alanina ekleme
	        a=sum(my_classes[class_name]["data"])
	        b=len(my_classes[class_name]["data"])    
	        my_classes[class_name]["mean"]=a/b
	        my_classes[class_name]["class_id"]=s
	
	
	        #varyans ve standartsapma hesabi 1. satir (data-mean)^2 --- 2.satir newline_degerlertoplami/elemansayisi
	        new_line=[(x-my_classes[class_name]["mean"])**2 for x in my_classes[class_name]["data"] ]
	        my_var=sum(new_line)/(len(new_line)-1)
	
	        my_classes[class_name]["var"]=math.floor(my_var)
	        my_classes[class_name]["std"]=math.sqrt(my_var)
	
	    #my_classes.keys()
	    return my_classes
	    ###--*end*-- sinif-dictionary olusturulan ve sinifa deger atanan bolum
	# generate_data()
	#normal dagilim hesabi
	def calculate_probability_one_value(x, mean, stdev):
	    import math
	    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
	def define_probability_for_data(data): # # _and_plot(data):
	    
	    my_classes=data
	    kac_sinif_olsun=len(my_classes.keys())
	    
	    for s in range(kac_sinif_olsun):
	        class_name=str(s)+"_class"
	
	        #sadece 0_class icin garfik cizimi
	        mean=my_classes[class_name]["mean"]
	        stdev=my_classes[class_name]["std"] 
	        values_raw=my_classes[class_name]["data"]
	        my_classes[class_name]["prob"]=[]
	     
	        for i in values_raw:
	            my_classes[class_name]["prob"].append(calculate_probability_one_value(i,mean,stdev))
	            
	    return my_classes
	        
	def plot_data(t,data):
	    d=data
	    import matplotlib.pyplot as plt
	    fig = plt.figure()
	    ax = plt.subplot(111)
	    for c in d.keys():
	        # plt.plot(d[c]['data'],d[c]['prob'],'.',label='cdfsdf')
	        
	        ax.plot(d[c]['data'],d[c]['prob'],'.',label=c)
	
	
	    r=calculate_test_probability(t,data)
	    x1=t
	    x2=t
	    y1=0
	    p_s=[y for (x,y) in r]
	    m=p_s.index(max(p_s))
	    print(" bu veri ",m," sınıfına ait ")
	    y2=r[m][1]
	    plt.plot([x1,x2],[y1,y2],"k")
	    plt.legend()
	    plt.show()
	def calculate_test_probability(test_value,d):
	    test_value_probabilities=[]
	    for c in d.keys():
	        mean=d[c]['mean']
	        stdev=d[c]['std']
	        test_value_probabilities.append((c,calculate_probability_one_value(test_value, mean, stdev)))
	    return test_value_probabilities
	def my_run(t):
	    data=generate_data()
	    data=define_probability_for_data(data)
	    calculate_test_probability(t,data)
	    plot_data(t,data) 
	    return calculate_test_probability(50,data)
	
	
	# In[4]:
	
	
	my_run(40)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

iris= datasets.load_iris()
x=iris.data[:,:2]
y=iris.target

for i in range(0,3):
    print(iris.data[i],iris.target[i])
for i in range(50,53):
    print(iris.data[i],iris.target[i])
for i in range(100,103):
    print(iris.data[i],iris.target[i])


from mpl_toolkits import mplot3d
%matplotlib inline 
import numpy as np
fig=plt.figure()
ax=plt.axes(projection='3d')


xdata=iris.data[:,0]
ydata=iris.data[:,2]
zdata=iris.data[:,3]
label=iris.target

ax.scatter3D(xdata[100:149],ydata[100:149],zdata[100:149],marker='o')
ax.scatter3D(xdata[50:99],ydata[50:99],zdata[50:99],marker='x')
ax.scatter3D(xdata[0:49],ydata[0:49],zdata[0:49],marker='x')



def get_mu_s():
    mu_0=[1,2,3]
    mu_1=[1,-2,5]
    mu_2=[1,5,8]
    
    return mu_0,mu_1,mu_2


def get_distance(mu,point):
    x=mu[0]-point[0]
    y=mu[1]-point[1]
    z=mu[2]-point[2]
    
    return (x**2+y**2+z**2)**0.5


my_flower_1=iris.data[0]
m_1=get_mu_s()[0]
get_distance(my_flower_1,m_1)


4.6497311750250674

def get_class_for_one_instance(flower):
    mu_s=get_mu_s()
    d_0=get_distance(mu_s[0],flower)
    d_1=get_distance(mu_s[1],flower)
    d_2=get_distance(mu_s[2],flower)
    

    if((d_0 < d_1) and (d_0 < d_2)):
        return "10"
    if((d_1 < d_0) and (d_1 < d_2)):
        return "20"
    if((d_2 < d_1) and (d_2 < d_0)):
        return "30"
    


def my_f_1(s_1=125):


    x=iris.data[s_1][0]
    y=iris.data[s_1][2]
    z=iris.data[s_1][3]
    my_f_1=[x,y,z]
    r=get_class_for_one_instance(my_f_1)
    print(r)
for i in range(150):
    my_f_1(i)

10

10

10

10

10

10

10

10



def get_flower(i):
    x=iris.data[i][0]
    y=iris.data[i][2]
    z=iris.data[i][3]
    return [x,y,z]



def update_mu():
    hata="none"
    mu_0_counter=0.0001
    mu_0_sum=0
    
    mu_1_counter=0.0001
    mu_1_sum=0
    
    mu_2_counter=0.0001
    mu_2_sum=0
    for i in range(150):
        my_flower_data=get_flower(i)
        
        f_class=get_class_for_one_instance(my_flower_data)
        hata="exits"
        c_1=[]
        c_2=[]
        c_3=[]
        
        if(f_class=="0"):
            c_1.append(my_flower_data)
            
        if(f_class=="1"):
            c_2.append(my_flower_data)
            
        if(f_class=="2"):
            c_2.append(my_flower_data)
            
        
    
    return c_1,c_2,c_3
epoch=15
update_mu():


